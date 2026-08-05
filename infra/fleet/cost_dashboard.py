#!/usr/bin/env python3
"""Local dashboard for RunsOn Fleet GPU CI cost and timing.

Serves an interactive page backed by a local JSON API:

  GET /api/runs                     qualified runs created in the last 7d
  GET /api/run?id=<run-id>          per-leg timing + spot cost for one run
  GET /api/account?start&end&gran   account usage/cost for a date range
  POST /api/account/refresh?...     force a Cost Explorer refresh

Per-run data is free to fetch (GitHub API + ec2:DescribeSpotPriceHistory
+ cloudtrail:LookupEvents carry no charge). Only the account panels touch
Cost Explorer, billed at $0.01 per GetCostAndUsage request. Hourly usage,
run qualification, and settled run detail are retained in separate
transactional SQLite stores that outlive the process; a fully priced and
terminated run is served from that store with no GitHub or AWS request.
The workflow list is refreshed at most once per minute, and
completed qualification decisions are reused until they age out.
Hourly usage is retained with per-hour confirmation.
Non-zero gross service cost confirms an hour immediately; zero or missing
cost confirms only after a successful observation at least 48 hours after
that hour began. Automatic refresh targets the most recent recoverable UTC
day with no confirmed hours. Failed attempts retry after 15 minutes. Force
fetch bypasses the automatic trigger, with its own five-minute limit.

The loopback server validates Host and Origin, requires a per-process
token for API requests, accepts paid force refreshes only by POST, and
serves a restrictive security policy. Missing run-cost telemetry remains
unknown throughout the API and UI rather than being coerced to zero. A
leg whose runner died mid-step never gets an archived job log; it prices
from its CloudTrail launch record instead, and reports its queue wait as
unknown.

Each leg is read in the region its own instance ran in. Each leg also
carries the spot market it was priced in, Windows and Linux being
separate markets for one instance type.

Run:  python infra/fleet/cost_dashboard.py   (opens http://localhost:8787)
Needs: gh authenticated to the repo, the cubie-fleet AWS profile.
"""

import argparse
import hmac
import json
import os
import re
import secrets
import sqlite3
import subprocess
import threading
import webbrowser
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import uuid4

REPO = "cubiepy/cubie"
WORKFLOW = "ci_cuda_tests.yml"
PROFILE = "cubie-fleet"
REGION = "us-east-2"
# Regions the fleet has run in; searched when a leg's own is unknown.
HISTORICAL_REGIONS = ("ap-southeast-2",)
SEARCH_REGIONS = (REGION, *HISTORICAL_REGIONS)
EC2_COMPUTE = "Amazon Elastic Compute Cloud - Compute"
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".dashboard-cache"
USAGE_DB = CACHE_DIR / "usage.sqlite3"
RUN_DB = CACHE_DIR / "runs.sqlite3"
DETAIL_DB = CACHE_DIR / "details.sqlite3"
RUN_LOOKBACK = timedelta(days=7)
RUN_LIST_TTL = timedelta(seconds=60)
RUN_SCAN_LEASE = timedelta(minutes=10)
REFRESH_LEASE = timedelta(minutes=10)
AUTO_ATTEMPT_THROTTLE = timedelta(minutes=15)
AUTO_REFRESH_CUTOFF = timedelta(minutes=15)
FORCE_RATE_LIMIT = timedelta(minutes=5)
COST_EXPLORER_HOURLY_RETENTION = timedelta(days=14)
ZERO_COST_CONFIRMATION_AGE = timedelta(hours=48)
CONFIRMED_OVERLAP = timedelta(hours=12)
LOG_ARCHIVE_GRACE = timedelta(minutes=5)
LAUNCH_LOOKBACK = timedelta(hours=1)
MAX_DAILY_RANGE_DAYS = 3660
MAX_HOURLY_RANGE_DAYS = 366
USAGE_SCHEMA_VERSION = 3
USAGE_QUERY_VERSION = "ce-usage-v1"
RUN_CACHE_SCHEMA_VERSION = 1
# Covers the stored payload shape as well as the tables.
DETAIL_CACHE_SCHEMA_VERSION = 1
API_TOKEN = secrets.token_urlsafe(32)
TOKEN_HEADER = "X-Cubie-Dashboard-Token"

# The AWS CLI encodes its own stdout with the process locale; on a Windows
# cp1252 console it dies rendering the U+202F characters CloudTrail events
# carry. Force UTF-8 for every subprocess.
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

_CE_LOCK = threading.Lock()


# ------------------------------------------------------------------ shells
class GitHubError(RuntimeError):
    """A failed gh invocation, carrying any HTTP status it reported."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


_GH_STATUS = re.compile(r"HTTP (\d{3})")


def gh(*args):
    out = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_ENV,
    )
    if out.returncode != 0:
        # gh reports a REST failure as either "gh: HTTP 404" or
        # "gh: Not Found (HTTP 404)" depending on whether the response
        # body was JSON; both carry the status in the same form.
        status_match = _GH_STATUS.search(out.stderr)
        raise GitHubError(
            f"gh {' '.join(args)} failed:\n{out.stderr}",
            int(status_match.group(1)) if status_match else None,
        )
    return out.stdout


def aws(*args):
    """Return (ok, parsed-json-or-stderr). ok=False on AccessDenied."""
    out = subprocess.run(
        ["aws", *args, "--profile", PROFILE, "--output", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_ENV,
    )
    if out.returncode != 0:
        msg = out.stderr
        if "AccessDenied" in msg or "not authorized" in msg:
            return False, msg.strip()
        raise RuntimeError(f"aws {' '.join(args)} failed:\n{msg}")
    return True, json.loads(out.stdout or "null")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def now_utc():
    return datetime.now(timezone.utc)


def _step_name(name):
    # Collapse names differing only by matrix axis (Python version, cubie
    # extra, action SHA) so a step is one series, not one per leg flavour.
    name = re.sub(r"@[0-9a-f]{7,}", "", name)
    name = re.sub(r"\s*\(dev[\w-]*\)", "", name)
    name = re.sub(r"\s+3\.\d+\b", "", name)
    return name.strip()


# ------------------------------------------------------------------ stores
class SQLiteStore:
    """Base for the dashboard's transactional local SQLite caches."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError as exc:
            # WAL persists once a peer initializer enables it. Continue
            # to the busy-timeout-protected transaction during that race.
            if "locked" not in str(exc).lower():
                connection.close()
                raise
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        raise NotImplementedError

    @staticmethod
    def _set_metadata(connection, key, value):
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


# ------------------------------------------------------------------ github
class RunStore(SQLiteStore):
    """Transactional cache for the qualified seven-day run snapshot."""

    def _initialize(self):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            schema_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            if schema_version not in (0, RUN_CACHE_SCHEMA_VERSION):
                raise RuntimeError(
                    "unsupported run-cache schema "
                    f"{schema_version}; expected {RUN_CACHE_SCHEMA_VERSION}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY CHECK (run_id > 0),
                    created_at TEXT NOT NULL,
                    created_epoch REAL NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    conclusion TEXT,
                    qualified INTEGER NOT NULL
                        CHECK (qualified IN (0, 1)),
                    started_leg_count INTEGER NOT NULL
                        CHECK (started_leg_count >= 0),
                    terminal INTEGER NOT NULL
                        CHECK (terminal IN (0, 1)),
                    inspected_at TEXT NOT NULL,
                    seen_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                f"PRAGMA user_version = {RUN_CACHE_SCHEMA_VERSION}"
            )

    def acquire_scan_lease(self, now):
        """Acquire the persisted list-refresh lease and attempt slot."""
        owner = uuid4().hex
        cutoff_epoch = (now - RUN_LOOKBACK).timestamp()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM runs WHERE created_epoch < ?", (cutoff_epoch,)
            )
            lease_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'scan_lease'"
            ).fetchone()
            if lease_row:
                lease = json.loads(lease_row[0])
                if ts(lease["until"]) > now:
                    connection.commit()
                    return "in_progress", None
            attempt_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'last_list_attempt'"
            ).fetchone()
            if attempt_row and now - ts(attempt_row[0]) < RUN_LIST_TTL:
                connection.commit()
                return "throttled", None
            self._set_metadata(
                connection, "last_list_attempt", now.isoformat()
            )
            lease = {
                "owner": owner,
                "until": (now + RUN_SCAN_LEASE).isoformat(),
            }
            self._set_metadata(connection, "scan_lease", json.dumps(lease))
            connection.commit()
        return "acquired", owner

    def release_scan_lease(self, owner):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'scan_lease'"
            ).fetchone()
            if row and json.loads(row[0]).get("owner") == owner:
                connection.execute(
                    "DELETE FROM metadata WHERE key = 'scan_lease'"
                )

    def cached_decisions(self, cutoff, now):
        """Return reusable qualification decisions in the live window."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT run_id, qualified, started_leg_count, terminal
                FROM runs
                WHERE created_epoch >= ? AND created_epoch <= ?
                """,
                (cutoff.timestamp(), now.timestamp()),
            ).fetchall()
        return {
            row[0]: {
                "qualified": bool(row[1]),
                "started_leg_count": row[2],
                "terminal": bool(row[3]),
            }
            for row in rows
        }

    def commit_scan(self, records, cutoff, completed_at, owner):
        """Atomically publish a complete workflow-list scan."""
        marker = completed_at.isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'scan_lease'"
            ).fetchone()
            lease = json.loads(lease_row[0]) if lease_row else {}
            if (
                lease.get("owner") != owner
                or ts(lease["until"]) <= completed_at
            ):
                raise RuntimeError(
                    "run scan lease expired or ownership was lost"
                )
            for record in records:
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, created_at, created_epoch, title,
                        status, conclusion, qualified,
                        started_leg_count, terminal, inspected_at,
                        seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        created_at = excluded.created_at,
                        created_epoch = excluded.created_epoch,
                        title = excluded.title,
                        status = excluded.status,
                        conclusion = excluded.conclusion,
                        qualified = excluded.qualified,
                        started_leg_count = excluded.started_leg_count,
                        terminal = excluded.terminal,
                        inspected_at = excluded.inspected_at,
                        seen_at = excluded.seen_at
                    """,
                    (
                        record["id"],
                        record["created_at"],
                        record["created_epoch"],
                        record["title"],
                        record["status"],
                        record["conclusion"],
                        int(record["qualified"]),
                        record["started_leg_count"],
                        int(record["terminal"]),
                        record["inspected_at"],
                        marker,
                    ),
                )
            connection.execute(
                """
                DELETE FROM runs
                WHERE created_epoch < ? OR seen_at != ?
                """,
                (cutoff.timestamp(), marker),
            )
            self._set_metadata(connection, "last_list_refresh", marker)
            connection.execute("DELETE FROM metadata WHERE key = 'scan_lease'")
            connection.commit()

    def has_snapshot(self):
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM metadata
                WHERE key = 'last_list_refresh'
                """
            ).fetchone()
        return row is not None

    def qualified_runs(self, cutoff, now):
        """Return the qualified live snapshot newest first."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT run_id, created_at, title, status, conclusion,
                       started_leg_count
                FROM runs
                WHERE qualified = 1
                  AND created_epoch >= ? AND created_epoch <= ?
                ORDER BY created_epoch DESC, run_id DESC
                """,
                (cutoff.timestamp(), now.timestamp()),
            ).fetchall()
        return [
            {
                "id": row[0],
                "created_at": row[1],
                "title": row[2],
                "status": row[3],
                "conclusion": row[4],
                "started_leg_count": row[5],
            }
            for row in rows
        ]

    def qualified_run(self, run_id, cutoff, now):
        """Return one qualified live run, or None outside the snapshot."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT run_id, created_at, title, status, conclusion,
                       started_leg_count
                FROM runs
                WHERE run_id = ? AND qualified = 1
                  AND created_epoch >= ? AND created_epoch <= ?
                """,
                (run_id, cutoff.timestamp(), now.timestamp()),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "title": row[2],
            "status": row[3],
            "conclusion": row[4],
            "started_leg_count": row[5],
        }


def _positive_integer(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError(f"{field} must be a positive integer")
    if parsed < 1:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _aware_timestamp(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a timestamp string")
    try:
        parsed = ts(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _optional_string(value, field):
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")


def _paginated_items(endpoint, list_key):
    pages = json.loads(gh("api", "--paginate", "--slurp", endpoint))
    if not isinstance(pages, list) or not pages:
        raise ValueError("paginated GitHub response must be a nonempty list")
    items = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"GitHub page {page_index} must be an object")
        if list_key not in page or not isinstance(page[list_key], list):
            raise ValueError(
                f"GitHub page {page_index} must contain a {list_key} list"
            )
        items.extend(page[list_key])
    return items


def _validated_workflow_run(run, item_index):
    if not isinstance(run, dict):
        raise ValueError(f"workflow run {item_index} must be an object")
    normalized = dict(run)
    normalized["id"] = _positive_integer(
        run.get("id"), f"workflow run {item_index} id"
    )
    _aware_timestamp(
        run.get("created_at"),
        f"workflow run {normalized['id']} created_at",
    )
    status = run.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(
            f"workflow run {normalized['id']} status must be a nonempty string"
        )
    _optional_string(
        run.get("conclusion"),
        f"workflow run {normalized['id']} conclusion",
    )
    for field in ("display_title", "name"):
        if field in run:
            _optional_string(
                run[field],
                f"workflow run {normalized['id']} {field}",
            )
    return normalized


def _validated_job(job, item_index):
    if not isinstance(job, dict):
        raise ValueError(f"workflow job {item_index} must be an object")
    normalized = dict(job)
    normalized["id"] = _positive_integer(
        job.get("id"), f"workflow job {item_index} id"
    )
    if job.get("started_at") is not None:
        _aware_timestamp(
            job["started_at"],
            f"workflow job {normalized['id']} started_at",
        )
    labels = job.get("labels")
    if labels is not None:
        if not isinstance(labels, list) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ValueError(
                f"workflow job {normalized['id']} labels "
                "must be a list of strings or null"
            )
    runner_name = job.get("runner_name")
    if runner_name is not None and not isinstance(runner_name, str):
        raise ValueError(
            f"workflow job {normalized['id']} runner_name "
            "must be a string or null"
        )
    return normalized


def _deduplicate_items(items, validator, item_name):
    unique = {}
    for item_index, item in enumerate(items):
        normalized = validator(item, item_index)
        item_id = normalized["id"]
        prior = unique.get(item_id)
        if prior is not None and prior != normalized:
            raise ValueError(f"conflicting duplicate {item_name} id {item_id}")
        unique[item_id] = normalized
    return list(unique.values())


def _workflow_runs(cutoff, now):
    query = urlencode(
        {
            "per_page": 100,
            "created": f">={cutoff.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        }
    )
    raw_runs = _paginated_items(
        f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs?{query}",
        "workflow_runs",
    )
    validated = _deduplicate_items(
        raw_runs, _validated_workflow_run, "workflow run"
    )
    runs = [
        run
        for run in validated
        if cutoff
        <= _aware_timestamp(run["created_at"], "workflow run created_at")
        <= now
    ]
    return sorted(
        runs,
        key=lambda run: (ts(run["created_at"]), int(run["id"])),
        reverse=True,
    )


def _fetch_jobs(run_id):
    raw_jobs = _paginated_items(
        f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
        "jobs",
    )
    return _deduplicate_items(raw_jobs, _validated_job, "workflow job")


def _started_gpu_leg_count(jobs):
    return sum(
        1
        for job in jobs
        if job.get("started_at")
        and any(
            str(label).startswith("runs-on/fleet=gpu-")
            for label in (job.get("labels") or [])
        )
        and re.search(
            r"runs-on--i-[0-9a-f]+--",
            job.get("runner_name") or "",
        )
    )


def _run_record(run, decision, inspected_at):
    title = str(
        run.get("display_title") or run.get("name") or f"Run {run['id']}"
    ).strip()
    if not title:
        title = f"Run {run['id']}"
    return {
        "id": int(run["id"]),
        "created_at": run["created_at"],
        "created_epoch": ts(run["created_at"]).timestamp(),
        "title": title[:200],
        "status": str(run["status"]),
        "conclusion": run.get("conclusion"),
        "qualified": decision["qualified"],
        "started_leg_count": decision["started_leg_count"],
        "terminal": run.get("status") == "completed",
        "inspected_at": inspected_at.isoformat(),
    }


def fetch_runs(store=None, now=None):
    """Return all qualified workflow runs created in the last seven days."""
    store = store or RunStore(RUN_DB)
    scan_started = now or now_utc()
    cutoff = scan_started - RUN_LOOKBACK
    lease_status, owner = store.acquire_scan_lease(scan_started)
    if lease_status != "acquired":
        return store.qualified_runs(cutoff, scan_started)
    try:
        workflow_runs = _workflow_runs(cutoff, scan_started)
        cached = store.cached_decisions(cutoff, scan_started)
        records = []
        for run in workflow_runs:
            run_id = int(run["id"])
            terminal = run.get("status") == "completed"
            prior = cached.get(run_id)
            if prior and prior["terminal"] and terminal:
                decision = prior
            elif terminal and run.get("conclusion") in {
                "skipped",
                "action_required",
            }:
                decision = {
                    "qualified": False,
                    "started_leg_count": 0,
                    "terminal": True,
                }
            else:
                count = _started_gpu_leg_count(_fetch_jobs(run_id))
                decision = {
                    "qualified": count > 0,
                    "started_leg_count": count,
                    "terminal": terminal,
                }
            records.append(_run_record(run, decision, scan_started))
        completed_at = now if now is not None else now_utc()
        store.commit_scan(records, cutoff, completed_at, owner)
    except Exception:
        store.release_scan_lease(owner)
        if store.has_snapshot():
            return store.qualified_runs(cutoff, scan_started)
        raise
    return store.qualified_runs(cutoff, scan_started)


def fetch_legs(run_id):
    """Return completed GPU legs and whether the run is safe to cache."""
    legs = []
    jobs = _fetch_jobs(run_id)
    gpu_jobs = [
        job
        for job in jobs
        if any(
            str(label).startswith("runs-on/fleet=gpu-")
            for label in (job.get("labels") or [])
        )
    ]
    for j in gpu_jobs:
        m = re.search(r"runs-on--(i-[0-9a-f]+)--", j.get("runner_name") or "")
        if not (m and j.get("started_at") and j.get("completed_at")):
            continue
        label_match = re.search(r"\((.+)\)$", j["name"])
        legs.append(
            {
                "label": label_match.group(1) if label_match else j["name"],
                "job_id": j["id"],
                "instance_id": m.group(1),
                "job_start": ts(j["started_at"]),
                "job_end": ts(j["completed_at"]),
                # A step GitHub reports as started but never completed
                # is one the runner was still running when it died, so
                # the step list stops short of the job's own end.
                "truncated": any(
                    s["started_at"] and not s["completed_at"]
                    for s in j["steps"]
                ),
                "steps": [
                    {
                        "name": s["name"],
                        "start": ts(s["started_at"]),
                        "end": ts(s["completed_at"]),
                    }
                    for s in j["steps"]
                    if s["started_at"] and s["completed_at"]
                ],
            }
        )
    run = json.loads(gh("api", f"repos/{REPO}/actions/runs/{run_id}"))
    settled = (
        bool(gpu_jobs)
        and len(legs) == len(gpu_jobs)
        and run.get("status") == "completed"
    )
    return legs, settled


def fetch_log(job_id):
    """Return a job's archived log, or None when GitHub has none.

    A runner that dies mid-step leaves its job marked failed with no log
    blob ever written, and the endpoint answers 404 forever. The absence
    is never cached: a log archived moments after a job completes would
    otherwise stay missing for the life of the process.
    """
    cache = CACHE_DIR / "logs" / f"job-{job_id}.log"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    try:
        text = gh("api", f"repos/{REPO}/actions/jobs/{job_id}/logs")
    except GitHubError as exc:
        if exc.status == 404:
            return None
        raise
    text = text.replace("\r", "")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


_BANNER = {
    "instance_type": re.compile(r"│ InstanceType\s+│ (\S+)"),
    "az": re.compile(r"│ AvailabilityZone\s+│ (\S+)"),
    "platform": re.compile(r"│ Platform\s+│ (\S+)"),
}
_PHASE = re.compile(r"│ (\d{4}-\d\d-\d\dT[\d:]+Z) │ ([a-z0-9-]+)\s+│ (\d+)ms")


def parse_log(text):
    """Return the RunsOn banner and phase fields a job log carries.

    With no log at all every field is unknown, including the queue wait:
    a log that omits the instance-pending phase reports a warm instance
    that waited zero, which a missing log cannot claim.
    """
    if text is None:
        return {
            **{key: None for key in _BANNER},
            "job_scheduled": None,
            "running_start": None,
            "wait_ms": None,
            "log_known": False,
        }
    info = {
        k: (m.group(1) if (m := rx.search(text)) else None)
        for k, rx in _BANNER.items()
    }
    phases = [(ts(a), b, int(c)) for a, b, c in _PHASE.findall(text)]
    by = {name: (t, ms) for t, name, ms in phases}
    info["job_scheduled"] = by.get("job-scheduled", (None,))[0]
    if "instance-pending" in by:
        info["running_start"], info["wait_ms"] = by["instance-pending"]
    else:
        info["running_start"], info["wait_ms"] = None, 0
    info["log_known"] = True
    return info


# -------------------------------------------------------- durable detail
class DetailStore(SQLiteStore):
    """Durable cache for settled run detail and immutable AWS reads.

    Holds whole run payloads, achieved spot prices, and terminated
    instances' launch/termination records: all final once observed. An
    unknown answer is never stored, so it is retried on the next
    request. Nothing is evicted by age. A schema or payload-shape
    change discards the file rather than migrating it.
    """

    def _initialize(self):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            schema_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            if schema_version != DETAIL_CACHE_SCHEMA_VERSION:
                for table in ("run_details", "instances", "spot_prices"):
                    connection.execute(f"DROP TABLE IF EXISTS {table}")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_details (
                    run_id INTEGER PRIMARY KEY CHECK (run_id > 0),
                    payload TEXT NOT NULL,
                    stored_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS instances (
                    instance_id TEXT PRIMARY KEY,
                    region TEXT NOT NULL,
                    launch TEXT,
                    termination TEXT NOT NULL,
                    stored_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS spot_prices (
                    instance_type TEXT NOT NULL,
                    az TEXT NOT NULL,
                    product TEXT NOT NULL,
                    at TEXT NOT NULL,
                    price REAL NOT NULL,
                    stored_at TEXT NOT NULL,
                    PRIMARY KEY (instance_type, az, product, at)
                )
                """
            )
            connection.execute(
                f"PRAGMA user_version = {DETAIL_CACHE_SCHEMA_VERSION}"
            )

    def run_detail(self, run_id):
        """Return a stored settled run payload, or None."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM run_details WHERE run_id = ?",
                (int(run_id),),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def store_run_detail(self, run_id, payload):
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO run_details(run_id, payload, stored_at)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    payload = excluded.payload,
                    stored_at = excluded.stored_at
                """,
                (
                    int(run_id),
                    json.dumps(payload),
                    now_utc().isoformat(),
                ),
            )

    def instance_history(self, instance_id):
        """Return a stored (launch, termination) pair, or None."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT launch, termination FROM instances
                WHERE instance_id = ?
                """,
                (instance_id,),
            ).fetchone()
        if row is None:
            return None
        launch = json.loads(row[0]) if row[0] else None
        if launch is not None and launch.get("launched_at"):
            launch["launched_at"] = ts(launch["launched_at"])
        return launch, ts(row[1])

    def store_instance_history(self, instance_id, region, launch, term):
        stored = dict(launch) if launch else None
        if stored and stored.get("launched_at"):
            stored["launched_at"] = stored["launched_at"].isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO instances(
                    instance_id, region, launch, termination, stored_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(instance_id) DO UPDATE SET
                    region = excluded.region,
                    launch = excluded.launch,
                    termination = excluded.termination,
                    stored_at = excluded.stored_at
                """,
                (
                    instance_id,
                    region,
                    json.dumps(stored) if stored else None,
                    term.isoformat(),
                    now_utc().isoformat(),
                ),
            )

    def spot_price(self, instance_type, az, product, at):
        """Return a stored achieved spot price, or None."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT price FROM spot_prices
                WHERE instance_type = ? AND az = ? AND product = ?
                  AND at = ?
                """,
                (instance_type, az, product, at),
            ).fetchone()
        return row[0] if row else None

    def store_spot_price(self, instance_type, az, product, at, price):
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO spot_prices(
                    instance_type, az, product, at, price, stored_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(instance_type, az, product, at) DO UPDATE SET
                    price = excluded.price,
                    stored_at = excluded.stored_at
                """,
                (
                    instance_type,
                    az,
                    product,
                    at,
                    float(price),
                    now_utc().isoformat(),
                ),
            )


# --------------------------------------------------------------------- aws
def az_region(az):
    """Return the region an availability zone belongs to, if known."""
    return az.rstrip("abcdef") if az else None


def spot_product(platform):
    """Return the spot market a leg's platform is priced in.

    Windows and Linux are separate markets for one instance type. An
    unknown platform is Linux: CloudTrail records `windows` explicitly
    and omits the field otherwise.
    """
    return "Windows" if (platform or "").startswith("win") else "Linux/UNIX"


def spot_price(itype, az, at, platform, details=None):
    """Return the achieved spot price for one leg, or None."""
    prod = spot_product(platform)
    at_key = at.strftime("%Y-%m-%dT%H:%M:%SZ")
    if details is not None:
        cached = details.spot_price(itype, az, prod, at_key)
        if cached is not None:
            return cached
    ok, res = aws(
        "ec2",
        "describe-spot-price-history",
        "--region",
        az_region(az),
        "--instance-types",
        itype,
        "--availability-zone",
        az,
        "--product-descriptions",
        prod,
        "--start-time",
        (at - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--end-time",
        at_key,
    )
    if not ok or not res or not res.get("SpotPriceHistory"):
        return None
    hist = sorted(res["SpotPriceHistory"], key=lambda h: h["Timestamp"])
    price = None
    for h in hist:
        if ts(h["Timestamp"]) <= at:
            price = float(h["SpotPrice"])
    if price is None:
        price = float(hist[-1]["SpotPrice"])
    if details is not None:
        details.store_spot_price(itype, az, prod, at_key, price)
    return price


def _launch_details(event, iid):
    """Return the launch fields a RunInstances event records for iid."""
    # A RunInstances call that failed records a null responseElements,
    # so every level here can legitimately be absent or null.
    detail = json.loads(event["CloudTrailEvent"])
    response = detail.get("responseElements") or {}
    items = (response.get("instancesSet") or {}).get("items") or []
    for item in items:
        if item.get("instanceId") != iid:
            continue
        return {
            "instance_type": item.get("instanceType"),
            "az": (item.get("placement") or {}).get("availabilityZone"),
            # CloudTrail reports "windows" and omits the field for
            # Linux, matching the RunsOn banner's Platform values.
            "platform": item.get("platform"),
            "launched_at": ts(event["EventTime"]),
        }
    return None


def _region_events(iid, around, region):
    """Return one region's CloudTrail events for an instance."""
    ok, res = aws(
        "cloudtrail",
        "lookup-events",
        "--region",
        region,
        "--lookup-attributes",
        f"AttributeKey=ResourceName,AttributeValue={iid}",
        "--start-time",
        (around - LAUNCH_LOOKBACK).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--end-time",
        (around + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return res.get("Events", []) if ok else []


def instance_history(iid, around, region=None, details=None):
    """Return one instance's launch record and termination time.

    CloudTrail answers only in the instance's own region. With no
    region given, each of SEARCH_REGIONS is tried until one has events.
    """
    if details is not None:
        cached = details.instance_history(iid)
        if cached is not None:
            return cached
    regions = [region] if region else list(SEARCH_REGIONS)
    launch = None
    termination = None
    found_region = None
    for candidate in regions:
        events = _region_events(iid, around, candidate)
        if not events:
            continue
        found_region = candidate
        for ev in events:
            name = ev["EventName"]
            if termination is None and name in (
                "TerminateInstances",
                "BidEvictedEvent",
            ):
                termination = ts(ev["EventTime"])
            elif launch is None and name == "RunInstances":
                launch = _launch_details(ev, iid)
        break
    if details is not None and termination is not None:
        details.store_instance_history(iid, found_region, launch, termination)
    return launch, termination


# ----------------------------------------------------------------- derive
def _billing_values(run_start, termination, price):
    """Return billed hours and cost only from complete telemetry."""
    if termination is None:
        return None, None
    billed_hours = (termination - run_start).total_seconds() / 3600
    cost = billed_hours * price if price is not None else None
    return billed_hours, cost


def _enrich_leg(leg, details=None):
    log = parse_log(fetch_log(leg["job_id"]))
    leg.update(log)
    anchor = log["running_start"] or leg["job_start"]
    launch, term = instance_history(
        leg["instance_id"], anchor, az_region(log["az"]), details
    )
    launch = launch or {}
    # The log is authoritative where it exists; the launch record only
    # ever adds identity the banner left out.
    for field in ("instance_type", "az", "platform"):
        leg[field] = log[field] or launch.get(field)
    if log["log_known"]:
        # A log without an instance-pending phase reports a warm
        # instance, whose launch long predates this job and must not be
        # billed to it.
        run_start = log["running_start"] or leg["job_start"]
    else:
        run_start = launch.get("launched_at") or leg["job_start"]
    price = (
        spot_price(
            leg["instance_type"],
            leg["az"],
            run_start,
            leg["platform"],
            details,
        )
        if leg["instance_type"] and leg["az"]
        else None
    )
    dur_h, cost = _billing_values(run_start, term, price)
    first = leg["steps"][0]["start"] if leg["steps"] else run_start
    # Work on a truncated leg ran until the job ended; drawing that time
    # as shutdown would attribute a live runner's last minutes to a
    # teardown that never happened.
    last = (
        leg["steps"][-1]["end"]
        if leg["steps"] and not leg["truncated"]
        else leg["job_end"]
    )
    merged = {}
    for s in leg["steps"]:
        nm = _step_name(s["name"])
        merged[nm] = (
            merged.get(nm, 0.0) + (s["end"] - s["start"]).total_seconds()
        )
    leg["_derived"] = {
        "run_start": run_start,
        "price": price,
        "cost": cost,
        "billed_hours": dur_h,
        "boot_s": max(0.0, (first - run_start).total_seconds()),
        "steps_s": max(0.0, (last - first).total_seconds()),
        "shutdown_s": (
            max(0.0, (term - last).total_seconds())
            if term is not None
            else None
        ),
        "wait_s": (
            None if log["wait_ms"] is None else log["wait_ms"] / 1000
        ),
        "step_durs": merged,
        "termination_known": term is not None,
        "price_known": price is not None,
        "log_known": log["log_known"],
    }
    return leg


def run_payload(run_id, store=None, details=None, now=None):
    """Build detail only for a cached qualified seven-day run."""
    run_id = int(run_id)
    store = store or RunStore(RUN_DB)
    details = details or DetailStore(DETAIL_DB)
    current = now or now_utc()
    run = store.qualified_run(run_id, current - RUN_LOOKBACK, current)
    if run is None:
        raise PermissionError(
            "run is not in the cached qualified seven-day set"
        )
    stored = details.run_detail(run_id)
    if stored is not None:
        return stored
    legs, settled = fetch_legs(run_id)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda leg: _enrich_leg(leg, details), legs))
    # A log absent within moments of the job finishing is still being
    # archived; only an older absence is the permanent one a dead runner
    # leaves behind, and only that is safe to memoize.
    settled = settled and all(
        leg["_derived"]["log_known"]
        or current - leg["job_end"] >= LOG_ARCHIVE_GRACE
        for leg in legs
    )
    legs.sort(key=lambda leg: leg["_derived"]["run_start"])
    t0 = min(
        (leg["job_scheduled"] or leg["_derived"]["run_start"] for leg in legs),
        default=now_utc(),
    )
    out = {
        "run_id": str(run_id),
        "t0": t0.isoformat(),
        "status": run["status"],
        "conclusion": run["conclusion"],
        "started_leg_count": run["started_leg_count"],
        "legs": [],
    }
    for leg in legs:
        d = leg["_derived"]
        out["legs"].append(
            {
                "label": leg["label"],
                "instance_id": leg["instance_id"],
                "type": leg["instance_type"],
                "az": leg["az"],
                "platform": leg["platform"],
                "product": spot_product(leg["platform"]),
                "price": d["price"],
                "cost": d["cost"],
                "billed_hours": d["billed_hours"],
                "wait_s": d["wait_s"],
                "boot_s": d["boot_s"],
                "steps_s": d["steps_s"],
                "shutdown_s": d["shutdown_s"],
                "termination_known": d["termination_known"],
                "price_known": d["price_known"],
                "log_known": d["log_known"],
                "offset_s": (
                    (leg["job_scheduled"] or d["run_start"]) - t0
                ).total_seconds(),
                "steps": [
                    {"name": n, "dur_s": v} for n, v in d["step_durs"].items()
                ],
            }
        )
    # Only a fully priced and terminated run is safe to store; missing
    # telemetry can still arrive.
    complete = all(
        leg["_derived"]["termination_known"] and leg["_derived"]["price_known"]
        for leg in legs
    )
    if settled and complete and out["legs"]:
        details.store_run_detail(run_id, out)
    return out


# ----------------------------------------------------- cost explorer cache
def _ce_query(gran, start, end, group_key, metric, service=None):
    dims = [{"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}}]
    if service:
        dims.append({"Dimensions": {"Key": "SERVICE", "Values": [service]}})
    flt = dims[0] if len(dims) == 1 else {"And": dims}
    ok, res = aws(
        "ce",
        "get-cost-and-usage",
        "--time-period",
        f"Start={start},End={end}",
        "--granularity",
        gran,
        "--metrics",
        metric,
        "--group-by",
        f"Type=DIMENSION,Key={group_key}",
        "--filter",
        json.dumps(flt),
    )
    if not ok:
        raise PermissionError(res)
    return res["ResultsByTime"]


def _validate_account_range(start, end, gran):
    """Validate a bounded account range whose two dates are inclusive."""
    gran = gran.upper()
    if gran not in {"DAILY", "HOURLY"}:
        raise ValueError("granularity must be DAILY or HOURLY")
    if not all(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in (start, end)
    ):
        raise ValueError("start and end must use YYYY-MM-DD")
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("start and end must use YYYY-MM-DD") from exc
    if start_date > end_date:
        raise ValueError("start must be on or before end")
    days = (end_date - start_date).days + 1
    limit = MAX_HOURLY_RANGE_DAYS if gran == "HOURLY" else MAX_DAILY_RANGE_DAYS
    if days > limit:
        raise ValueError(
            f"{gran.lower()} ranges are limited to {limit} inclusive days"
        )
    return (
        start_date,
        end_date,
        end_date + timedelta(days=1),
        gran,
    )


def _expected_buckets(start, end, gran, timezone_offset=0):
    """Return bucket starts for inclusive start and end dates."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    offset = (
        timedelta(minutes=timezone_offset) if gran == "HOURLY" else timedelta()
    )
    current = (
        datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        - offset
    )
    stop = (
        datetime.combine(
            end_date + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        - offset
    )
    step = timedelta(hours=1) if gran == "HOURLY" else timedelta(days=1)
    fmt = "%Y-%m-%dT%H:00:00Z" if gran == "HOURLY" else "%Y-%m-%d"
    out = []
    while current < stop:
        out.append((current.strftime(fmt), current))
        current += step
    return out


def _parse_ce(usage_rt, cost_rt):
    out = {}
    for rt in usage_rt:
        d = out.setdefault(
            rt["TimePeriod"]["Start"], {"usage": {}, "cost": {}}
        )
        for g in rt["Groups"]:
            t = g["Keys"][0]
            if t != "NoInstanceType":
                d["usage"][t] = float(g["Metrics"]["UsageQuantity"]["Amount"])
    for rt in cost_rt:
        d = out.setdefault(
            rt["TimePeriod"]["Start"], {"usage": {}, "cost": {}}
        )
        for g in rt["Groups"]:
            d["cost"][g["Keys"][0]] = float(
                g["Metrics"]["UnblendedCost"]["Amount"]
            )
    return out


def _total(d):
    return sum(abs(value) for value in d.get("usage", {}).values())


def _gross_service_cost(payload):
    """Return aggregate gross service cost for a Cost Explorer hour."""
    return sum(payload.get("cost", {}).values())


def _hour_confirmed(period_start, payload, fetched_at=None):
    """Return whether one observed hour is settled enough to retain."""
    if _gross_service_cost(payload) != 0.0:
        return 1
    if fetched_at is None:
        return 0
    return int(fetched_at >= ts(period_start) + ZERO_COST_CONFIRMATION_AGE)


def _rollup(hours, day):
    u, c = {}, {}
    for h, d in hours.items():
        if h[:10] != day:
            continue
        for k, v in d.get("usage", {}).items():
            u[k] = u.get(k, 0.0) + v
        for k, v in d.get("cost", {}).items():
            c[k] = c.get(k, 0.0) + v
    return {"usage": u, "cost": c}


class UsageStore(SQLiteStore):
    """Transactional local store for retained Cost Explorer usage."""

    def _initialize(self):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            schema_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            if schema_version not in (0, 1, 2, USAGE_SCHEMA_VERSION):
                raise RuntimeError(
                    "unsupported usage database schema "
                    f"{schema_version}; expected {USAGE_SCHEMA_VERSION}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hourly (
                    period_start TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    total REAL NOT NULL,
                    confirmed INTEGER NOT NULL DEFAULT 0
                        CHECK (confirmed IN (0, 1))
                )
                """
            )
            hourly_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(hourly)"
                ).fetchall()
            }
            confirmation_added = "confirmed" not in hourly_columns
            if confirmation_added:
                connection.execute(
                    """
                    ALTER TABLE hourly
                    ADD COLUMN confirmed INTEGER NOT NULL DEFAULT 0
                        CHECK (confirmed IN (0, 1))
                    """
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily (
                    day TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    total REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            migrated = connection.execute(
                "SELECT value FROM metadata WHERE key = 'legacy_migrated'"
            ).fetchone()
            if migrated is None:
                self._migrate_legacy(connection)
                self._set_metadata(
                    connection, "legacy_migrated", now_utc().isoformat()
                )
            query_version = connection.execute(
                "SELECT value FROM metadata WHERE key = 'query_version'"
            ).fetchone()
            if query_version and query_version[0] != USAGE_QUERY_VERSION:
                raise RuntimeError(
                    "unsupported usage query version "
                    f"{query_version[0]}; expected {USAGE_QUERY_VERSION}"
                )
            self._set_metadata(
                connection, "query_version", USAGE_QUERY_VERSION
            )
            if schema_version in (0, 1):
                for table in ("hourly", "daily"):
                    rows = connection.execute(
                        f"SELECT rowid, payload FROM {table}"
                    ).fetchall()
                    connection.executemany(
                        f"UPDATE {table} SET total = ? WHERE rowid = ?",
                        [
                            (_total(json.loads(payload)), rowid)
                            for rowid, payload in rows
                        ],
                    )
            if schema_version < USAGE_SCHEMA_VERSION or confirmation_added:
                rows = connection.execute(
                    "SELECT rowid, payload FROM hourly"
                ).fetchall()
                connection.executemany(
                    "UPDATE hourly SET confirmed = ? WHERE rowid = ?",
                    [
                        (
                            int(
                                _gross_service_cost(json.loads(payload)) != 0.0
                            ),
                            rowid,
                        )
                        for rowid, payload in rows
                    ],
                )
                self._reconcile_daily_rows(connection)
            connection.execute(f"PRAGMA user_version = {USAGE_SCHEMA_VERSION}")

    def _migrate_legacy(self, connection):
        legacy = {}
        for name in ("hours.json", "days.json", "meta.json"):
            path = self.path.parent / name
            legacy[name] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {}
            )
        for period_start, payload in legacy["hours.json"].items():
            self._upsert_hourly(
                connection,
                period_start,
                payload,
                _hour_confirmed(period_start, payload),
            )
        for day, payload in legacy["days.json"].items():
            self._upsert_daily(connection, day, payload)
        for key, value in legacy["meta.json"].items():
            self._set_metadata(connection, key, str(value))

    @staticmethod
    def _upsert_hourly(connection, period_start, payload, confirmed):
        connection.execute(
            """
            INSERT INTO hourly(period_start, payload, total, confirmed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(period_start) DO UPDATE SET
                payload = excluded.payload,
                total = excluded.total,
                confirmed = excluded.confirmed
            """,
            (
                period_start,
                json.dumps(payload, sort_keys=True),
                _total(payload),
                confirmed,
            ),
        )

    @staticmethod
    def _upsert_daily(connection, day, payload):
        connection.execute(
            """
            INSERT INTO daily(day, payload, total)
            VALUES (?, ?, ?)
            ON CONFLICT(day) DO UPDATE SET
                payload = excluded.payload,
                total = excluded.total
            """,
            (day, json.dumps(payload, sort_keys=True), _total(payload)),
        )

    def _reconcile_daily_rows(self, connection):
        """Rebuild or remove daily rows covered by migrated hourly data."""
        days = connection.execute(
            """
            SELECT day FROM daily
            UNION
            SELECT DISTINCT substr(period_start, 1, 10) FROM hourly
            ORDER BY 1
            """
        ).fetchall()
        for (day,) in days:
            self._reconcile_daily_row(connection, day)

    def _reconcile_daily_row(self, connection, day):
        """Reconcile one daily rollup against its confirmed hours."""
        next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        rows = connection.execute(
            """
            SELECT period_start, payload, confirmed FROM hourly
            WHERE period_start >= ? AND period_start < ?
            """,
            (f"{day}T00:00:00Z", f"{next_day}T00:00:00Z"),
        ).fetchall()
        if len(rows) == 24 and all(row[2] for row in rows):
            hourly = {row[0]: json.loads(row[1]) for row in rows}
            self._upsert_daily(connection, day, _rollup(hourly, day))
        else:
            connection.execute("DELETE FROM daily WHERE day = ?", (day,))

    def metadata(self, key):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def frontier(self):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT MAX(period_start) FROM hourly WHERE total > 0"
            ).fetchone()
        return row[0] if row else None

    def hourly_bounds(self):
        """Return the first and last retained hourly bucket starts."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT MIN(period_start), MAX(period_start) FROM hourly"
            ).fetchone()
        return row if row else (None, None)

    def hourly_day_states(self, start, end):
        """Return retained and confirmed hourly counts by UTC day."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT substr(period_start, 1, 10),
                       COUNT(*),
                       SUM(confirmed)
                FROM hourly
                WHERE period_start >= ? AND period_start < ?
                GROUP BY substr(period_start, 1, 10)
                """,
                (f"{start}T00:00:00Z", f"{end}T00:00:00Z"),
            ).fetchall()
        return {
            day: (bucket_count, confirmed_count)
            for day, bucket_count, confirmed_count in rows
        }

    def confirmed_day_counts(self, start, end):
        """Return confirmed hourly counts by UTC day."""
        return {
            day: confirmed
            for day, (_, confirmed) in self.hourly_day_states(
                start, end
            ).items()
        }

    def latest_confirmed_hour(self, start, end):
        """Return the latest confirmed hour in an exclusive interval."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT MAX(period_start) FROM hourly
                WHERE confirmed = 1
                  AND period_start >= ? AND period_start < ?
                """,
                (start, end),
            ).fetchone()
        return row[0] if row else None

    def _range(self, table, key, start, end):
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {key}, payload FROM {table}
                WHERE {key} >= ? AND {key} < ?
                ORDER BY {key}
                """,
                (start, end),
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def hourly_range(self, start, end):
        return self._range("hourly", "period_start", start, end)

    def daily_range(self, start, end):
        return self._range("daily", "day", start, end)

    def acquire_refresh_lease(self, now, force=False):
        owner = uuid4().hex
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'refresh_lease'"
            ).fetchone()
            if lease_row:
                lease = json.loads(lease_row[0])
                if ts(lease["until"]) > now:
                    connection.rollback()
                    return "in_progress", None
            if force:
                last_force = connection.execute(
                    """
                    SELECT value FROM metadata
                    WHERE key = 'last_force_attempt'
                    """
                ).fetchone()
                if last_force and now - ts(last_force[0]) < FORCE_RATE_LIMIT:
                    connection.rollback()
                    return "rate_limited", None
                self._set_metadata(
                    connection, "last_force_attempt", now.isoformat()
                )
            else:
                last_auto = connection.execute(
                    """
                    SELECT value FROM metadata
                    WHERE key = 'last_auto_attempt'
                    """
                ).fetchone()
                if (
                    last_auto
                    and now - ts(last_auto[0]) < AUTO_ATTEMPT_THROTTLE
                ):
                    connection.rollback()
                    return "auto_throttled", None
                self._set_metadata(
                    connection, "last_auto_attempt", now.isoformat()
                )
            lease = {
                "owner": owner,
                "until": (now + REFRESH_LEASE).isoformat(),
            }
            self._set_metadata(connection, "refresh_lease", json.dumps(lease))
            connection.commit()
        return "acquired", owner

    def release_refresh_lease(self, owner):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'refresh_lease'"
            ).fetchone()
            if row and json.loads(row[0]).get("owner") == owner:
                connection.execute(
                    "DELETE FROM metadata WHERE key = 'refresh_lease'"
                )

    @staticmethod
    def _delete_owned_lease(connection, owner):
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'refresh_lease'"
        ).fetchone()
        if row and json.loads(row[0]).get("owner") == owner:
            connection.execute(
                "DELETE FROM metadata WHERE key = 'refresh_lease'"
            )

    @staticmethod
    def _assert_owned_lease(connection, owner, completed_at):
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'refresh_lease'"
        ).fetchone()
        lease = json.loads(row[0]) if row else {}
        if lease.get("owner") != owner or ts(lease["until"]) <= completed_at:
            raise RuntimeError("refresh lease expired or ownership was lost")

    def commit_hourly_refresh(
        self, start, end, buckets, fetched_at, owner, force=False
    ):
        """Replace a fetched interval and rebuild its confirmed days."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_owned_lease(connection, owner, fetched_at)
            connection.execute(
                """
                DELETE FROM hourly
                WHERE period_start >= ? AND period_start < ?
                """,
                (start, end),
            )
            for period_start, payload in buckets.items():
                self._upsert_hourly(
                    connection,
                    period_start,
                    payload,
                    _hour_confirmed(period_start, payload, fetched_at),
                )
            current_day = ts(start).date()
            last_touched_day = (ts(end) - timedelta(microseconds=1)).date()
            while current_day <= last_touched_day:
                day = current_day.isoformat()
                self._reconcile_daily_row(connection, day)
                current_day += timedelta(days=1)
            self._set_metadata(
                connection, "last_fetch", fetched_at.isoformat()
            )
            if force:
                self._set_metadata(
                    connection, "last_force_fetch", fetched_at.isoformat()
                )
            self._delete_owned_lease(connection, owner)
            connection.commit()

    def recompute_daily(self, days):
        """Reconcile daily rows from 24 confirmed retained hours."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for day in days:
                self._reconcile_daily_row(connection, day)


def _hourly_exposure_window(now):
    """Return Cost Explorer's recoverable completed-hour interval."""
    exposed_end = now.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    return exposed_end - COST_EXPLORER_HOURLY_RETENTION, exposed_end


def _fully_exposed_utc_days(start, end):
    """Return full UTC days wholly inside an hourly retention interval."""
    first_day = datetime.combine(
        start.date(), datetime.min.time(), tzinfo=timezone.utc
    )
    if first_day < start:
        first_day += timedelta(days=1)
    days = []
    current = first_day
    while current + timedelta(days=1) <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _automatic_refresh_decision(store, now):
    """Return refresh status and the zero-confirmed target UTC day."""
    exposed_start, exposed_end = _hourly_exposure_window(now)
    days = _fully_exposed_utc_days(exposed_start, exposed_end)
    if not days:
        return False, "no_recoverable_days", None
    confirmed = store.confirmed_day_counts(
        days[0].date().isoformat(),
        (days[-1] + timedelta(days=1)).date().isoformat(),
    )
    target_day = next(
        (
            day
            for day in reversed(days)
            if confirmed.get(day.date().isoformat(), 0) == 0
        ),
        None,
    )
    if target_day is None:
        return False, "confirmed", None
    target = target_day + timedelta(days=1) + AUTO_REFRESH_CUTOFF
    if now.astimezone(timezone.utc) < target:
        return False, "before_target", target_day
    return True, "unconfirmed", target_day


def _complete_hourly_buckets(start, end, parsed):
    """Represent zero/missing CE hours explicitly for coverage checks."""
    current = ts(start)
    stop = ts(end)
    buckets = {}
    while current < stop:
        key = current.strftime("%Y-%m-%dT%H:00:00Z")
        buckets[key] = parsed.get(key, {"usage": {}, "cost": {}})
        current += timedelta(hours=1)
    return buckets


def _hourly_refresh_window(store, now, target_day=None):
    """Return retained overlap through the latest completed UTC hour."""
    oldest, end_dt = _hourly_exposure_window(now)
    fmt = "%Y-%m-%dT%H:00:00Z"
    oldest_key = oldest.strftime(fmt)
    end_key = end_dt.strftime(fmt)
    latest = store.latest_confirmed_hour(oldest_key, end_key)
    start_dt = (
        max(ts(latest) - CONFIRMED_OVERLAP, oldest) if latest else oldest
    )
    if target_day is not None:
        start_dt = max(oldest, min(start_dt, target_day))
    return start_dt.strftime(fmt), end_key


def _refresh(store, now, force=False, target_day=None):
    """Fetch and transactionally replace the retained overlap window."""
    lease_status, owner = store.acquire_refresh_lease(now, force)
    if lease_status != "acquired":
        return lease_status
    try:
        window_target = None if force else target_day
        start, end = _hourly_refresh_window(store, now, window_target)
        usage_results = None
        cost_results = None
        usage_error = None
        cost_error = None
        try:
            usage_results = _ce_query(
                "HOURLY",
                start,
                end,
                "INSTANCE_TYPE",
                "UsageQuantity",
                EC2_COMPUTE,
            )
        except Exception as exc:  # noqa: BLE001  attempt both paid queries
            usage_error = exc
        try:
            cost_results = _ce_query(
                "HOURLY", start, end, "SERVICE", "UnblendedCost"
            )
        except Exception as exc:  # noqa: BLE001  preserve original failure
            cost_error = exc
        if usage_error is not None:
            raise usage_error
        if cost_error is not None:
            raise cost_error
        parsed = _parse_ce(usage_results, cost_results)
        buckets = _complete_hourly_buckets(start, end, parsed)
        store.commit_hourly_refresh(
            start, end, buckets, now_utc(), owner, force
        )
    except Exception:
        store.release_refresh_lease(owner)
        raise
    return "fetched"


def _coverage(expected, shown, gran, hourly_high_water, today=None):
    available = {key for key, _ in shown}
    missing = [key for key, _ in expected if key not in available]
    has_partial_bucket = False
    if gran == "DAILY" and hourly_high_water:
        high_water_day = hourly_high_water[:10]
        high_water_hour = ts(hourly_high_water).hour
        today = today or now_utc().date()
        has_partial_bucket = high_water_day in available and (
            high_water_hour < 23 or high_water_day == today.isoformat()
        )
    return {
        "complete": not missing and not has_partial_bucket,
        "requested_buckets": len(expected),
        "available_buckets": len(shown),
        "missing_buckets": len(missing),
        "first_available": shown[0][0] if shown else None,
        "last_available": shown[-1][0] if shown else None,
    }


def account_payload(
    start,
    end,
    gran,
    timezone_offset=0,
    force=False,
    store=None,
    now=None,
):
    """Assemble account data for an inclusive, bounded date range.

    The most recent retained full UTC day with no confirmed hours is
    automatically fetched from 00:15 UTC on the following day. The
    selected display range never affects that decision. Force bypasses
    the automatic content and time gate.
    """
    start_date, _, end_exclusive, gran = _validate_account_range(
        start, end, gran
    )
    try:
        timezone_offset = int(timezone_offset)
    except (TypeError, ValueError) as exc:
        raise ValueError("timezone offset must be an integer") from exc
    if not -840 <= timezone_offset <= 840:
        raise ValueError("timezone offset must be between -840 and 840")
    if timezone_offset % 60:
        raise ValueError("timezone offset must be a whole number of hours")
    store = store or UsageStore(USAGE_DB)
    now = now or now_utc()
    bucket_offset = (
        timedelta(minutes=timezone_offset) if gran == "HOURLY" else timedelta()
    )
    requested_start = (
        datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        - bucket_offset
    )
    requested_end = (
        datetime.combine(
            end_exclusive, datetime.min.time(), tzinfo=timezone.utc
        )
        - bucket_offset
    )
    completed_hour_end = now.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    data_end = min(requested_end, completed_hour_end)
    has_data_window = requested_start < data_end
    refresh_status = "cached"
    charged = False
    with _CE_LOCK:
        if force:
            should_refresh = True
            target_day = None
        else:
            (
                should_refresh,
                refresh_status,
                target_day,
            ) = _automatic_refresh_decision(store, now)
        if should_refresh:
            refresh_status = _refresh(store, now, force, target_day)
            charged = refresh_status == "fetched"

        frontier = store.frontier()
        _, hourly_high_water = store.hourly_bounds()
        utc_today = now.astimezone(timezone.utc).date()
        cache_end_date = min(end_exclusive, utc_today + timedelta(days=1))
        stop_day = cache_end_date.isoformat()
        hourly_states = (
            store.hourly_day_states(start_date.isoformat(), stop_day)
            if start_date < cache_end_date
            else {}
        )
        cached_days = {}
        if gran == "DAILY" and start_date < cache_end_date:
            store.recompute_daily(hourly_states)
            cached_days = store.daily_range(start_date.isoformat(), stop_day)

        expected = _expected_buckets(
            start, end, gran, timezone_offset=timezone_offset
        )
        local_today = (now + timedelta(minutes=timezone_offset)).date()
        if gran == "HOURLY" and end_exclusive == local_today + timedelta(
            days=1
        ):
            expected = [
                bucket for bucket in expected if bucket[1] < completed_hour_end
            ]
        data_expected = [bucket for bucket in expected if bucket[1] < data_end]
        if gran == "HOURLY":
            hourly = (
                store.hourly_range(
                    requested_start.strftime("%Y-%m-%dT%H:00:00Z"),
                    data_end.strftime("%Y-%m-%dT%H:00:00Z"),
                )
                if has_data_window
                else {}
            )
            shown = [
                (period, hourly[period])
                for period, _ in data_expected
                if period in hourly
            ]
        else:
            shown = []
            for day, bucket_start in data_expected:
                state = hourly_states.get(day)
                if state == (24, 24) and day in cached_days:
                    shown.append((day, cached_days[day]))
                    continue
                bucket_end = min(bucket_start + timedelta(days=1), data_end)
                hourly = store.hourly_range(
                    f"{day}T00:00:00Z",
                    bucket_end.strftime("%Y-%m-%dT%H:00:00Z"),
                )
                if hourly:
                    shown.append((day, _rollup(hourly, day)))

    times = [key for key, _ in expected]
    shown_by_key = dict(shown)

    def series(field):
        keys = sorted(
            {name for _, payload in shown for name in payload.get(field, {})}
        )
        output = []
        for name in keys:
            row = [
                (
                    shown_by_key[period].get(field, {}).get(name, 0.0)
                    if period in shown_by_key
                    else None
                )
                for period, _ in expected
            ]
            if any(value for value in row if value is not None):
                output.append({"name": name, "data": row})
        return output

    coverage = _coverage(
        data_expected, shown, gran, hourly_high_water, utc_today
    )
    coverage["future_buckets"] = len(expected) - len(data_expected)
    return {
        "start": start,
        "end": end,
        "end_inclusive": True,
        "granularity": gran,
        "charged": charged,
        "refresh_status": refresh_status,
        "times": times,
        "frontier": frontier,
        "last_fetch": store.metadata("last_fetch"),
        "coverage": coverage,
        "usage": series("usage"),
        "cost": series("cost"),
    }


# ------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = (
            obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
        )
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'none'; "
                "script-src 'self' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; "
                "base-uri 'none'; frame-ancestors 'none'; "
                "form-action 'none'"
            ),
        )
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Permissions-Policy",
            (
                "accelerometer=(), camera=(), geolocation=(), "
                "microphone=(), payment=(), usb=()"
            ),
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _allowed_origins(self):
        port = self.server.server_address[1]
        return {
            f"http://localhost:{port}",
            f"http://127.0.0.1:{port}",
        }

    def _authorize(self, api=False, require_origin=False):
        allowed = self._allowed_origins()
        allowed_hosts = {urlparse(origin).netloc for origin in allowed}
        if self.headers.get("Host") not in allowed_hosts:
            self._send({"error": "forbidden-host"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in allowed:
            self._send({"error": "forbidden-origin"}, 403)
            return False
        if require_origin and origin not in allowed:
            self._send({"error": "origin-required"}, 403)
            return False
        if api and not hmac.compare_digest(
            self.headers.get(TOKEN_HEADER, ""), API_TOKEN
        ):
            self._send({"error": "forbidden-token"}, 403)
            return False
        return True

    @staticmethod
    def _account_query(query):
        return (
            query["start"][0],
            query["end"][0],
            query.get("gran", ["DAILY"])[0],
            query.get("tz", ["0"])[0],
        )

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if not self._authorize(api=u.path.startswith("/api/")):
                return
            if u.path in ("/", "/index.html"):
                html = (
                    (HERE / "cost_dashboard.html")
                    .read_text(encoding="utf-8")
                    .replace("__API_TOKEN__", API_TOKEN)
                    .encode("utf-8")
                )
                return self._send(html, ctype="text/html; charset=utf-8")
            if u.path == "/cost_dashboard.js":
                script = (HERE / "cost_dashboard.js").read_bytes()
                return self._send(
                    script, ctype="text/javascript; charset=utf-8"
                )
            if u.path == "/api/runs":
                return self._send(fetch_runs())
            if u.path == "/api/run":
                run_id = q["id"][0]
                if not run_id.isdecimal() or int(run_id) < 1:
                    raise ValueError("run id must be a positive integer")
                return self._send(run_payload(run_id))
            if u.path == "/api/account":
                if "force" in q:
                    return self._send(
                        {
                            "error": "method-not-allowed",
                            "detail": (
                                "force refresh requires POST "
                                "/api/account/refresh"
                            ),
                        },
                        405,
                    )
                return self._send(account_payload(*self._account_query(q)))
            return self._send({"error": "not found"}, 404)
        except (KeyError, ValueError) as exc:
            self._send({"error": "invalid-request", "detail": str(exc)}, 400)
        except PermissionError as e:
            self._send({"error": "access-denied", "detail": str(e)}, 403)
        except Exception as e:  # noqa: BLE001  surface any failure to UI
            self._send({"error": type(e).__name__, "detail": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if not self._authorize(api=True, require_origin=True):
                return
            if u.path != "/api/account/refresh":
                return self._send({"error": "not found"}, 404)
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length:
                self.close_connection = True
                return self._send(
                    {
                        "error": "invalid-request",
                        "detail": "refresh requests do not accept a body",
                    },
                    400,
                )
            return self._send(
                account_payload(*self._account_query(q), force=True)
            )
        except (KeyError, ValueError) as exc:
            self._send({"error": "invalid-request", "detail": str(exc)}, 400)
        except PermissionError as exc:
            self._send({"error": "access-denied", "detail": str(exc)}, 403)
        except Exception as exc:  # noqa: BLE001  surface failure to UI
            self._send({"error": type(exc).__name__, "detail": str(exc)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RunStore(RUN_DB)
    UsageStore(USAGE_DB)
    DetailStore(DETAIL_DB)
    url = f"http://localhost:{args.port}"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fleet cost dashboard on {url}  (Ctrl-C to stop)")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
