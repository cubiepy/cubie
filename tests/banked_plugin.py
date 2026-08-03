"""Record driverless test verdicts, or deselect the tests they cover."""
import json
import os
from pathlib import Path

RECORD_PATH = os.environ.get("CUBIE_BANKED_OUT", "").strip()
DESELECT_PATH = os.environ.get("CUBIE_BANKED_RESULTS", "").strip()

_passed = set()
_failed = set()
_written = [0]


def pytest_collection_modifyitems(config, items):
    """Drop items whose verdict is already banked."""
    if not DESELECT_PATH:
        return
    path = Path(DESELECT_PATH)
    if not path.is_file():
        return
    banked = set(json.loads(path.read_text(encoding="utf-8")))
    if not banked:
        return
    deselected = [item for item in items if item.nodeid in banked]
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item.nodeid not in banked]


def pytest_runtest_logreport(report):
    """A node is banked only if every phase ran clean."""
    if not RECORD_PATH:
        return
    if report.failed or report.skipped:
        _failed.add(report.nodeid)
    elif report.when == "call" and report.passed:
        _passed.add(report.nodeid)


def pytest_testnodedown(node, error):
    if not RECORD_PATH:
        return
    output = getattr(node, "workeroutput", {})
    _passed.update(output.get("cubie_banked_passed", ()))
    _failed.update(output.get("cubie_banked_failed", ()))


def pytest_sessionfinish(session, exitstatus):
    if not RECORD_PATH:
        return
    if hasattr(session.config, "workeroutput"):
        session.config.workeroutput["cubie_banked_passed"] = sorted(_passed)
        session.config.workeroutput["cubie_banked_failed"] = sorted(_failed)
        return
    banked = sorted(_passed - _failed)
    _written[0] = len(banked)
    Path(RECORD_PATH).write_text(
        json.dumps(banked, indent=1), encoding="utf-8"
    )


def pytest_terminal_summary(terminalreporter):
    if RECORD_PATH:
        terminalreporter.write_line(f"BANKED wrote {_written[0]} node ids")
    elif DESELECT_PATH:
        terminalreporter.write_line(f"BANKED read {DESELECT_PATH}")
