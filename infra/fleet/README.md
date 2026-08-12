# RunsOn Fleet stack (GPU CI runners)

OpenTofu configuration for the RunsOn **Fleet** stack that provides
the GPU runners for `.github/workflows/ci_cuda_tests.yml`.

## Why Fleet

The account's "All G and VT Spot" quota is a fixed 8 vCPU and the
On-Demand G/VT quota is 0. Under the
old Flex setup, each queued matrix job's webhook launched an instance
immediately — Flex has no `max-parallel` awareness — so the matrix
overran the quota, quota rejections tripped Flex's fixed 5-minute
on-demand snooze, and with 0 on-demand quota those legs simply failed.
The workflow worked around this with 4-vCPU runners plus a
re-dispatching `retry` job.

Fleet uses GitHub runner **scale sets**: jobs queue on the GitHub side
and the Fleet runtime launches EC2 capacity per *assigned* job, so
`strategy.max-parallel` genuinely bounds instance demand and the retry
apparatus goes away. The workflow runs `max-parallel: 1`. Windows
runners are g5.xlarge; Linux runners span three families in both sizes under
price-capacity allocation. On a launch failure the Fleet runtime
retries with backoff while the job stays queued, so capacity droughts
cost latency, not red legs.

The fleets have no `schedule` (warm standby) on purpose: RunsOn warm
pools use on-demand capacity, which this account cannot launch for G
instances. All capacity is cold spot launches.

## One-time setup

1. **GitHub organization.** Fleet registers organization-scoped runner
   scale sets; personal accounts are not supported. The repo must live
   in an organization (a free plan works — scale sets register into
   the org's default runner group).
2. **GitHub App** (organization mode). Create it from the pre-filled
   link (replace `<ORG>`):

   ```text
   https://github.com/organizations/<ORG>/settings/apps/new?name=RunsOn%20Fleet%20%5B<ORG>%5D&url=https%3A%2F%2Fruns-on.com&public=false&webhook_active=false&organization_self_hosted_runners=write&actions=read
   ```

   Generate a private key (.pem), install the App on the organization,
   and note the App ID.
3. **AWS deployer credentials.** Paste
   [`bootstrap/cloudshell-iam.sh`](bootstrap/cloudshell-iam.sh) into an
   AWS CloudShell session. It creates a name-scoped, region-locked
   deployer role, `cubie-fleet-deployer`. Point a profile at it, with
   `source_profile` naming an IAM user that is allowed to assume that
   role and whose access key is in `~/.aws/credentials`:

   ```ini
   # ~/.aws/config
   [profile cubie-fleet]
   role_arn       = arn:aws:iam::<account-id>:role/cubie-fleet-deployer
   source_profile = cubie-fleet-bootstrap
   region         = us-east-2
   ```

   The CLI assumes the role per call and refreshes the 1-hour session.
   To change permissions, edit the script and paste it again: it
   republishes both deployer policies (`cubie-fleet-deployer`,
   `cubie-fleet-deployer-scoped`) as new default versions, and live
   sessions pick them up.
4. **Variables.** Copy `terraform.tfvars.example` to
   `terraform.tfvars` (gitignored) and fill in the App ID, key path,
   RunsOn license key (one license covers Flex and Fleet), and alert
   email. Networking needs no input: the stack creates its own VPC
   with public subnets in all three AZs (GPU spot pools span 2a/2b/2c
   and g5 exists only in 2a/2c; public-only means no NAT cost).

## Deploy

```powershell
cd infra/fleet
tofu init
tofu plan
tofu apply
```

State is local (`terraform.tfstate`, gitignored) and contains the
license key and App private key — keep it on the machine that manages
the stack.

After apply, the scale sets `cubie-fleet-gpu-linux` and
`cubie-fleet-gpu-windows` appear under the organization's Actions
runner settings, and workflows target them with:

```yaml
runs-on: runs-on/fleet=gpu-linux/env=production
```

## Caching

Runners deliberately do **not** enable RunsOn's `s3-cache` (Magic
Cache) extra. It requires a `runs-on/action@v2` step in every job
(without it, the sidecar intercepts the GitHub artifact service and
every `actions/upload-artifact` call fails on a non-JSON
CreateArtifact response — observed live), and RunsOn documents that
the shared S3 cache bucket must not be enabled for runners that
public repositories can use — cubie is public. Workflow-level caching
(setup-uv) uses GitHub's cache service instead.

## CloudWatch custom metrics

ECS Container Insights is **off** on the `cubie-fleet` cluster. The
RunsOn runtime submodule creates the cluster with
`containerInsights = enabled` and offers no way to change that: its
`container_insights_enabled` variable defaults to true, and neither the
`fleet` module nor the `control_plane/fleet` module in between passes a
value through (checked up to module 3.2.2). Enabled, the single
always-on `fleetd` Fargate task publishes 38 `ECS/ContainerInsights`
metrics; CloudWatch bills custom metrics at $0.30/metric/month past the
10 that are always free, so it cost about **$8.40/month** for telemetry
nothing here reads — the module defines no alarms, grants the Fleet
worker no `cloudwatch:GetMetric*`, and `cost_dashboard.py` gets its
numbers from the GitHub Actions API, EC2, CloudTrail and Cost Explorer.

With no module input to set, `terraform_data.disable_container_insights`
in `main.tf` calls `ecs:UpdateClusterSettings` through the AWS CLI after
the module applies. It re-runs on **every** apply, deliberately: the
module's `aws_ecs_cluster` resource re-asserts `enabled` each time it is
applied, so a trigger that only fired on change would leave the setting
on afterwards. Consequently `tofu plan` always reports that one resource
as replaced — that line is expected, and everything else in the plan
still reads as genuine drift. Apply therefore needs the `aws` CLI on
`PATH` and the `cubie-fleet` profile, which the deploy already assumes.

## Cost & timeline dashboard

`cost_dashboard.py` serves a local interactive dashboard for GPU CI cost
and timing.

```powershell
python infra/fleet/cost_dashboard.py    # opens http://localhost:8787
```

Pick a run from the dropdown to see, per leg: a timeline of spot-capacity
wait / boot / CI steps / shutdown with the run total broken down beside
it; time in each CI step (with a run-total bar beside it); cost at the
achieved spot price; minutes and cost per instance type and spot product
with the average spot rate annotated; and spot-capacity wait per leg.
Windows and Linux are separate spot markets for one instance type and
get their own bars. The account section
takes inclusive from/to date pickers and a granularity and charts
whole-account usage hours per instance type and gross usage $ by service.
Hourly ranges are limited to 366 inclusive days and daily ranges to 3,660
days.

The run dropdown contains every `ci_cuda_tests.yml` workflow record
created in the exact rolling last seven days that started at least one
RunsOn GPU instance. Success, failure, cancellation, and in-progress runs
are eligible; skipped, approval-only, gate-only, precompile-only, and
queued-without-instance records are excluded. Entries are newest first
and show the browser-local creation time, run or PR title, started-leg
count, and final or current status. The newest qualifying run is always
selected initially; page URL parameters do not select runs.

Qualification uses fully paginated workflow and job results. A separate
transactional SQLite cache at `.dashboard-cache/runs.sqlite3` retains
completed positive and negative decisions until they leave the seven-day
window. Nonterminal runs are reinspected on each upstream list refresh.
The upstream workflow list is attempted at most once per 60 seconds
across dashboard processes, and a persisted lease coalesces concurrent
scans. Failed refreshes preserve and serve the last usable snapshot. Run
detail requests are accepted only for IDs in the cached qualified
seven-day snapshot; the endpoint cannot be used to inspect arbitrary
positive IDs.

It correlates three data planes, keyed on the EC2 instance id RunsOn
embeds in each runner name (`runs-on--i-<id>--...`): the GitHub Actions
Jobs API (step timings), each leg's `Set up job` log (RunsOn boot
timeline, instance type/AZ, launch time), and AWS via the `cubie-fleet`
profile — `ec2:DescribeSpotPriceHistory` (achieved spot rate),
`cloudtrail:LookupEvents` (instance launch and terminate), and Cost Explorer
(`ce:GetCostAndUsage`) for the account panels. The last two are the
read-only grants the bootstrap policy's `ReadOnly` / `HistoryReadOnly` /
`CostExplorerReadOnly` statements add.

Both AWS instance reads are made in the region a leg's own instance ran
in, taken from its AZ. A leg with no log, and therefore no AZ, is looked
up in each region in `SEARCH_REGIONS` (current region first) until one
answers. The matching grant is `HistoryReadOnly`:
`ec2:DescribeSpotPriceHistory` and `cloudtrail:LookupEvents` across
`HISTORY_REGIONS` in the bootstrap script, while every other deployer
permission stays locked to the active region. **A new fleet region must
be added to both lists, and the bootstrap script rerun.** Without the
grant, that region's runs render with unknown price and cost.

Settled run detail is cached durably in a third transactional SQLite
store, `.dashboard-cache/details.sqlite3` (gitignored): a run already
looked at redraws with no GitHub or AWS request, across process
restarts (12.1 s to 0.03 s for a 16-leg run). A run payload is stored
once every GPU leg has completed, its log has arrived or is permanently
absent, and every leg has both a spot price and a termination time; an
incomplete view is served but not stored. The two per-leg AWS reads are
cached in their own right, which is what makes a run that is still
settling cheap to reload. Only known answers are stored; a denied or
not-yet-recorded read is retried. Nothing is evicted by age. A schema or
payload-shape change discards the file rather than migrating it, and
deleting it costs only the refetch.

**Cost of use:** per-run views are free (GitHub API, `ec2:Describe*` and
`cloudtrail:LookupEvents` carry no charge). Only the account panels touch
Cost Explorer, billed $0.01 per `GetCostAndUsage` request. Account usage is
sourced from **hourly** Cost Explorer data and daily values are aggregated
from it (this matches CE's own daily totals to the cent).

The dashboard owns a transactional SQLite usage database at
`.dashboard-cache/usage.sqlite3` (gitignored). Existing `hours.json`,
`days.json`, and `meta.json` caches are imported once. Acquired hourly
buckets are retained indefinitely. Each hour stores its own confirmation
state, and days are rolled up only when all 24 retained hours are
confirmed. An overlapping fetch replaces both payload and confirmation
state in one transaction. Migrated daily rows without 24 confirmed
supporting hours are discarded rather than treated as authoritative.

Automatic refresh is independent of the selected display range. Non-zero
aggregate gross service cost confirms an observed hour immediately. A
zero-cost or missing hour confirms only when a successful fetch completes
at least 48 hours after that hour began, allowing Cost Explorer billing to
settle without causing genuinely idle hours to be retried forever.
Existing database rows migrate conservatively: non-zero-cost hours become
confirmed, while zero-cost and missing hours require a new sufficiently
late observation.

Hourly Cost Explorer data is treated as recoverable for approximately 14
days. Within that window, the dashboard considers only complete UTC days;
the partially expired oldest day and the current partial day are excluded.
It finds the most recent complete day with zero confirmed buckets (missing
buckets count as zero). At or after **00:15 UTC** on the following day,
that day triggers an automatic fetch. A day with even one confirmed hour
does not trigger, and no automatic fetch occurs when every eligible day
has at least one confirmed hour.

An accepted automatic attempt records its timestamp before AWS is called.
Reloads are then throttled for 15 minutes even if the AWS request fails or
returns all zeroes. A persisted ten-minute lease coalesces concurrent
dashboard processes. Every accepted fetch makes two Cost Explorer calls
even if the first fails: one for EC2 usage by instance type and one for
gross cost by service. An automatic fetch starts at the earlier of the
zero-confirmed target day's 00:00 UTC boundary and 12 hours before the
latest confirmed retained bucket, clipped to the oldest hourly-retention
boundary. With no confirmed bucket it starts at that boundary. It ends at
the start of the current UTC hour, so “latest exposed” means the latest
completed UTC hour; Cost Explorer exposes no separate hourly watermark.
`last_fetch` is committed only when both calls succeed and the replacement
transaction completes.

Account plots load automatically and reload when their date or granularity
controls change. **Force fetch** is the only fetch control; it bypasses the
automatic time and content gate with an authenticated POST and has its own
persisted five-minute attempt limit. It fetches the same retention-aware
overlap window without an automatic target-day extension, not the selected
historical range. The dashboard never asks Cost Explorer for hourly data
older than its recoverable boundary or for the current/future hour; it
renders available older cache data and reports unavailable coverage.
Requests may extend into the future, where plot buckets remain visible as
empty slots. The default view is hourly for the latest three browser-local
calendar days through the latest completed hour, and visible absolute
timestamps use the browser's local timezone.

The local server binds only to `127.0.0.1`. It validates the exact
localhost Host and Origin, injects a per-process token into the page, and
requires that token in a custom header for every API request. Responses
disable caching and set restrictive CSP, framing, referrer, and MIME
headers. ECharts remains CDN-hosted, but its exact bytes are pinned with
Subresource Integrity and `crossorigin="anonymous"`; all dashboard
JavaScript is served locally. Missing spot-price or termination telemetry
is shown as incomplete and is never converted to a zero-cost leg.

A runner that dies mid-step takes its job log with it: GitHub marks the
job failed, leaves the running step with no end time, and answers the job
log endpoint with 404 forever. Such a leg no longer fails the whole run
view. Its CloudTrail `RunInstances` event supplies the instance type, AZ
and platform the log banner would have, so it still prices, and billing
runs from the instance launch. Its steps band extends to the job's end
rather than to its last completed step, because the runner was still
working. The leg is labelled `(no log)` on the timeline and step axes,
its RunsOn queue wait stays unknown rather than zero, and the unfinished
step contributes no duration to the per-step totals. A 404 is never
cached, so a log archived moments after a job completes is still picked
up.

Requirements: `gh` authenticated to the repo and the `cubie-fleet` AWS
profile; the pinned ECharts asset needs browser internet access. The AWS
CLI subprocess is forced to UTF-8 (it otherwise dies on Windows rendering
the non-breaking spaces CloudTrail events carry).
