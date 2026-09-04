# Persistent elevated profiler session

`profiler_host.ps1` keeps one normally elevated PowerShell supervisor alive
until **2026-09-06 11:59 UTC**, or an explicit stop. It starts the fixed
`profiler_session.ps1` worker with its inherited token and restarts that
worker after failures without another UAC prompt. Each worker starts
disarmed. It
accepts only `ping`, `profile`, and `stop`; profiling invokes the fixed
Nsight Compute 2026.2.1 executable and the existing CuBIE venv Python.
There is no command string evaluation, executable override, permission
change, ordinary benchmark action, or automatic elevation retry.

Root reviews the script and owns GPU scheduling. Launch once, using the
normal Windows UAC prompt (the hidden window stays available through its
PID and files):

```powershell
$worker = 'C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\benchmarks\hardware_model\profiler_session.ps1'
$profilerHost = 'C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\benchmarks\hardware_model\profiler_host.ps1'
$pwsh = 'C:\Users\cca79\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
Start-Process -FilePath $pwsh -Verb RunAs -WindowStyle Hidden -ArgumentList @('-NoProfile', '-File', $profilerHost, '-SessionName', 'weekend_20260904') -PassThru
```

The persistent files are under
`C:\local_working_projects\cubie-notes\hardware_unroll_placement\_profiler_sessions\weekend_20260904`.
`status.json` reports worker PID, elevation, deadline, current job,
child PID, state, and last error. An exclusive `worker.lock` prevents
two workers using the same session. A stopped worker can be restarted
before the deadline with the same queue history; IDs cannot be reused.
`host_status.json` reports the supervisor PID, worker PID, restart count,
worker hash and last error. `host.lock` prevents duplicate supervisors.
Worker stdout/stderr logs retain each restart's output. Both processes
retry Windows sharing/access conflicts during status replacement;
status publication is best-effort after the bounded retry, so a reader
cannot terminate the session merely by holding its status file open.

The supervisor assigns each worker to a private Windows job object with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; its worker and descendants are
cleaned up before a restart, including a child left by an unexpected
worker exit. It does not assign or signal unrelated processes. These
are the documented [Windows job object lifecycle semantics](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).
Only the fixed PowerShell executable and worker script can be launched
by the supervisor. It has no general command or profile action itself.

Write each request as UTF-8 JSON to a temporary filename, then rename it
atomically to `queue/<id>.json`. IDs are conservative single components;
the filename must match the JSON ID. Do not overwrite queue files.
Processed originals go to `claimed/` and receipts to `results/`. Invalid
or duplicate requests receive distinct rejection receipts; the service
continues. The worker serializes profile jobs and services ping/stop
while a profile is running. The queue is a trusted local research
interface, not a sandbox for untrusted Python scripts.

A ping launches no executable and accesses no GPU. Two pings with
different IDs must report the same elevated worker PID to demonstrate
reuse without another UAC prompt:

```json
{"id":"ping_001","action":"ping"}
```

GPU jobs remain queued until root writes `gpu_release.json` atomically:

```json
{
  "allow_profile_jobs": true,
  "external_jobs_finished": true,
  "blocked_process_ids": [48528, 56356]
}
```

The worker also checks the known prior bisect PID 48528. Root confirms
all other GPU work has finished before release and supplies additional
known PIDs as needed. This is an explicit scheduling gate, not GPU
activity detection. Remove the release file or set its first flag false
to prevent the next profile starting; wait for `idle_disarmed` before
launching GPU work elsewhere. Disarming does not cancel an active job.
The worker never signals unrelated processes. A recycled blocker PID
keeps the gate closed conservatively.
The release file must be newer than the current worker's start time.
After a supervised restart, inspect the failure and rewrite the release
file only when root intentionally releases the GPU again.

A profile request specifies either the fixed `hardware_probes` module
or one `.py` script under the selected tree's `benchmarks` directory.
Allowed trees are `research` and `epoch_ff3a567f`; their absolute paths
are fixed in the script. Supply the current SHA256 of the target file.
For script targets, optional `runtime_tree` selects one of those same
fixed trees for the working directory and runtime imports. For example,
`tree: research` with `runtime_tree: epoch_ff3a567f` runs a new research
benchmark against frozen CuBIE/harness imports. The fixed module target
requires matching trees so its hashed source is the module actually run.
The worker validates the hash, then holds the target open against
write/delete while profiling and rechecks that hash. Reparse points in
the source/output path ancestry are rejected. Imported modules are not
locked by this mechanism: use the frozen source tree and retain the
benchmark's source/compiler identity receipts.

```json
{
  "id": "icc_gate_001",
  "action": "profile",
  "tree": "research",
  "target": "hardware_probes",
  "sha256": "REPLACE_WITH_TARGET_FILE_SHA256",
  "arguments": ["icache", "--body-kib", "128", "--resident-warps", "8", "--iterations", "4096", "--profile-once"],
  "metrics": ["sm__icc_requests.sum", "gcc__cache_requests_type_instruction_lookup_miss.sum"],
  "sections": ["LaunchStats", "Occupancy"],
  "kernel_filter": "regex:probe",
  "launch_skip": 0,
  "launch_count": 1,
  "timeout_seconds": 900,
  "output_name": "icc_worker_gate_20260904",
  "output_flag": "--output"
}
```

For a script, set `target` to `script` and add its absolute `script`
path. `output_flag` must be `--out` or `--output`, matching that
benchmark's CLI. The worker adds the output argument itself; duplicate
output flags in `arguments` are rejected. Arguments are passed through
`.NET ProcessStartInfo.ArgumentList`, never evaluated as shell syntax.
The benchmark is responsible for accepting the injected output flag
and recording its own run/config status. Its CLI must expose one of
those flags; the worker does not rewrite benchmarks.

`output_name` must be a fresh single-component directory under the raw
data root. Each output includes the request, executable/argument/source
receipt, profiler stdout/stderr, diagnostic CSV, `.ncu-rep`, imported
wide `metrics.csv`, and benchmark artifacts under `benchmark/`. Existing
output directories are rejected. Each profile fixes clock/cache control
to `none`; metrics, sections, kernel filter, skip and count are structured
request fields. The child environment selects MLIR, disables CUDASIM,
and fixes `PYTHONPATH` to runtime `src`, runtime `benchmarks`, script-tree
`benchmarks`, and runtime root, in that order. Both resolved trees and
the complete `PYTHONPATH` are recorded in `command.json`; its CuBIE cache
is inside the fresh output directory. No environment override is accepted.
For `epoch_ff3a567f` only, the child environment removes inherited
`CUDA_HOME`, `CUDA_PATH`, `CUDA_PATH_V13_2`, and `CUDA_PATH_V13_3` to
reproduce the recorded epoch environment where all four were absent.
Their original inherited names/values are saved in `command.json` as
`removed_inherited_environment`. The parent's environment is unchanged;
no other inherited variable is removed by this rule. The strict benchmark
source/compiler identity comparison remains unchanged.

Success requires profiler exit zero, a saved report, import exit zero,
kernel counter rows, and every explicitly requested metric column. This
does not establish benchmark correctness, complete cohort coverage,
nonempty values for each metric, or physical counter interpretation;
inspect the raw artifacts and benchmark statuses. Profiled CUDA-event
times are not ordinary performance samples. Submit a one-kernel counter
gate and inspect its receipt before queuing a larger contrast. Failed
jobs keep their logs and do not close the worker; root must inspect a
failed gate before submitting additional work.

```json
{"id":"stop_001","action":"stop"}
```

Stop or the fixed deadline terminates only the worker's active profiler
process tree, retains artifacts, and ends the worker. A job timeout
terminates its own process tree and records failure, then the worker
can accept another job. Queue entries are not silently retried.
An orderly queued `stop` ends the supervisor too. To stop the supervisor
even when its worker is unhealthy, create an empty `host.stop` file in
the session directory. The host closes its private job and exits. Remove
that explicit stop file only when intentionally starting another host.
For a reviewed worker update, replace the fixed worker file, then root
disarms profiling and waits for `idle_disarmed`. Create an empty
`worker.restart` file in the session directory. The host accepts that
marker only when the status belongs to its current worker and is
`idle_disarmed` or `idle_ready`; it consumes the marker, cleans up its
private job, and restarts the same fixed worker, retaining elevation.
A marker arriving while busy waits for an idle state. Root must rewrite
the GPU release file for the restarted worker before profiling. This
fixed lifecycle control does not accept executable or command overrides.

CPU-only checks require no elevated token and start no child process:

```powershell
& $worker -ValidateOnly
& $worker -ValidateOnly -RequestPath 'C:\path\to\request.json'
& $profilerHost -ValidateOnly
```

The runtime must be PowerShell 7.2 or later (`ArgumentList`, asynchronous
stream copying, process-tree termination, static SHA256 helpers). The supplied runtime is
7.6.5. Runtime profiling is only verified after root's reviewed elevated
ping/profile gate; CPU validation is not a GPU execution claim.
