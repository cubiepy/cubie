#requires -Version 7.2
<#
.SYNOPSIS
Keeps one normally elevated, fixed profiler worker supervised.
.DESCRIPTION
Only the fixed profiler_session.ps1 worker is launched. Worker failures
restart without another RunAs/UAC prompt. No queued commands are evaluated.
Create host.stop in the session directory for an explicit host stop.
#>
[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')]
    [string]$SessionName = 'weekend_20260904',
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$hostRoot = 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\_profiler_sessions'
$hostSession = Join-Path $hostRoot $SessionName
$hostPwsh = 'C:\Users\cca79\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
$hostWorker = 'C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\benchmarks\hardware_model\profiler_session.ps1'
$hostDeadline = [DateTimeOffset]::Parse('2026-09-06T11:59:00Z')
$hostStop = Join-Path $hostSession 'host.stop'
$workerRestart = Join-Path $hostSession 'worker.restart'
$hostStatus = [ordered]@{
    schema_version = 1; pid = $PID; worker_pid = $null; restarts = 0
    state = 'starting'; last_error = $null
    started_utc = [DateTimeOffset]::UtcNow.ToString('o')
    deadline_utc = $hostDeadline.ToString('o')
    elevated = [Security.Principal.WindowsPrincipal]::new(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    executable = $hostPwsh; worker = $hostWorker; session = $hostSession
}

function Save-HostStatus {
    $hostStatus.updated_utc = [DateTimeOffset]::UtcNow.ToString('o')
    $path = Join-Path $hostSession 'host_status.json'
    $temporary = "$path.$PID.tmp"
    $json = $hostStatus | ConvertTo-Json -Depth 5
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            [IO.File]::WriteAllText($temporary, $json)
            [IO.File]::Move($temporary, $path, $true)
            return
        } catch {
            $hostStatus.last_error = $_.Exception.Message
            Start-Sleep -Milliseconds 50
        }
    }
}

function Assert-HostPath($Path) {
    $cursor = [IO.Path]::GetFullPath($Path)
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band
                    [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse points are not allowed: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}

foreach ($path in @($hostPwsh, $hostWorker)) {
    Assert-HostPath $path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Fixed host dependency missing: $path"
    }
}
Assert-HostPath $hostSession
if ($ValidateOnly) {
    $hostStatus | ConvertTo-Json -Depth 5
    return
}
if (-not $hostStatus.elevated) { throw 'Launch this host with normal RunAs.' }
if ([DateTimeOffset]::UtcNow -ge $hostDeadline) {
    throw 'The fixed profiler deadline has passed.'
}
$null = New-Item -ItemType Directory -Path $hostSession -Force
$hostLock = [IO.File]::Open((Join-Path $hostSession 'host.lock'),
    'OpenOrCreate', 'ReadWrite', 'None')

# A per-worker Windows job object owns that worker and descendants.
# Closing it removes orphan children before any worker restart.
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public sealed class CubieProfilerJob : IDisposable {
    [StructLayout(LayoutKind.Sequential)] struct Basic {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinWorking, MaxWorking;
        public uint ActiveLimit;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct Counters {
        public ulong ReadOps, WriteOps, OtherOps, ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] struct Extended {
        public Basic Basic;
        public Counters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcess, PeakJob;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool SetInformationJobObject(IntPtr job, int kind,
        ref Extended info, uint size);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    IntPtr handle;
    public CubieProfilerJob() {
        handle = CreateJobObject(IntPtr.Zero, null);
        if (handle == IntPtr.Zero) throw new Win32Exception();
        var info = new Extended();
        info.Basic.Flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (!SetInformationJobObject(handle, 9, ref info,
                (uint)Marshal.SizeOf<Extended>())) {
            var error = new Win32Exception(); Dispose(); throw error;
        }
    }
    public void Assign(IntPtr process) {
        if (!AssignProcessToJobObject(handle, process))
            throw new Win32Exception();
    }
    public void Dispose() {
        if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; }
    }
}
'@

try {
    while ([DateTimeOffset]::UtcNow -lt $hostDeadline -and
            -not (Test-Path -LiteralPath $hostStop)) {
        $child = $null
        $job = $null
        $started = $false
        $outTask = $null
        $errTask = $null
        $stdout = $null
        $stderr = $null
        $explicitWorkerStop = $false
        try {
            $job = [CubieProfilerJob]::new()
            $info = [Diagnostics.ProcessStartInfo]::new($hostPwsh)
            $info.UseShellExecute = $false
            $info.CreateNoWindow = $true
            $info.RedirectStandardOutput = $true
            $info.RedirectStandardError = $true
            foreach ($argument in @('-NoProfile', '-File', $hostWorker,
                    '-SessionName', $SessionName)) {
                $info.ArgumentList.Add($argument)
            }
            $prefix = Join-Path $hostSession "worker_${PID}_$($hostStatus.restarts)"
            $stdout = [IO.File]::Open("$prefix.stdout.log", 'CreateNew', 'Write', 'Read')
            $stderr = [IO.File]::Open("$prefix.stderr.log", 'CreateNew', 'Write', 'Read')
            $child = [Diagnostics.Process]::new()
            $child.StartInfo = $info
            $null = $child.Start()
            $started = $true
            $job.Assign($child.Handle)
            $outTask = $child.StandardOutput.BaseStream.CopyToAsync($stdout)
            $errTask = $child.StandardError.BaseStream.CopyToAsync($stderr)
            $hostStatus.worker_pid = $child.Id
            $hostStatus.worker_sha256 = (Get-FileHash -LiteralPath $hostWorker).Hash
            $hostStatus.state = 'worker_running'
            Save-HostStatus
            while (-not $child.WaitForExit(500)) {
                Save-HostStatus
                if ([DateTimeOffset]::UtcNow -ge $hostDeadline -or
                        (Test-Path -LiteralPath $hostStop)) {
                    break
                }
                if (Test-Path -LiteralPath $workerRestart) {
                    try {
                        $workerStatus = Get-Content -LiteralPath (
                            Join-Path $hostSession 'status.json') -Raw |
                            ConvertFrom-Json -AsHashtable
                        if ($workerStatus.pid -eq $child.Id -and
                                $workerStatus.state -in @('idle_disarmed', 'idle_ready')) {
                            [IO.File]::Delete($workerRestart)
                            $hostStatus.last_restart_reason = 'maintenance_marker'
                            break
                        }
                    } catch { $hostStatus.last_error = $_.Exception.Message }
                }
            }
            if ($child.HasExited) {
                $hostStatus.last_worker_exit_code = $child.ExitCode
                try {
                    $workerStatus = Get-Content -LiteralPath (
                        Join-Path $hostSession 'status.json') -Raw |
                        ConvertFrom-Json -AsHashtable
                    $explicitWorkerStop = $workerStatus.pid -eq $child.Id -and
                        $workerStatus.state -eq 'stopped'
                } catch { $hostStatus.last_error = $_.Exception.Message }
            }
        } catch {
            $hostStatus.last_error = $_.Exception.Message
        } finally {
            if ($started -and -not $child.HasExited) {
                try { $child.Kill($true); $child.WaitForExit() } catch {
                    $hostStatus.last_error = $_.Exception.Message
                }
            }
            if ($job) { $job.Dispose() }
            foreach ($task in @($outTask, $errTask)) {
                if ($null -ne $task) {
                    try { $null = $task.GetAwaiter().GetResult() } catch {
                        $hostStatus.last_error = $_.Exception.Message
                    }
                }
            }
            if ($stdout) { $stdout.Dispose() }
            if ($stderr) { $stderr.Dispose() }
            if ($child) { $child.Dispose() }
            $hostStatus.worker_pid = $null
        }
        if ($explicitWorkerStop) { break }
        $hostStatus.restarts++
        $hostStatus.state = 'restart_wait'
        Save-HostStatus
        Start-Sleep -Seconds 2
    }
    $hostStatus.state = if ([DateTimeOffset]::UtcNow -ge $hostDeadline) {
        'deadline'
    } else { 'stopped' }
} finally {
    $hostStatus.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
    Save-HostStatus
    $hostLock.Dispose()
}
