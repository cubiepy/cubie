#requires -Version 7.2
<#
.SYNOPSIS
Runs one bounded, elevated Nsight Compute queue until the fixed deadline.
.DESCRIPTION
Start once through normal RunAs. Queue JSON ping/profile/stop requests.
Profile jobs stay queued until gpu_release.json explicitly releases the GPU.
No shell commands, executable overrides, or driver policy changes are accepted.
Use -ValidateOnly [-RequestPath request.json] for CPU-only validation.
#>
[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')]
    [string]$SessionName = 'weekend_20260904',
    [switch]$ValidateOnly,
    [string]$RequestPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$rawRoot = 'C:\local_working_projects\cubie-notes\hardware_unroll_placement'
$trees = @{
    research = 'C:\local_working_projects\cubie-worktrees\hardware-unroll-placement'
    epoch_ff3a567f = 'C:\local_working_projects\cubie-worktrees\hardware-epoch-ff3a567f'
}
$python = 'C:\local_working_projects\cubie\.venv\Scripts\python.exe'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe'
$deadline = [DateTimeOffset]::Parse('2026-09-06T11:59:00Z')
$sessionRoot = Join-Path $rawRoot "_profiler_sessions\$SessionName"
$script:stopRequested = $false
$script:status = [ordered]@{
    schema_version = 1
    pid = $PID
    elevated = [Security.Principal.WindowsPrincipal]::new(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    started_utc = [DateTimeOffset]::UtcNow.ToString('o')
    deadline_utc = $deadline.ToString('o')
    state = 'starting'
    current_job = $null
    child_pid = $null
    last_error = $null
    python = $python
    ncu = $ncu
    session_root = $sessionRoot
}

function Write-JsonAtomic($Path, $Value, $BestEffort = $false) {
    $temporary = "$Path.$PID.tmp"
    $json = $Value | ConvertTo-Json -Depth 20
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            [IO.File]::WriteAllText($temporary, $json)
            [IO.File]::Move($temporary, $Path, $true)
            return
        } catch {
            $failure = $_.Exception
            while ($failure.InnerException) { $failure = $failure.InnerException }
            $nativeCode = $failure.HResult -band 65535
            if ($nativeCode -notin @(5, 32, 33) -or $attempt -eq 19) {
                if ($BestEffort) {
                    $script:status['last_status_error'] = $_.Exception.Message
                    return
                }
                throw
            }
            Start-Sleep -Milliseconds 50
        }
    }
}

function Save-Status {
    $script:status.updated_utc = [DateTimeOffset]::UtcNow.ToString('o')
    Write-JsonAtomic (Join-Path $sessionRoot 'status.json') $script:status $true
}

function Assert-SafePath($Path, $Root) {
    $full = [IO.Path]::GetFullPath($Path)
    $base = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    if ($full -ne $base -and -not $full.StartsWith(
            "$base\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside permitted root: $full"
    }
    $cursor = $full
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse points are not allowed: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

function Assert-Keys($Value, $Allowed) {
    foreach ($key in $Value.Keys) {
        if ($key -notin $Allowed) { throw "Unknown request field: $key" }
    }
}

function Read-Request($Path) {
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -gt 65536) { throw 'Request exceeds 64 KiB.' }
    $request = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
    if ($request -isnot [System.Collections.IDictionary]) {
        throw 'Request must be a JSON object.'
    }
    Assert-Keys $request @('id', 'action', 'tree', 'runtime_tree', 'target', 'script',
        'sha256', 'arguments', 'metrics', 'sections', 'kernel_filter',
        'launch_skip', 'launch_count', 'timeout_seconds', 'output_name',
        'output_flag')
    if ($request.id -isnot [string] -or
            $request.id -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$') {
        throw 'Invalid request id.'
    }
    if ($request.action -notin @('ping', 'profile', 'stop')) {
        throw 'Action must be ping, profile, or stop.'
    }
    if ($request.action -ne 'profile') {
        Assert-Keys $request @('id', 'action')
        return $request
    }
    if ($request.tree -notin $trees.Keys) { throw 'Unknown source tree.' }
    $tree = $trees[$request.tree]
    $runtimeName = if ($request.ContainsKey('runtime_tree')) {
        $request.runtime_tree
    } else { $request.tree }
    if ($runtimeName -notin $trees.Keys) { throw 'Unknown runtime tree.' }
    $runtimeTree = $trees[$runtimeName]
    if ($request.target -eq 'hardware_probes') {
        if ($runtimeName -ne $request.tree) {
            throw 'Fixed module requires matching source and runtime trees.'
        }
        if ($request.ContainsKey('script')) { throw 'Module has a fixed path.' }
        $source = Join-Path $tree 'benchmarks\hardware_model\hardware_probes.py'
        if ($request.output_flag -ne '--output') {
            throw 'hardware_probes requires output_flag --output.'
        }
        if (@($request.arguments).Count -eq 0 -or
                $request.arguments[0] -notin @('icache', 'fp32', 'memory')) {
            throw 'Unknown hardware_probes subcommand.'
        }
        if ('--profile-once' -notin $request.arguments) {
            throw 'hardware_probes profiling requires --profile-once.'
        }
    } elseif ($request.target -eq 'script') {
        if ($request.script -isnot [string] -or
                [IO.Path]::GetExtension($request.script) -ne '.py') {
            throw 'Script target requires a .py path.'
        }
        $source = Assert-SafePath $request.script (Join-Path $tree 'benchmarks')
        if ($request.output_flag -notin @('--out', '--output')) {
            throw 'Script output_flag must be --out or --output.'
        }
    } else { throw 'Target must be hardware_probes or script.' }
    $source = Assert-SafePath $source (Join-Path $tree 'benchmarks')
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing source: $source"
    }
    if ($request.sha256 -isnot [string] -or
            $request.sha256 -notmatch '^[A-Fa-f0-9]{64}$') {
        throw 'A source SHA256 is required.'
    }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
            $request.sha256) { throw 'Source SHA256 does not match request.' }
    if ($request.arguments -isnot [array] -or $request.arguments.Count -gt 128) {
        throw 'arguments must be an array of at most 128 strings.'
    }
    foreach ($argument in $request.arguments) {
        if ($argument -isnot [string] -or $argument.Length -gt 2048 -or
                $argument.Contains([char]0) -or
                $argument -match '^--(?:out|output)(?:=|$)') {
            throw 'Invalid benchmark argument or output override.'
        }
    }
    foreach ($field in @('metrics', 'sections')) {
        if ($request[$field] -isnot [array] -or $request[$field].Count -gt 128) {
            throw "$field must be an array of at most 128 names."
        }
        foreach ($name in $request[$field]) {
            if ($name -isnot [string] -or
                    $name -notmatch '^[A-Za-z][A-Za-z0-9_.:]{0,255}$') {
                throw "Invalid $field name."
            }
        }
    }
    if (($request.metrics.Count + $request.sections.Count) -eq 0) {
        throw 'At least one metric or section is required.'
    }
    if ($request.kernel_filter -isnot [string] -or
            $request.kernel_filter.Length -notin 1..512 -or
            $request.kernel_filter.Contains([char]0)) {
        throw 'kernel_filter must be a nonempty bounded string.'
    }
    foreach ($field in @('launch_skip', 'launch_count', 'timeout_seconds')) {
        if (-not $request.ContainsKey($field) -or
                $request[$field] -isnot [long] -and
                $request[$field] -isnot [int]) {
            throw "$field must be an integer."
        }
    }
    if ($request.launch_skip -lt 0 -or $request.launch_skip -gt 1000000 -or
            $request.launch_count -lt 1 -or $request.launch_count -gt 10000 -or
            $request.timeout_seconds -lt 1 -or
            $request.timeout_seconds -gt 172800) {
        throw 'Launch limits or timeout are outside permitted bounds.'
    }
    if ($request.output_name -isnot [string] -or
            $request.output_name -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$') {
        throw 'output_name must be a single safe directory name.'
    }
    $output = Assert-SafePath (Join-Path $rawRoot $request.output_name) $rawRoot
    if (Test-Path -LiteralPath $output) { throw 'Output already exists.' }
    $request['_source'] = $source
    $request['_tree'] = $tree
    $request['_runtime_tree'] = $runtimeTree
    $request['_pythonpath'] = "$runtimeTree\src;$runtimeTree\benchmarks;$tree\benchmarks;$runtimeTree"
    $request['_output'] = $output
    return $request
}

function Write-Receipt($Id, $Value) {
    $path = Join-Path $sessionRoot "results\$Id.json"
    if (Test-Path -LiteralPath $path) { throw "Duplicate receipt id: $Id" }
    Write-JsonAtomic $path $Value
}

function Claim-Request($File) {
    $destination = Join-Path $sessionRoot "claimed\$($File.Name)"
    [IO.File]::Move($File.FullName, $destination)
    return $destination
}

function Read-QueuedRequest($File) {
    $null = Assert-SafePath $File.FullName (Join-Path $sessionRoot 'queue')
    $request = Read-Request $File.FullName
    if ($File.BaseName -cne $request.id) {
        throw 'Queue filename must be exactly <request id>.json.'
    }
    foreach ($used in @("claimed\$($request.id).json", "results\$($request.id).json")) {
        if (Test-Path -LiteralPath (Join-Path $sessionRoot $used)) {
            throw "Request id has already been used: $($request.id)"
        }
    }
    return $request
}

function Reject-Request($File, $Message) {
    $script:status.last_error = $Message
    $rejection = "rejected_$([Guid]::NewGuid().ToString('N'))"
    if (Test-Path -LiteralPath $File.FullName) {
        [IO.File]::Move($File.FullName, (Join-Path $sessionRoot "claimed\$rejection.json"))
    }
    Write-JsonAtomic (Join-Path $sessionRoot "results\$rejection.json") @{
        state = 'rejected'; error = $Message; request_file = $File.Name
        utc = [DateTimeOffset]::UtcNow.ToString('o'); worker_pid = $PID
    }
}

function Handle-ControlRequests {
    $queue = Join-Path $sessionRoot 'queue'
    foreach ($file in @(Get-ChildItem -LiteralPath $queue -Filter '*.json' -File |
            Sort-Object Name)) {
        try {
            $request = Read-QueuedRequest $file
            if ($request.action -eq 'profile') { continue }
            $null = Claim-Request $file
            if ($request.action -eq 'stop') { $script:stopRequested = $true }
            Write-Receipt $request.id ([ordered]@{
                id = $request.id; action = $request.action; state = 'complete'
                utc = [DateTimeOffset]::UtcNow.ToString('o')
                worker_status = $script:status
            })
        } catch {
            Reject-Request $file $_.Exception.Message
        }
    }
}

function Test-GpuRelease {
    $releasePath = Join-Path $sessionRoot 'gpu_release.json'
    if (-not (Test-Path -LiteralPath $releasePath)) { return $false }
    if ((Get-Item -LiteralPath $releasePath).LastWriteTimeUtc -lt
            [DateTimeOffset]::Parse($script:status.started_utc).UtcDateTime) {
        return $false
    }
    $release = Get-Content -LiteralPath $releasePath -Raw |
        ConvertFrom-Json -AsHashtable
    Assert-Keys $release @('allow_profile_jobs', 'external_jobs_finished',
        'blocked_process_ids')
    if ($release.allow_profile_jobs -isnot [bool] -or
            $release.external_jobs_finished -isnot [bool]) {
        throw 'GPU release flags must be JSON booleans.'
    }
    if (-not $release.allow_profile_jobs -or
            -not $release.external_jobs_finished) { return $false }
    if ($release.blocked_process_ids -isnot [array]) {
        throw 'GPU release requires blocked_process_ids array.'
    }
    foreach ($blocker in @(@(48528) + $release.blocked_process_ids)) {
        if ($blocker -isnot [long] -and $blocker -isnot [int] -or $blocker -le 0) {
            throw 'Invalid blocked process id.'
        }
        if (Get-Process -Id $blocker -ErrorAction SilentlyContinue) {
            return $false
        }
    }
    return $true
}

function Set-EpochRuntimeEnvironment($Environment, $WorkingDirectory) {
    $removed = [ordered]@{}
    if ($WorkingDirectory -eq $trees.epoch_ff3a567f) {
        foreach ($name in @('CUDA_HOME', 'CUDA_PATH', 'CUDA_PATH_V13_2', 'CUDA_PATH_V13_3')) {
            if ($Environment.ContainsKey($name)) {
                $removed[$name] = $Environment[$name]
                $null = $Environment.Remove($name)
            }
        }
    }
    return $removed
}

function Invoke-FixedProcess($Arguments, $WorkingDirectory, $Prefix, $EndTime, $PythonPath = $null) {
    $info = [Diagnostics.ProcessStartInfo]::new($ncu)
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.WorkingDirectory = $WorkingDirectory
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    $info.Environment['PYTHONPATH'] = if ($PythonPath) { $PythonPath } else {
        "$WorkingDirectory\src;$WorkingDirectory\benchmarks;$WorkingDirectory"
    }
    $info.Environment['PYTHONNOUSERSITE'] = '1'
    $info.Environment['CUBIE_CUDA_BACKEND'] = 'mlir'
    $info.Environment['NUMBA_ENABLE_CUDASIM'] = '0'
    $info.Environment['CUBIE_CACHE_DIR'] = Join-Path (Split-Path $Prefix) 'cache'
    $null = Set-EpochRuntimeEnvironment $info.Environment $WorkingDirectory
    $stdout = [IO.File]::Open("$Prefix.stdout.log", 'CreateNew', 'Write', 'Read')
    $stderr = [IO.File]::Open("$Prefix.stderr.log", 'CreateNew', 'Write', 'Read')
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    $started = $false
    try {
        $null = $process.Start()
        $started = $true
        $script:status.child_pid = $process.Id
        Save-Status
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $errTask = $process.StandardError.BaseStream.CopyToAsync($stderr)
        while (-not $process.WaitForExit(500)) {
            Handle-ControlRequests
            Save-Status
            if ($script:stopRequested -or [DateTimeOffset]::UtcNow -ge $EndTime) {
                $process.Kill($true)
                $process.WaitForExit()
                throw 'Own profiler process stopped by request, timeout, or deadline.'
            }
        }
        $null = $outTask.GetAwaiter().GetResult()
        $null = $errTask.GetAwaiter().GetResult()
        return $process.ExitCode
    } finally {
        if ($started -and -not $process.HasExited) {
            $process.Kill($true)
            $process.WaitForExit()
        }
        $stdout.Dispose()
        $stderr.Dispose()
        $process.Dispose()
        $script:status.child_pid = $null
    }
}

function New-LockedSourceSnapshot($SourceLock, $Output, $ExpectedHash) {
    $path = Join-Path $Output 'benchmark_source.py'
    $snapshot = [IO.File]::Open($path, 'CreateNew', 'ReadWrite', 'Read')
    try {
        $SourceLock.Position = 0
        $SourceLock.CopyTo($snapshot)
        $snapshot.Flush($true)
        $snapshot.Position = 0
        $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($snapshot))
        if ($hash -ne $ExpectedHash) { throw 'Source snapshot hash differs.' }
        return @{ path = $path; sha256 = $hash; bytes = $snapshot.Length
            stream = $snapshot }
    } catch {
        $snapshot.Dispose()
        throw
    }
}

function Assert-LockedSourceSnapshot($SourceLock, $Snapshot, $ExpectedHash) {
    foreach ($stream in @($SourceLock, $Snapshot.stream)) {
        $stream.Position = 0
        $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream))
        if ($hash -ne $ExpectedHash) {
            throw 'Locked source or snapshot changed before child launch.'
        }
    }
}

function Run-Profile($Request) {
    $sourceLock = [IO.File]::Open($Request._source, 'Open', 'Read', 'Read')
    $snapshot = $null
    try {
        $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($sourceLock))
        if ($hash -ne $Request.sha256) { throw 'Source changed after validation.' }
        $output = $Request._output
        if (Test-Path -LiteralPath $output) { throw 'Output already exists.' }
        $null = New-Item -ItemType Directory -Path $output -ErrorAction Stop
        $snapshot = New-LockedSourceSnapshot $sourceLock $output $hash
        Write-JsonAtomic (Join-Path $output 'request.json') $Request
        $arguments = @('--clock-control', 'none', '--cache-control', 'none',
            '--kernel-name-base', 'function', '--kernel-name', $Request.kernel_filter,
            '--launch-skip', "$($Request.launch_skip)", '--launch-count',
            "$($Request.launch_count)")
        foreach ($section in $Request.sections) {
            $arguments += @('--section', $section)
        }
        if ($Request.metrics.Count) {
            $arguments += @('--metrics', ($Request.metrics -join ','))
        }
        $report = Join-Path $output 'profile'
        $arguments += @('--csv', '--log-file', (Join-Path $output 'diagnostic.csv'),
            '--export', $report, $python)
        if ($Request.target -eq 'hardware_probes') {
            $arguments += @('-m', 'benchmarks.hardware_model.hardware_probes')
        } else { $arguments += $Request._source }
        $arguments += @($Request.arguments)
        $arguments += @($Request.output_flag, (Join-Path $output 'benchmark'))
        Write-JsonAtomic (Join-Path $output 'command.json') @{
            executable = $ncu; arguments = $arguments
            working_directory = $Request._runtime_tree; source_sha256 = $hash
            source_path = $Request._source
            source_snapshot = @{ path = $snapshot.path; sha256 = $snapshot.sha256
                bytes = $snapshot.bytes }
            script_tree = $Request._tree; runtime_tree = $Request._runtime_tree
            pythonpath = $Request._pythonpath
            removed_inherited_environment = (Set-EpochRuntimeEnvironment (
                [Environment]::GetEnvironmentVariables('Process')) $Request._runtime_tree)
            ncu_file_version = (Get-Item -LiteralPath $ncu).VersionInfo.FileVersion
            python_file_version = (Get-Item -LiteralPath $python).VersionInfo.FileVersion
        }
        $endTime = [DateTimeOffset]::UtcNow.AddSeconds($Request.timeout_seconds)
        if ($endTime -gt $deadline) { $endTime = $deadline }
        Assert-LockedSourceSnapshot $sourceLock $snapshot $hash
        $code = Invoke-FixedProcess $arguments $Request._runtime_tree (Join-Path $output 'profile') $endTime $Request._pythonpath
        if ($code -ne 0) { throw "Nsight Compute exited $code; inspect raw logs." }
        if (-not (Test-Path -LiteralPath "$report.ncu-rep")) {
            throw 'Nsight Compute did not create a report.'
        }
        $metricsPath = Join-Path $output 'metrics.csv'
        $code = Invoke-FixedProcess @('--import', "$report.ncu-rep", '--page',
            'raw', '--csv', '--log-file', $metricsPath) $Request._runtime_tree (
            Join-Path $output 'import') $endTime $Request._pythonpath
        if ($code -ne 0) { throw "Report import exited $code." }
        $rows = @(Import-Csv -LiteralPath $metricsPath)
        $dataRows = @($rows | Where-Object { $_.ID -match '^\d+$' })
        if ($dataRows.Count -eq 0) { throw 'Imported report has no kernel counter rows.' }
        foreach ($metric in $Request.metrics) {
            if ($metric -notin $rows[0].PSObject.Properties.Name) {
                throw "Requested metric missing from imported report: $metric"
            }
        }
        return @{
            output = $output; counter_kernel_rows = $dataRows.Count
            source_sha256 = $hash; profile_exit_code = 0; import_exit_code = 0
            source_snapshot_path = $snapshot.path
            timing_use = 'Profiled event times are not ordinary performance samples.'
        }
    } finally {
        if ($snapshot) { $snapshot.stream.Dispose() }
        $sourceLock.Dispose()
    }
}

foreach ($fixedPath in @($python, $ncu)) {
    if (-not (Test-Path -LiteralPath $fixedPath -PathType Leaf)) {
        throw "Required fixed executable missing: $fixedPath"
    }
}
$null = Assert-SafePath $sessionRoot $rawRoot
if ($ValidateOnly) {
    if ($RequestPath) { Read-Request $RequestPath | ConvertTo-Json -Depth 20 }
    else { $script:status | ConvertTo-Json -Depth 5 }
    return
}
if ($RequestPath) { throw 'RequestPath is only accepted with ValidateOnly.' }
if (-not $script:status.elevated) { throw 'Start this worker with normal RunAs.' }
if ([DateTimeOffset]::UtcNow -ge $deadline) { throw 'Fixed session deadline has passed.' }
foreach ($directory in @($sessionRoot, (Join-Path $sessionRoot 'queue'),
        (Join-Path $sessionRoot 'claimed'), (Join-Path $sessionRoot 'results'))) {
    $null = Assert-SafePath $directory $rawRoot
    $null = New-Item -ItemType Directory -Path $directory -Force
}
$sessionLock = [IO.File]::Open((Join-Path $sessionRoot 'worker.lock'),
    'OpenOrCreate', 'ReadWrite', 'None')
try {
    $script:status.state = 'idle_disarmed'
    Save-Status
    while (-not $script:stopRequested -and [DateTimeOffset]::UtcNow -lt $deadline) {
        Handle-ControlRequests
        if ($script:stopRequested) { break }
        $released = $false
        try { $released = Test-GpuRelease } catch {
            $script:status.last_error = $_.Exception.Message
        }
        $script:status.state = if ($released) { 'idle_ready' } else { 'idle_disarmed' }
        if ($released) {
            $queue = Join-Path $sessionRoot 'queue'
            $file = Get-ChildItem -LiteralPath $queue -Filter '*.json' -File |
                Sort-Object Name | Select-Object -First 1
            if ($file) {
                try {
                    $request = Read-QueuedRequest $file
                    $null = Claim-Request $file
                } catch {
                    Reject-Request $file $_.Exception.Message
                    continue
                }
                $script:status.state = 'profiling'
                $script:status.current_job = $request.id
                Save-Status
                $receipt = [ordered]@{
                    id = $request.id; action = 'profile'; worker_pid = $PID
                    started_utc = [DateTimeOffset]::UtcNow.ToString('o')
                }
                try {
                    $receipt.result = Run-Profile $request
                    $receipt.state = 'complete'
                } catch {
                    $receipt.state = 'failed'
                    $receipt.error = $_.Exception.Message
                    $script:status.last_error = $receipt.error
                }
                $receipt.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
                Write-Receipt $request.id $receipt
                $script:status.current_job = $null
            }
        }
        Save-Status
        Start-Sleep -Milliseconds 500
    }
    $script:status.state = if ($script:stopRequested) { 'stopped' } else { 'deadline' }
} catch {
    $script:status.state = 'failed'
    $script:status.last_error = $_.Exception.Message
    throw
} finally {
    $script:status.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
    Save-Status
    $sessionLock.Dispose()
}
