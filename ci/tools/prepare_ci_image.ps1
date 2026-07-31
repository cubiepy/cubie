##  Bake-time prep for the cubie Windows GPU CI AMI (run by Packer).

$ErrorActionPreference = 'Stop'

# Python versions the CUDA test matrix uses on Windows runners.
$pythonVersions = @('3.10', '3.11')

$pythonManifestUrl = 'https://raw.githubusercontent.com/actions/' +
    'python-versions/main/versions-manifest.json'
$runnerReleaseUrl =
    'https://api.github.com/repos/actions/runner/releases/latest'

function Set-LocalOnlyDriverSearch {
    # PnP device installs use the local driver store only.
    $prefKey = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion' +
        '\DriverSearching'
    New-Item -Path $prefKey -Force | Out-Null
    New-ItemProperty -Path $prefKey -Name 'SearchOrderConfig' `
        -PropertyType DWord -Value 0 -Force | Out-Null
    # The policy key wins over the preference key where both exist.
    $polKey = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows' +
        '\DriverSearching'
    New-Item -Path $polKey -Force | Out-Null
    New-ItemProperty -Path $polKey -Name 'DontSearchWindowsUpdate' `
        -PropertyType DWord -Value 1 -Force | Out-Null
    Write-Host 'PREP-MARKER driver-search: local-only'
}

function Get-ToolcacheRoot {
    $root = 'C:\hostedtoolcache\windows'
    if (-not (Test-Path -Path $root)) {
        $found = Get-ChildItem -Path 'C:\' -Directory `
            -Filter '*toolcache*' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) {
            $root = Join-Path $found.FullName 'windows'
        }
    }
    # A root without pre-baked Pythons is not the runner's toolcache.
    $probe = Get-ChildItem -Path (Join-Path $root 'Python') `
        -Directory -ErrorAction SilentlyContinue
    if (-not $probe) {
        throw "No pre-baked Pythons under $root; wrong toolcache root."
    }
    return $root
}

function Install-ToolcachePython {
    param(
        [Parameter(Mandatory = $true)][string]$Spec,
        [Parameter(Mandatory = $true)][string]$ToolsDirectory,
        [Parameter(Mandatory = $true)]$Manifest
    )
    $pythonRoot = Join-Path $ToolsDirectory 'Python'
    $cached = Get-ChildItem -Path $pythonRoot -Directory `
        -Filter "$Spec.*" -ErrorAction SilentlyContinue
    if ($cached) {
        Write-Host "PREP-MARKER python-${Spec}: already present"
        return
    }
    # Newest stable entry that still ships a Windows build.
    $release = $Manifest |
        Where-Object {
            $_.stable -and $_.version -like "$Spec.*" -and
            ($_.files | Where-Object {
                $_.platform -eq 'win32' -and $_.arch -eq 'x64'
            })
        } |
        Select-Object -First 1
    if (-not $release) {
        throw "No stable win32/x64 python-versions release for $Spec."
    }
    $file = $release.files |
        Where-Object { $_.platform -eq 'win32' -and $_.arch -eq 'x64' } |
        Select-Object -First 1
    Write-Host "Installing Python $($release.version) into $pythonRoot"
    $zipPath = Join-Path $env:TEMP "python-$($release.version).zip"
    Invoke-WebRequest -Uri $file.download_url -OutFile $zipPath `
        -UseBasicParsing
    $extractDir = Join-Path $env:TEMP "python-$($release.version)"
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
    # setup.ps1 installs the actions/setup-python toolcache layout.
    $env:AGENT_TOOLSDIRECTORY = $ToolsDirectory
    Push-Location $extractDir
    try {
        # Vendored script; its non-terminating errors stay non-fatal.
        $ErrorActionPreference = 'Continue'
        & .\setup.ps1
    } finally {
        $ErrorActionPreference = 'Stop'
        Pop-Location
    }
    $marker = Join-Path $pythonRoot "$($release.version)\x64.complete"
    if (-not (Test-Path -Path $marker)) {
        throw "Python $($release.version) install left no $marker."
    }
    Write-Host "PREP-MARKER python-${Spec}: installed $($release.version)"
}

function Get-RunnerDirectory {
    $roots = @(
        'C:\actions-runner', 'C:\runners', 'C:\runner', 'C:\a',
        'C:\actions', 'C:\ProgramData\RunsOn'
    )
    $hits = foreach ($root in $roots) {
        if (Test-Path -Path $root) {
            Get-ChildItem -Path $root -Recurse -Depth 3 `
                -Filter 'Runner.Listener.exe' `
                -ErrorAction SilentlyContinue
        }
    }
    if (-not $hits) {
        $skip = @(
            'Windows', 'Program Files', 'Program Files (x86)',
            'Users', 'PerfLogs', 'hostedtoolcache'
        )
        $shallow = Get-ChildItem -Path 'C:\' -Directory `
            -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notin $skip }
        $hits = foreach ($dir in $shallow) {
            Get-ChildItem -Path $dir.FullName -Recurse -Depth 3 `
                -Filter 'Runner.Listener.exe' `
                -ErrorAction SilentlyContinue
        }
    }
    # The listener lives in <runner root>\bin.
    $dirs = @($hits |
        Where-Object { $_.Directory.Name -eq 'bin' } |
        ForEach-Object { $_.Directory.Parent.FullName } |
        Sort-Object -Unique)
    if ($dirs.Count -ne 1) {
        Write-Host ("PREP-MARKER runner-agent: skipped " +
            "($($dirs.Count) candidate installs: $($dirs -join ', '))")
        return $null
    }
    return $dirs[0]
}

function Get-RunnerVersion {
    # File metadata, not execution: ProductVersion is the release
    # version plus '+<build sha>', and running the listener would leave
    # a non-zero $LASTEXITCODE for Packer's `exit $LastExitCode`.
    param([Parameter(Mandatory = $true)][string]$Listener)
    $product = (Get-Item -Path $Listener).VersionInfo.ProductVersion
    if (-not $product) {
        return $null
    }
    return "$product".Split('+')[0].Trim()
}

function Update-RunnerAgent {
    $runnerDir = Get-RunnerDirectory
    if (-not $runnerDir) {
        return
    }
    $listener = Join-Path $runnerDir 'bin\Runner.Listener.exe'
    $current = Get-RunnerVersion -Listener $listener
    if (-not $current) {
        Write-Host ("PREP-MARKER runner-agent: skipped " +
            "(no version metadata on $listener)")
        return
    }
    try {
        $latest = Invoke-RestMethod -Uri $runnerReleaseUrl `
            -UseBasicParsing
    } catch {
        Write-Host ("PREP-MARKER runner-agent: skipped " +
            "(release query failed: $_)")
        return
    }
    $version = $latest.tag_name.TrimStart('v')
    if ($current -eq $version) {
        Write-Host ("PREP-MARKER runner-agent: current $current " +
            "($runnerDir)")
        return
    }
    $assetName = "actions-runner-win-x64-$version.zip"
    $asset = $latest.assets |
        Where-Object { $_.name -eq $assetName } |
        Select-Object -First 1
    if (-not $asset) {
        throw "actions/runner $($latest.tag_name) has no $assetName."
    }
    $zipPath = Join-Path $env:TEMP $assetName
    Invoke-WebRequest -Uri $asset.browser_download_url `
        -OutFile $zipPath -UseBasicParsing
    # Overlay: standard runner files are replaced, extra files kept.
    Expand-Archive -Path $zipPath -DestinationPath $runnerDir -Force
    $updated = Get-RunnerVersion -Listener $listener
    if ($updated -ne $version) {
        throw "Runner agent update failed: reports $updated."
    }
    Write-Host ("PREP-MARKER runner-agent: updated " +
        "$current -> $updated ($runnerDir)")
}

Set-LocalOnlyDriverSearch

$toolsDirectory = Get-ToolcacheRoot
Write-Host "Toolcache root: $toolsDirectory"
$manifest = Invoke-RestMethod -Uri $pythonManifestUrl -UseBasicParsing
foreach ($spec in $pythonVersions) {
    Install-ToolcachePython -Spec $spec `
        -ToolsDirectory $toolsDirectory -Manifest $manifest
}

Update-RunnerAgent

# Packer exits with the wrapper's $LastExitCode; leave it clean.
exit 0
