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
    $keyPath = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion' +
        '\DriverSearching'
    New-Item -Path $keyPath -Force | Out-Null
    New-ItemProperty -Path $keyPath -Name 'SearchOrderConfig' `
        -PropertyType DWord -Value 0 -Force | Out-Null
    Write-Host 'Driver search restricted to the local driver store.'
}

function Get-ToolcacheRoot {
    $default = 'C:\hostedtoolcache\windows'
    if (Test-Path -Path $default) {
        return $default
    }
    $found = Get-ChildItem -Path 'C:\' -Directory `
        -Filter '*toolcache*' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($found) {
        return (Join-Path $found.FullName 'windows')
    }
    return $default
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
        Write-Host "Python $Spec already in the toolcache."
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
        throw "No stable win32/x64 python-versions release matches $Spec."
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
        & .\setup.ps1
    } finally {
        Pop-Location
    }
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
    $dirs = @($hits |
        ForEach-Object { $_.Directory.Parent.FullName } |
        Sort-Object -Unique)
    if ($dirs.Count -ne 1) {
        Write-Host ("Runner agent refresh skipped: " +
            "$($dirs.Count) candidate installs found " +
            "($($dirs -join ', '))")
        return $null
    }
    return $dirs[0]
}

function Update-RunnerAgent {
    $runnerDir = Get-RunnerDirectory
    if (-not $runnerDir) {
        return
    }
    $listener = Join-Path $runnerDir 'bin\Runner.Listener.exe'
    $current = (& $listener --version).Trim()
    $latest = Invoke-RestMethod -Uri $runnerReleaseUrl -UseBasicParsing
    $version = $latest.tag_name.TrimStart('v')
    Write-Host ("Runner agent at ${runnerDir}: " +
        "installed $current, latest $version")
    if ($current -eq $version) {
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
    $updated = (& $listener --version).Trim()
    if ($updated -ne $version) {
        throw "Runner agent update failed: reports $updated."
    }
    Write-Host "Runner agent updated to $updated."
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
