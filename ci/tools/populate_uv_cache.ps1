##  Bake-time uv wheel-cache populate for the cubie Windows GPU CI AMI.

$ErrorActionPreference = 'Stop'

$cacheDir = 'C:\uv-cache'
$pyprojectPath = 'C:\Windows\Temp\pyproject.toml'
$uvReleaseUrl =
    'https://api.github.com/repos/astral-sh/uv/releases/latest'

# (python spec, extra) pairs the Windows CUDA matrix installs.
$combos = @(
    @('3.10', 'dev12'), @('3.10', 'dev13'),
    @('3.11', 'dev-mlir12'), @('3.11', 'dev-mlir13'),
    @('3.14', 'dev12'), @('3.14', 'dev13'),
    @('3.14', 'dev-mlir12'), @('3.14', 'dev-mlir13')
)

function Get-UvExecutable {
    $latest = Invoke-RestMethod -Uri $uvReleaseUrl -UseBasicParsing
    $asset = $latest.assets |
        Where-Object { $_.name -eq 'uv-x86_64-pc-windows-msvc.zip' } |
        Select-Object -First 1
    if (-not $asset) {
        throw 'No uv-x86_64-pc-windows-msvc.zip on the latest release.'
    }
    $zipPath = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest -Uri $asset.browser_download_url `
        -OutFile $zipPath -UseBasicParsing
    $binDir = Join-Path $env:TEMP 'uv-bin'
    Expand-Archive -Path $zipPath -DestinationPath $binDir -Force
    $exe = Get-ChildItem -Path $binDir -Recurse -Filter 'uv.exe' |
        Select-Object -First 1
    if (-not $exe) {
        throw 'uv.exe missing from the uv release archive.'
    }
    return $exe.FullName
}

function Get-ToolcachePython {
    param([Parameter(Mandatory = $true)][string]$Spec)
    $dir = Get-ChildItem -Path 'C:\hostedtoolcache\windows\Python' `
        -Directory -Filter "$Spec.*" -ErrorAction SilentlyContinue |
        Sort-Object -Property Name |
        Select-Object -Last 1
    if (-not $dir) {
        throw "Python $Spec is not in the toolcache."
    }
    return (Join-Path $dir.FullName 'x64\python.exe')
}

function Invoke-Uv {
    param([Parameter(Mandatory = $true)][string]$Uv,
          [Parameter(Mandatory = $true)][string[]]$Arguments)
    try {
        # uv reports progress on stderr; keep that non-terminating.
        $ErrorActionPreference = 'Continue'
        & $Uv @Arguments 2>&1 | ForEach-Object { "$_" } | Write-Host
    } finally {
        $ErrorActionPreference = 'Stop'
    }
    if ($LASTEXITCODE -ne 0) {
        throw "uv $($Arguments -join ' ') exited $LASTEXITCODE."
    }
}

if (-not (Test-Path -Path $pyprojectPath)) {
    throw "No pyproject.toml at $pyprojectPath."
}
$uv = Get-UvExecutable
$env:UV_CACHE_DIR = $cacheDir
New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
$stagingDir = Join-Path $env:TEMP 'uv-cache-staging'

foreach ($combo in $combos) {
    $spec = $combo[0]
    $extra = $combo[1]
    $python = Get-ToolcachePython -Spec $spec
    $venv = Join-Path $stagingDir "venv-$spec-$extra"
    $elapsed = Measure-Command {
        Invoke-Uv -Uv $uv -Arguments @('venv', $venv, '--python', $python)
        Invoke-Uv -Uv $uv -Arguments @(
            'pip', 'install',
            '--python', (Join-Path $venv 'Scripts\python.exe'),
            '-r', $pyprojectPath, '--extra', $extra
        )
    }
    Remove-Item -Path $venv -Recurse -Force
    Write-Host ("PREP-MARKER uv-cache ${spec}/${extra}: " +
        "$([int]$elapsed.TotalSeconds)s")
}

$bytes = (Get-ChildItem -Path $cacheDir -Recurse -File |
    Measure-Object -Property Length -Sum).Sum
Write-Host ("PREP-MARKER uv-cache total: {0:N1} GB" -f ($bytes / 1GB))
