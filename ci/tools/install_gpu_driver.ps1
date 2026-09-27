##  Install and verify the NVIDIA GPU driver for the cubie Windows GPU CI
##  AMI. Adapted from runs-on/runner-images-for-aws (Install-GPU.ps1).
##
##  Installs the AWS GRID driver on Windows Server 2025 -- one package
##  that covers G4dn (T4), G5 (A10G), and G6 (L4), so the single AMI runs
##  on any of those spot families. The CUDA toolkit is NOT installed here:
##  cubie resolves the toolkit from pip wheels
##  (cubie-numba-cuda-mlir[cu12]/[cu13]); only the kernel driver is
##  needed on the host.
##
##  Called twice by the Packer build with a windows-restart between: the
##  first run installs the driver, the second (marker present) verifies
##  nvidia-smi.

$ErrorActionPreference = 'Stop'

$markerDirectory    = 'C:\ProgramData\RunsOn'
$markerFile         = Join-Path $markerDirectory 'gpu-installed.txt'
$downloadDirectory  = Join-Path $markerDirectory 'GPU'
$driverLogDirectory = Join-Path $downloadDirectory 'Logs'
$driverInstaller    = Join-Path $downloadDirectory 'nvidia-grid-driver.exe'
$driverBucketUrl    = 'https://ec2-windows-nvidia-drivers.s3.amazonaws.com'
$gridLicenseRegistryPath = 'HKLM:\SOFTWARE\NVIDIA Corporation\Global\GridLicensing'

function New-RunsOnDirectory {
    param([string]$Path)
    if (-not (Test-Path -Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [int[]]$ValidExitCodes = @(0)
    )
    Write-Host "Running: $FilePath $($ArgumentList -join ' ')"
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -Wait -PassThru -NoNewWindow
    if ($ValidExitCodes -notcontains $process.ExitCode) {
        throw "Command failed with exit code $($process.ExitCode): $FilePath"
    }
    return $process.ExitCode
}

function Get-LatestAwsGridDriverKey {
    [xml]$listing = (Invoke-WebRequest -Uri "$driverBucketUrl/?prefix=latest/" `
        -UseBasicParsing).Content
    $keys = @($listing.ListBucketResult.Contents | ForEach-Object { $_.Key })

    # AWS ships one GRID DCH driver, e.g.
    #   latest/596.36_grid_win10_win11_server2022_64bit_dch_international_aws_swl.exe
    # It is a multi-OS DCH package that runs on Server 2025 too (there is no
    # separate "server2025" build), so match any GRID .exe rather than a
    # specific Windows version. The post-install Assert-CudaAtLeast13 guards
    # against an unexpectedly old driver.
    $matchingKey = $keys |
        Where-Object { $_ -match '^latest/.*grid.*\.exe$' } |
        Select-Object -First 1

    if (-not $matchingKey) {
        Write-Host "Available driver keys under latest/:"
        $keys | ForEach-Object { Write-Host "  $_" }
        throw "No AWS GRID driver (.exe) found under $driverBucketUrl/latest/."
    }

    return $matchingKey
}

function Ensure-AwsGridDriverInstaller {
    New-RunsOnDirectory -Path $downloadDirectory
    New-RunsOnDirectory -Path $driverLogDirectory

    if (Test-Path -Path $driverInstaller) {
        Write-Host "Reusing AWS GRID driver installer from $driverInstaller"
        return
    }

    $driverKey = Get-LatestAwsGridDriverKey
    $driverUri = "$driverBucketUrl/$driverKey"
    Write-Host "Downloading AWS GRID driver from $driverUri"
    Invoke-WebRequest -Uri $driverUri -OutFile $driverInstaller -UseBasicParsing
}

function Install-AwsGridDriver {
    Ensure-AwsGridDriverInstaller

    Invoke-ExternalCommand -FilePath $driverInstaller -ArgumentList @(
        '-s',
        '-n',
        'Display.Driver',
        "-log:$driverLogDirectory",
        '-loglevel:6'
    ) -ValidExitCodes @(0, 1)

    New-Item -Path $gridLicenseRegistryPath -Force | Out-Null
    New-ItemProperty `
        -Path $gridLicenseRegistryPath `
        -Name 'NvCplDisableManageLicensePage' `
        -PropertyType DWord `
        -Value 1 `
        -Force | Out-Null
}

function Get-NvidiaSmiPath {
    # The NVIDIA Windows display driver installs nvidia-smi.exe into
    # %SystemRoot%\System32, which is on PATH. Resolve via PATH first, then
    # that canonical location.
    $command = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $system32Path = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
    if (Test-Path -Path $system32Path) {
        return $system32Path
    }
    throw "nvidia-smi.exe not found after driver install"
}

function Assert-CudaAtLeast13 {
    param([Parameter(Mandatory = $true)][string]$NvidiaSmiPath)

    # cubie's matrix includes a cuda13 leg, whose pip wheels require a driver
    # that exposes CUDA >= 13. Fail the bake loudly if the resolved GRID
    # driver is older, rather than shipping an AMI that silently breaks
    # cuda13 jobs.
    $smiOutput = & $NvidiaSmiPath
    $match = $smiOutput | Select-String -Pattern 'CUDA Version:\s*(\d+)\.(\d+)' | Select-Object -First 1
    if (-not $match) {
        throw "Could not parse CUDA version from nvidia-smi output."
    }
    $cudaMajor = [int]$match.Matches[0].Groups[1].Value
    $cudaFull = "$($match.Matches[0].Groups[1].Value).$($match.Matches[0].Groups[2].Value)"
    Write-Host "Driver exposes CUDA $cudaFull"
    if ($cudaMajor -lt 13) {
        throw "Baked driver exposes CUDA $cudaFull; cubie's cuda13 matrix leg requires CUDA >= 13. Pin a newer driver."
    }
}

# Second invocation (post-restart): verify and clear the marker.
if (Test-Path -Path $markerFile) {
    $nvidiaSmi = Get-NvidiaSmiPath
    & $nvidiaSmi
    & $nvidiaSmi -L
    Assert-CudaAtLeast13 -NvidiaSmiPath $nvidiaSmi
    Remove-Item -Path $markerFile -Force
    exit 0
}

# First invocation: install the driver (twice, to cover a reboot-needed
# state on the initial bind).
New-RunsOnDirectory -Path $markerDirectory
Set-Content -Path $markerFile -Value 'installed'
Install-AwsGridDriver
Install-AwsGridDriver
