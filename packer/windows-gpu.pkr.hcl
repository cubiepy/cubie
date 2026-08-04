packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = ">= 1.8.0, < 2.0.0"
    }
  }
}

# Region MUST match the region your RunsOn CloudFormation stack deploys
# runners into (custom AMIs are region-local).
variable "region" {
  type    = string
  default = "us-east-2"
}

# Bake needs a real GPU; g5.xlarge only (the Windows fleet's exact shape), spot, via the builder role's Fleet IAM actions.
variable "spot_instance_types" {
  type    = list(string)
  default = ["g5.xlarge"]
}

# GitHub run ID stamped on the builder instance for post-run cleanup.
variable "build_run_id" {
  type    = string
  default = ""
}

# Max hourly spot bid. A ceiling only EXCLUDES pools -- AWS charges the
# live spot rate (itself capped at on-demand), never this number -- so it
# just needs to clear every listed type's market spot price (all well
# under $0.65 for these xlarge/2xlarge GPU types). 1.50 leaves headroom.
variable "spot_price" {
  type    = string
  default = "1.50"
}

# Empty -> Packer picks a subnet in the default VPC. Set explicitly if the
# account has no default VPC.
variable "subnet_id" {
  type    = string
  default = ""
}

# AWS account that publishes the (public) RunsOn base AMIs.
variable "runs_on_owner" {
  type    = string
  default = "135269210855"
}

# RunsOn Windows 2025 base image; already carries the RunsOn agent.
variable "source_ami_name" {
  type    = string
  default = "runs-on-v2.2-windows25-full-x64-*"
}

locals {
  timestamp = formatdate("YYYYMMDD-hhmmss", timestamp())
}

source "amazon-ebs" "windows_gpu" {
  region                                     = var.region
  spot_instance_types                        = var.spot_instance_types
  spot_allocation_strategy                   = "capacity-optimized"
  spot_price                                 = var.spot_price
  subnet_id                                  = var.subnet_id
  associate_public_ip_address                = true
  temporary_security_group_source_public_ip  = true
  ebs_optimized                              = true
  force_deregister                           = true
  force_delete_snapshot                      = true

  communicator   = "winrm"
  winrm_username = "Administrator"
  winrm_use_ssl  = true
  winrm_insecure = true
  winrm_timeout  = "12m"

  aws_polling {
    delay_seconds = 30
    max_attempts  = 300
  }

  source_ami_filter {
    filters = {
      name                = var.source_ami_name
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    owners      = [var.runs_on_owner]
    most_recent = true
  }

  # EC2Launch user-data enabling WinRM over HTTPS so Packer can connect.
  # Bootstrap pattern from runs-on/runner-images-for-aws
  # (patches/windows/templates/windows25-gpu-x64.pkr.hcl).
  user_data = <<EOF
<powershell>
Enable-PSRemoting -SkipNetworkProfileCheck -Force
winrm set winrm/config/service/auth '@{Basic="true"}'
Set-Service -Name WinRM -StartupType Automatic
$Cert = New-SelfSignedCertificate -CertstoreLocation Cert:\LocalMachine\My -DnsName "cubie-packer"
Get-ChildItem WSMan:\Localhost\Listener | Where-Object Keys -eq "Transport=HTTP" | Remove-Item -Recurse
New-Item -Path WSMan:\LocalHost\Listener -Transport HTTPS -Address * -CertificateThumbPrint $Cert.Thumbprint -Force
New-NetFirewallRule -DisplayName "Windows Remote Management (HTTPS-In)" -Name "Windows Remote Management (HTTPS-In)" -Profile Any -LocalPort 5986 -Protocol TCP
</powershell>
<persist>false</persist>
EOF

  ami_name        = "cubie-win-gpu-${local.timestamp}"
  ami_description = "RunsOn Windows 2025 + NVIDIA GRID driver (T4/A10G/L4) for cubie CUDA CI"

  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name      = "cubie-win-gpu-${local.timestamp}"
    Project   = "cubie"
    Purpose   = "cuda-ci-windows-gpu"
    BaseImage = var.source_ami_name
  }

  # Builder-instance tags; the cleanup workflow terminates by these.
  run_tags = {
    Purpose    = "cubie-ami-bake"
    BuildRunId = var.build_run_id
  }
}

build {
  sources = ["source.amazon-ebs.windows_gpu"]

  # Install the AWS GRID driver, restart, then verify (marker pattern in
  # the script short-circuits the second run to nvidia-smi verification).
  provisioner "powershell" {
    pause_before = "2m0s"
    scripts      = ["ci/tools/install_gpu_driver.ps1"]
  }

  provisioner "windows-restart" {
    restart_timeout = "30m"
  }

  provisioner "powershell" {
    scripts = ["ci/tools/install_gpu_driver.ps1"]
  }

  # Toolcache Pythons, latest runner agent, local-only driver search.
  provisioner "powershell" {
    scripts = ["ci/tools/prepare_ci_image.ps1"]
  }

  # Stage pyproject.toml for dependency resolution during the bake.
  provisioner "file" {
    source      = "pyproject.toml"
    destination = "C:/Windows/Temp/pyproject.toml"
  }

  # Pre-download the CUDA test matrix's wheels into C:\uv-cache.
  provisioner "powershell" {
    scripts = ["ci/tools/populate_uv_cache.ps1"]
  }

  # EC2Launch reset + Sysprep shutdown; first boot reruns the RunsOn bootstrap.
  provisioner "powershell" {
    inline = [
      "& 'C:/Program Files/Amazon/EC2Launch/ec2launch' reset --block",
      "& 'C:/Program Files/Amazon/EC2Launch/ec2launch' sysprep --shutdown --block",
    ]
  }
}
