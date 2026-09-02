# RunsOn Fleet stack for cubie's GPU CI (ci_cuda_tests.yml).
#
# Fleet (runs-on.com/docs/flex-vs-fleet/) registers GitHub runner scale
# sets and launches EC2 capacity from *assigned-job* demand, so the
# workflow's `strategy.max-parallel` bounds runner demand on the RunsOn
# side as well, keeping it inside the 16-vCPU G/VT spot quota.
#
# Deliberately no `schedule` (hot/stopped standby) on the fleets: warm
# pool inventory uses on-demand EC2 capacity, which this account cannot
# launch for G instances. All capacity comes from cold spot launches.

terraform {
  required_version = ">= 1.5.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.45"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

locals {
  # Custom image catalog. Same contract as the Flex .github/runs-on.yml
  # `images:` block: RunsOn picks the newest AMI matching `name` owned
  # by this account ("self"), so the Packer bake workflow
  # (.github/workflows/build-windows-gpu-ami.yml) keeps working
  # unchanged.
  images = {
    cubie-win-gpu = {
      platform = "windows"
      arch     = "x64"
      owner    = "self"
      name     = "cubie-win-gpu-*"
    }
  }

  # max-parallel: 4 xlarge legs fill the 16-vCPU G/VT spot quota.
  runners = {
    gpu-linux-2xl = {
      # Three GPU families in both sizes; price-capacity allocation chooses.
      family = [
        "g4dn.2xlarge", "g5.2xlarge", "g6.2xlarge",
        "g4dn.xlarge", "g5.xlarge", "g6.xlarge",
      ]
      image = "ubuntu24-gpu-x64"
      spot  = "price-capacity-optimized"
      # No s3-cache (Magic Cache) extra, deliberately: it requires a
      # runs-on/action@v2 step in every job (without one the sidecar
      # intercepts the GitHub artifact service and CreateArtifact
      # fails on a non-JSON response), and RunsOn documents that
      # the shared cache bucket must not be enabled for
      # runners public repositories can use; cubie is public.
    }
    gpu-windows-g5 = {
      # The AMI bake family only, so the driver is bound before first boot.
      family = ["g5.xlarge"]
      image  = "cubie-win-gpu"
      spot   = "price-capacity-optimized"
      # No s3-cache (Magic Cache) extra, deliberately: it requires a
      # runs-on/action@v2 step in every job (without one the sidecar
      # intercepts the GitHub artifact service and CreateArtifact
      # fails on a non-JSON response), and RunsOn documents that
      # the shared cache bucket must not be enabled for
      # runners public repositories can use; cubie is public.
    }
  }

  # One fleet per OS; each maps to one GitHub runner scale set named
  # <stack_name>-<fleet name>. Workflows target them with
  #   runs-on: runs-on/fleet=gpu-linux/env=production
  # No runner_group: scale sets register into the organization's
  # default runner group (custom groups need a paid GitHub plan).
  fleets = {
    gpu-linux = {
      timezone = "UTC"
      runner   = "gpu-linux-2xl"
    }
    gpu-windows = {
      timezone = "UTC"
      runner   = "gpu-windows-g5"
    }
  }
}

# Self-contained network: the stack owns its VPC and shares no
# networking with any other stack, so nothing outside this
# configuration can take the fleet's network down. Public subnets in
# all three AZs: the GPU spot pools span them, so full AZ coverage
# maximises reachable pools. Public-only (no NAT) keeps the VPC free.
resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name    = "${var.stack_name}-vpc"
    stack   = var.stack_name
    project = "cubie"
  }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name    = "${var.stack_name}-igw"
    stack   = var.stack_name
    project = "cubie"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = {
    Name    = "${var.stack_name}-public"
    stack   = var.stack_name
    project = "cubie"
  }
}

resource "aws_subnet" "public" {
  count = length(var.availability_zones)

  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.stack_name}-public-${var.availability_zones[count.index]}"
    stack   = var.stack_name
    project = "cubie"
  }
}

resource "aws_route_table_association" "public" {
  count = length(var.availability_zones)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

module "runs_on_fleet" {
  source  = "runs-on/runs-on/aws//modules/fleet"
  version = "3.1.3"

  stack_name  = var.stack_name
  environment = "production"

  # Organization mode: a GitHub App installed on exactly one
  # organization, with organization self-hosted runner write access.
  github_app_id          = var.github_app_id
  github_app_private_key = file(var.github_app_private_key_path)

  license_key = var.license_key
  email       = var.alert_email

  images  = local.images
  runners = local.runners
  fleets  = local.fleets

  vpc_id            = aws_vpc.this.id
  public_subnet_ids = aws_subnet.public[*].id

  # CI runs three times a week; fargate_spot keeps the always-on Fleet
  # worker's idle cost down, and the Fleet runtime reconciles any
  # in-flight jobs if the Fargate task is interrupted.
  app_size              = "small"
  app_capacity_provider = "fargate_spot"

  # A full-matrix leg (install + flake8 + real-GPU pytest) fits well
  # inside an hour today; 120 leaves headroom for suite growth without
  # letting a hung leg hold the quota for long.
  runner_max_runtime = 120

  tags = {
    project = "cubie"
  }
}

# Turn off ECS Container Insights, which the module enables with no input to
# set. The module re-asserts it on every apply, so plantimestamp() re-runs
# this every apply too; tofu plan therefore always shows this resource as
# replaced.
resource "terraform_data" "disable_container_insights" {
  triggers_replace = plantimestamp()

  provisioner "local-exec" {
    command = join(" ", [
      "aws ecs update-cluster-settings",
      "--cluster ${var.stack_name}",
      "--settings name=containerInsights,value=disabled",
      "--region ${var.aws_region}",
      "--profile ${var.aws_profile}",
      "--output text",
    ])
  }

  depends_on = [module.runs_on_fleet]
}
