variable "aws_region" {
  description = "AWS region the fleet stack deploys into."
  type        = string
  default     = "us-east-2"
}

variable "aws_profile" {
  description = "AWS CLI profile holding the short-term deployer credentials (see bootstrap/cloudshell-iam.sh)."
  type        = string
  default     = "cubie-fleet"
}

variable "stack_name" {
  description = "Name of the RunsOn Fleet stack; prefixes every AWS resource and the GitHub scale-set names."
  type        = string
  default     = "cubie-fleet"
}

variable "github_app_id" {
  description = "App ID of the RunsOn Fleet GitHub App installed on the organization that owns cubie."
  type        = number
}

variable "github_app_private_key_path" {
  description = "Path to the .pem private key generated for the RunsOn Fleet GitHub App."
  type        = string
}

variable "license_key" {
  description = "RunsOn license key (the same key as the existing Flex CloudFormation install; one license covers both products)."
  type        = string
  sensitive   = true
}

variable "alert_email" {
  description = "Email address RunsOn sends alerts to (requires confirmation on first deploy)."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the stack-owned VPC; public /20 subnets are carved from it per availability zone."
  type        = string
  default     = "10.2.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones given a public subnet. All three, because the GPU spot pools span them."
  type        = list(string)
  default     = ["us-east-2a", "us-east-2b", "us-east-2c"]
}
