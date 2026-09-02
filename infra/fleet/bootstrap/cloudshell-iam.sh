#!/usr/bin/env bash
# One-shot AWS CloudShell bootstrap for the fleet deployer identity.
#
# Paste this whole file into an AWS CloudShell session (console ->
# CloudShell icon). It creates:
#   1. two customer-managed policies, `cubie-fleet-deployer` and
#      `cubie-fleet-deployer-scoped` -- together the minimal
#      permission set `tofu apply` in infra/fleet needs (see the
#      policy documents below for the per-service rationale; the
#      split exists only for IAM's 6144-character policy size limit);
#   2. a role `cubie-fleet-deployer` assumable by IAM identities in
#      this account only, with both policies attached.
#
# Local access is the `cubie-fleet` profile in ~/.aws/config: `role_arn`
# for this role, `source_profile` naming an IAM user's key. The CLI
# assumes the role per call and refreshes the 1-hour session.
# Rerun this file to republish permissions; live sessions pick them up.
set -euo pipefail

REGION="ap-northeast-2"
# Regions the cost dashboard reads instance and spot history from.
HISTORY_REGIONS=("${REGION}" "us-east-2")
HISTORY_REGIONS_JSON=$(printf '"%s",' "${HISTORY_REGIONS[@]}")
HISTORY_REGIONS_JSON="[${HISTORY_REGIONS_JSON%,}]"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Permissions, by statement:
# - ReadOnly: region-locked Describe/Get/List across the services the
#   RunsOn Fleet Terraform module touches, plus read-only
#   CloudFormation and Service Quotas for diagnostics, and
#   cloudtrail:LookupEvents for reading the free 90-day management-event
#   history (instance launch/terminate times, used by the CI
#   cost/timeline report). Reads carry no secret material:
#   secretsmanager:GetSecretValue is NOT here -- it lives in
#   SecretsScoped, bound to this stack's secret prefix.
# - HistoryReadOnly: CloudTrail and spot-price reads across every
#   region in HISTORY_REGIONS, for runs from before a region move.
#   Read-only, and carries no secret material.
# - CostExplorerReadOnly: read-only Cost Explorer for the CI
#   cost/usage report. Cost Explorer is a global service reached
#   through us-east-1, so it CANNOT sit in the region-locked ReadOnly
#   statement (aws:RequestedRegion would deny it), so it is un-region-locked.
#   Read-only and carries no secret material.
# - Ec2Provision: creation of brand-new EC2 networking/template
#   resources only -- creating a resource cannot touch an existing
#   one, so these stay region-locked but otherwise unscoped.
#   RunInstances is deliberately absent: the deployer never launches
#   instances -- only the fleet runtime's own role does.
# - Ec2TagOnCreate: tagging only as part of those create calls
#   (ec2:CreateAction). Standalone CreateTags is NOT granted on
#   arbitrary resources: it would let the deployer tag any foreign
#   resource stack=cubie-fleet and then mutate or destroy it through
#   the tag-scoped statements below.
# - Ec2StackMutate: attribute/rule/route/tag changes bound to
#   resources already tagged stack=cubie-fleet. The provider tags on
#   create (tag specifications), so an apply can immediately modify
#   what it just created, and the module tags every SG and launch
#   template with stack=<stack_name>.
# - Ec2SgRuleLeg: security-group rule actions on the rule ARN.
# - Ec2ScopedDestroy: terminate/delete only for EC2 resources tagged
#   stack=cubie-fleet, so the credentials cannot touch instances,
#   VPCs, or security groups belonging to anything else.
# - S3/Secrets/Dynamo/Lambda/Logs/Sns/Ecs/Events/Cw *Scoped: full
#   management bound to this stack's ARN name prefixes. Resource ARNs
#   bind regardless of which endpoint a call reaches, which also
#   covers S3 calls made through the global endpoint.
# - IamScoped: IAM role/policy/instance-profile management restricted
#   to names starting with cubie-fleet or runs-on. Attach/detach and
#   PassRole live in their own statements: IamAttachScoped allowlists
#   (iam:PolicyARN) exactly the managed policies the module attaches,
#   so a broad AWS-managed policy such as AdministratorAccess cannot
#   be attached to a deployer-controlled role in one call, and
#   IamPassScoped limits PassRole to the services the module actually
#   passes roles to (EC2, Lambda, ECS tasks).
#   RESIDUAL RISK, stated plainly: the module manages inline role
#   policies, so iam:PutRolePolicy on cubie-fleet*/runs-on* roles is
#   required and cannot be content-restricted; likewise the deployer
#   can author cubie-fleet* customer policies with arbitrary content.
#   A leaked live session can therefore still self-escalate by
#   writing a broad policy onto a role it creates. Fully closing that
#   needs a permissions boundary on every module-created role, and
#   runs-on/runs-on v3.1.3 applies permission_boundary_arn only to
#   the EC2 instance role, not the control-plane roles. Until the
#   module supports a boundary on all roles, treat live deployer
#   credentials as account-admin-equivalent for their 1-hour life.
# - ServiceLinkedRoles: lets the first ECS/spot/autoscaling use in the
#   account create AWS's own service-linked roles, nothing else.
# - DenySelfMutation: the deployer's own role and policies match the
#   cubie-fleet* patterns above, so without this deny the role could
#   rewrite its own policy (CreatePolicyVersion) into anything.
#   Changing deployer permissions is CloudShell's job: edit this file
#   and rerun it.
#
# Part 1: region-locked reads plus the create-only surface.
cat > /tmp/cubie-fleet-deployer-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadOnly",
      "Effect": "Allow",
      "Action": [
        "ec2:Describe*",
        "ec2:Get*",
        "ecs:Describe*",
        "ecs:List*",
        "logs:Describe*",
        "logs:Get*",
        "logs:List*",
        "logs:FilterLogEvents",
        "logs:StartQuery",
        "logs:StopQuery",
        "cloudformation:Describe*",
        "cloudformation:List*",
        "cloudformation:Get*",
        "cloudtrail:LookupEvents",
        "servicequotas:Get*",
        "servicequotas:List*",
        "sns:Get*",
        "sns:List*",
        "cloudwatch:Describe*",
        "cloudwatch:Get*",
        "cloudwatch:List*",
        "application-autoscaling:Describe*",
        "application-autoscaling:List*",
        "events:Describe*",
        "events:List*",
        "scheduler:Get*",
        "scheduler:List*",
        "dynamodb:Describe*",
        "dynamodb:List*",
        "lambda:Get*",
        "lambda:List*",
        "secretsmanager:Describe*",
        "secretsmanager:List*",
        "ssm:DescribeParameters",
        "ecr:Get*",
        "ecr:Describe*"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": "${REGION}" }
      }
    },
    {
      "Sid": "HistoryReadOnly",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeSpotPriceHistory",
        "cloudtrail:LookupEvents"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": ${HISTORY_REGIONS_JSON} }
      }
    },
    {
      "Sid": "CostExplorerReadOnly",
      "Effect": "Allow",
      "Action": "ce:GetCostAndUsage",
      "Resource": "*"
    },
    {
      "Sid": "Ec2Provision",
      "Effect": "Allow",
      "Action": [
        "ec2:CreateVpc",
        "ec2:CreateSubnet",
        "ec2:CreateInternetGateway",
        "ec2:CreateRouteTable",
        "ec2:CreateSecurityGroup",
        "ec2:CreateLaunchTemplate"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": "${REGION}" }
      }
    },
    {
      "Sid": "Ec2TagOnCreate",
      "Effect": "Allow",
      "Action": "ec2:CreateTags",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "${REGION}",
          "ec2:CreateAction": [
            "CreateVpc",
            "CreateSubnet",
            "CreateInternetGateway",
            "CreateRouteTable",
            "CreateSecurityGroup",
            "CreateLaunchTemplate",
            "CreateLaunchTemplateVersion",
            "AuthorizeSecurityGroupIngress",
            "AuthorizeSecurityGroupEgress"
          ]
        }
      }
    },
    {
      "Sid": "Ec2StackMutate",
      "Effect": "Allow",
      "Action": [
        "ec2:ModifyVpcAttribute",
        "ec2:ModifySubnetAttribute",
        "ec2:AttachInternetGateway",
        "ec2:CreateRoute",
        "ec2:ReplaceRoute",
        "ec2:DeleteRoute",
        "ec2:AssociateRouteTable",
        "ec2:DisassociateRouteTable",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:AuthorizeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:ModifySecurityGroupRules",
        "ec2:CreateLaunchTemplateVersion",
        "ec2:ModifyLaunchTemplate",
        "ec2:DeleteLaunchTemplateVersions",
        "ec2:CreateTags",
        "ec2:DeleteTags"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "${REGION}",
          "aws:ResourceTag/stack": "cubie-fleet"
        }
      }
    },
    {
      "Sid": "Ec2SgRuleLeg",
      "Effect": "Allow",
      "Action": [
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:AuthorizeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:ModifySecurityGroupRules"
      ],
      "Resource": "arn:aws:ec2:${REGION}:${ACCOUNT_ID}:security-group-rule/*"
    },
    {
      "Sid": "Ec2ScopedDestroy",
      "Effect": "Allow",
      "Action": [
        "ec2:TerminateInstances",
        "ec2:DeleteVpc",
        "ec2:DeleteSubnet",
        "ec2:DeleteInternetGateway",
        "ec2:DetachInternetGateway",
        "ec2:DeleteRouteTable",
        "ec2:DeleteSecurityGroup",
        "ec2:DeleteLaunchTemplate"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "${REGION}",
          "aws:ResourceTag/stack": "cubie-fleet"
        }
      }
    },
    {
      "Sid": "EcsUnscoped",
      "Effect": "Allow",
      "Action": [
        "ecs:CreateCluster",
        "ecs:RegisterTaskDefinition",
        "ecs:DeregisterTaskDefinition"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": "${REGION}" }
      }
    },
    {
      "Sid": "ServiceLinkedRoles",
      "Effect": "Allow",
      "Action": "iam:CreateServiceLinkedRole",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "iam:AWSServiceName": [
            "ecs.amazonaws.com",
            "spot.amazonaws.com",
            "ecs.application-autoscaling.amazonaws.com",
            "autoscaling.amazonaws.com"
          ]
        }
      }
    }
  ]
}
EOF

# Part 2: full management of the stack's named resources, the scoped
# IAM statements, and the self-mutation deny.
cat > /tmp/cubie-fleet-deployer-scoped-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3Scoped",
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::cubie-fleet-*",
        "arn:aws:s3:::cubie-fleet-*/*"
      ]
    },
    {
      "Sid": "SecretsScoped",
      "Effect": "Allow",
      "Action": "secretsmanager:*",
      "Resource": [
        "arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:/runs-on/cubie-fleet/*",
        "arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:cubie-fleet*"
      ]
    },
    {
      "Sid": "DynamoScoped",
      "Effect": "Allow",
      "Action": "dynamodb:*",
      "Resource": [
        "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/cubie-fleet-*",
        "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/cubie-fleet-*/index/*"
      ]
    },
    {
      "Sid": "LambdaScoped",
      "Effect": "Allow",
      "Action": "lambda:*",
      "Resource": "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:cubie-fleet-*"
    },
    {
      "Sid": "LogsScoped",
      "Effect": "Allow",
      "Action": "logs:*",
      "Resource": [
        "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:*cubie-fleet*",
        "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:*cubie-fleet*:*"
      ]
    },
    {
      "Sid": "SnsScoped",
      "Effect": "Allow",
      "Action": "sns:*",
      "Resource": "arn:aws:sns:${REGION}:${ACCOUNT_ID}:cubie-fleet-*"
    },
    {
      "Sid": "CloudWatchScoped",
      "Effect": "Allow",
      "Action": "cloudwatch:*",
      "Resource": [
        "arn:aws:cloudwatch:${REGION}:${ACCOUNT_ID}:alarm:cubie-fleet*",
        "arn:aws:cloudwatch::${ACCOUNT_ID}:dashboard/cubie-fleet*"
      ]
    },
    {
      "Sid": "EcsScoped",
      "Effect": "Allow",
      "Action": "ecs:*",
      "Resource": [
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:cluster/cubie-fleet*",
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:service/cubie-fleet*/*",
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task/cubie-fleet*/*",
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task-definition/cubie-fleet-*:*",
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:container-instance/cubie-fleet*/*"
      ]
    },
    {
      "Sid": "EventsScoped",
      "Effect": "Allow",
      "Action": [
        "events:*"
      ],
      "Resource": "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/cubie-fleet*"
    },
    {
      "Sid": "SchedulerScoped",
      "Effect": "Allow",
      "Action": "scheduler:*",
      "Resource": [
        "arn:aws:scheduler:${REGION}:${ACCOUNT_ID}:schedule/*/cubie-fleet*",
        "arn:aws:scheduler:${REGION}:${ACCOUNT_ID}:schedule-group/cubie-fleet*"
      ]
    },
    {
      "Sid": "SsmScoped",
      "Effect": "Allow",
      "Action": "ssm:*",
      "Resource": "arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter/cubie-fleet*"
    },
    {
      "Sid": "ResourceGroupsScoped",
      "Effect": "Allow",
      "Action": "resource-groups:*",
      "Resource": "arn:aws:resource-groups:${REGION}:${ACCOUNT_ID}:group/cubie-fleet*"
    },
    {
      "Sid": "IamScoped",
      "Effect": "Allow",
      "Action": [
        "iam:AddRoleToInstanceProfile",
        "iam:CreateInstanceProfile",
        "iam:CreatePolicy",
        "iam:CreatePolicyVersion",
        "iam:CreateRole",
        "iam:DeleteInstanceProfile",
        "iam:DeletePolicy",
        "iam:DeletePolicyVersion",
        "iam:DeleteRole",
        "iam:DeleteRolePolicy",
        "iam:GetInstanceProfile",
        "iam:GetPolicy",
        "iam:GetPolicyVersion",
        "iam:GetRole",
        "iam:GetRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:ListInstanceProfilesForRole",
        "iam:ListPolicyTags",
        "iam:ListPolicyVersions",
        "iam:ListRolePolicies",
        "iam:ListRoleTags",
        "iam:PutRolePolicy",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:TagInstanceProfile",
        "iam:TagPolicy",
        "iam:TagRole",
        "iam:UntagInstanceProfile",
        "iam:UntagPolicy",
        "iam:UntagRole",
        "iam:UpdateAssumeRolePolicy",
        "iam:UpdateRole"
      ],
      "Resource": [
        "arn:aws:iam::${ACCOUNT_ID}:role/cubie-fleet*",
        "arn:aws:iam::${ACCOUNT_ID}:role/runs-on*",
        "arn:aws:iam::${ACCOUNT_ID}:policy/cubie-fleet*",
        "arn:aws:iam::${ACCOUNT_ID}:policy/runs-on*",
        "arn:aws:iam::${ACCOUNT_ID}:instance-profile/cubie-fleet*",
        "arn:aws:iam::${ACCOUNT_ID}:instance-profile/runs-on*"
      ]
    },
    {
      "Sid": "IamAttachScoped",
      "Effect": "Allow",
      "Action": [
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy"
      ],
      "Resource": [
        "arn:aws:iam::${ACCOUNT_ID}:role/cubie-fleet*",
        "arn:aws:iam::${ACCOUNT_ID}:role/runs-on*"
      ],
      "Condition": {
        "ArnLike": {
          "iam:PolicyARN": [
            "arn:aws:iam::${ACCOUNT_ID}:policy/cubie-fleet*",
            "arn:aws:iam::${ACCOUNT_ID}:policy/runs-on*",
            "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
            "arn:aws:iam::aws:policy/AmazonElasticContainerRegistryPublicReadOnly",
            "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
          ]
        }
      }
    },
    {
      "Sid": "IamPassScoped",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": [
        "arn:aws:iam::${ACCOUNT_ID}:role/cubie-fleet*",
        "arn:aws:iam::${ACCOUNT_ID}:role/runs-on*"
      ],
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": [
            "ec2.amazonaws.com",
            "lambda.amazonaws.com",
            "ecs-tasks.amazonaws.com"
          ]
        }
      }
    },
    {
      "Sid": "DenySelfMutation",
      "Effect": "Deny",
      "Action": "iam:*",
      "Resource": [
        "arn:aws:iam::${ACCOUNT_ID}:policy/cubie-fleet-deployer*",
        "arn:aws:iam::${ACCOUNT_ID}:role/cubie-fleet-deployer"
      ]
    }
  ]
}
EOF

# Only identities inside this account can assume the role; combined
# with the 1-hour session cap this is the whole exposure surface.
cat > /tmp/cubie-fleet-deployer-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "AWS": "arn:aws:iam::${ACCOUNT_ID}:root" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

if ! aws iam get-role --role-name cubie-fleet-deployer >/dev/null 2>&1; then
  aws iam create-role --role-name cubie-fleet-deployer \
    --assume-role-policy-document file:///tmp/cubie-fleet-deployer-trust.json \
    --max-session-duration 3600
fi

# Publish one policy document idempotently and attach it to the role:
# create the policy on first run; on reruns push the document as a new
# default version, pruning the oldest non-default version first (IAM
# keeps at most five).
publish_policy() {
  local name="$1" doc="$2" arn oldest
  arn="arn:aws:iam::${ACCOUNT_ID}:policy/${name}"
  if aws iam get-policy --policy-arn "$arn" >/dev/null 2>&1; then
    oldest=$(aws iam list-policy-versions --policy-arn "$arn" \
      --query 'Versions[?!IsDefaultVersion]|[-1].VersionId' --output text)
    if [ "$oldest" != "None" ] && [ "$(aws iam list-policy-versions \
        --policy-arn "$arn" --query 'length(Versions)')" -ge 5 ]; then
      aws iam delete-policy-version --policy-arn "$arn" \
        --version-id "$oldest"
    fi
    aws iam create-policy-version --policy-arn "$arn" \
      --policy-document "file://${doc}" --set-as-default
  else
    aws iam create-policy --policy-name "$name" \
      --policy-document "file://${doc}"
  fi
  aws iam attach-role-policy --role-name cubie-fleet-deployer \
    --policy-arn "$arn"
}

publish_policy cubie-fleet-deployer \
  /tmp/cubie-fleet-deployer-policy.json
publish_policy cubie-fleet-deployer-scoped \
  /tmp/cubie-fleet-deployer-scoped-policy.json

echo
echo "cubie-fleet-deployer role and policies are up to date."
