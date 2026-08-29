terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.0" }
    archive = { source = "hashicorp/archive", version = "~> 2.4" }
  }
}
provider "aws" {
  region = var.aws_region
  default_tags { tags = { Project = var.project, Environment = "dev", ManagedBy = "Terraform" } }
}
module "platform" {
  source      = "../../modules/platform"
  project     = var.project
  environment = "dev"
  aws_region  = var.aws_region
}
variable "project" {
  type    = string
  default = "decision-platform"
}
variable "aws_region" {
  type    = string
  default = "us-east-1"
}
output "api_url" { value = module.platform.api_url }
