terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.0" }
    archive = { source = "hashicorp/archive", version = "~> 2.4" }
  }
}
provider "aws" { region = var.aws_region }
module "platform" {
  source      = "../../modules/platform"
  project     = var.project
  environment = "production"
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
