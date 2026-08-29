variable "project" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "lambda_zip" {
  type    = string
  default = "../../../../dist/lambda.zip"
}
