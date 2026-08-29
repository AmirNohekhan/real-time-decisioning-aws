terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.0" }
    archive = { source = "hashicorp/archive", version = "~> 2.4" }
  }
}

locals { name = "${var.project}-${var.environment}" }

resource "aws_s3_bucket" "data" {
  bucket_prefix = "${local.name}-data-"
  force_destroy = false
}
resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}
resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "raw-tiering"
    status = "Enabled"
    filter {}
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_kinesis_stream" "events" {
  name = "${local.name}-events"
  stream_mode_details { stream_mode = "ON_DEMAND" }
  encryption_type  = "KMS"
  kms_key_id       = "alias/aws/kinesis"
  retention_period = 48
}
resource "aws_dynamodb_table" "state" {
  name         = "${local.name}-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  attribute {
    name = "pk"
    type = "S"
  }
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
  server_side_encryption { enabled = true }
  point_in_time_recovery { enabled = true }
}
resource "aws_ecr_repository" "inference" {
  name                 = "${local.name}-inference"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "lambda" {
  name               = "${local.name}-api"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}
data "aws_iam_policy_document" "lambda" {
  statement {
    actions   = ["kinesis:PutRecord"]
    resources = [aws_kinesis_stream.events.arn]
  }
  statement {
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem"]
    resources = [aws_dynamodb_table.state.arn]
  }
  statement {
    actions   = ["sagemaker:InvokeEndpoint"]
    resources = ["arn:aws:sagemaker:${var.aws_region}:*:endpoint/${local.name}-*"]
  }
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.aws_region}:*:*"]
  }
}
resource "aws_iam_role_policy" "lambda" {
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}
data "archive_file" "stub" {
  type        = "zip"
  output_path = "${path.module}/lambda_stub.zip"
  source {
    content  = "def handler(event, context):\n return {'statusCode': 503, 'body': 'Deploy application artifact'}"
    filename = "handler.py"
  }
}
resource "aws_lambda_function" "api" {
  function_name    = "${local.name}-api"
  role             = aws_iam_role.lambda.arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.stub.output_path
  source_code_hash = data.archive_file.stub.output_base64sha256
  timeout          = 10
  memory_size      = 512
  environment {
    variables = {
      DP_KINESIS_STREAM = aws_kinesis_stream.events.name
      DP_DYNAMODB_TABLE = aws_dynamodb_table.state.name
      DP_ENV            = var.environment
    }
  }
  tracing_config { mode = "Active" }
}
resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${aws_lambda_function.api.function_name}"
  retention_in_days = 30
}
resource "aws_apigatewayv2_api" "api" {
  name          = local.name
  protocol_type = "HTTP"
}
resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}
resource "aws_apigatewayv2_route" "recommend" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /v1/recommendations"
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true
  default_route_settings {
    throttling_burst_limit = 100
    throttling_rate_limit  = 50
  }
}
resource "aws_lambda_permission" "api" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

data "aws_iam_policy_document" "sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "sfn" {
  name               = "${local.name}-training-orchestrator"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume.json
}
resource "aws_iam_role_policy" "sfn" {
  role   = aws_iam_role.sfn.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = ["sagemaker:StartPipelineExecution", "sagemaker:DescribePipelineExecution"], Resource = "arn:aws:sagemaker:${var.aws_region}:*:pipeline/${local.name}-*" }] })
}
resource "aws_sfn_state_machine" "training" {
  name     = "${local.name}-training"
  role_arn = aws_iam_role.sfn.arn
  definition = jsonencode({
    StartAt = "StartPipeline"
    States = {
      StartPipeline = {
        Type       = "Task"
        Resource   = "arn:aws:states:::aws-sdk:sagemaker:startPipelineExecution"
        Parameters = { PipelineName = "${local.name}-ml" }
        End        = true
      }
    }
  })
}
