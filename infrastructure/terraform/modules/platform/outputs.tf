output "data_bucket" { value = aws_s3_bucket.data.id }
output "api_url" { value = aws_apigatewayv2_api.api.api_endpoint }
output "ecr_repository" { value = aws_ecr_repository.inference.repository_url }

