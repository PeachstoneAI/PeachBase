# Integration Tests

This directory contains integration tests for PeachBase that test integration with external services or systems.

## Test Categories

### S3 Integration (`test_s3.py`)
Tests for AWS S3 integration including:
- Lazy loading of boto3
- Listing collections from S3 buckets
- Downloading and caching from S3
- Deleting S3 objects
- Database operations with S3 URIs

**Note**: These tests use `moto` to mock AWS S3 services, so they don't require actual AWS credentials or internet connectivity. However, they test the integration layer with S3, which is why they're in the integration directory.

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/

# Run specific integration test file
pytest tests/integration/test_s3.py

# Run with verbose output
pytest tests/integration/ -v
```

## Dependencies

Integration tests may require additional dependencies:
- `boto3` - AWS SDK for Python
- `moto[s3]` - AWS service mocking library

Install with:
```bash
pip install -r requirements-test.txt
```
