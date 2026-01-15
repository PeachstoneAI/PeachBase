# Deployment Guide

This guide covers deploying PeachBase to various environments: AWS Lambda, Docker containers, and traditional servers.

---

## Table of Contents

1. [AWS Lambda Deployment](#aws-lambda-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Traditional Server Deployment](#traditional-server-deployment)
4. [Environment Variables](#environment-variables)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)

---

## AWS Lambda Deployment

PeachBase is optimized for AWS Lambda with fast cold starts and minimal dependencies.

### Prerequisites

- AWS account with Lambda access
- AWS CLI configured
- Python 3.11 or 3.12
- S3 bucket for storing collections (optional)

### Step 1: Build Lambda Package

#### Option A: With OpenMP (Better Performance, Larger Package)

```bash
# Install dependencies
pip install build wheel

# Build the package
python -m build

# Install to lambda directory
pip install dist/peachbase-*.whl -t ./lambda_package
cd lambda_package
zip -r ../lambda_function.zip .
cd ..
```

#### Option B: Without OpenMP (Smaller Package, Lambda-Optimized)

```bash
# Build without OpenMP
PEACHBASE_DISABLE_OPENMP=1 python -m build

# Install to lambda directory
pip install dist/peachbase-*.whl -t ./lambda_package
cd lambda_package

# Optional: Strip debug symbols to reduce size
find . -name "*.so" -exec strip {} \;

zip -r ../lambda_function.zip .
cd ..
```

### Step 2: Add Your Lambda Function

Create `lambda_function.py`:

```python
import json
import peachbase

# Initialize outside handler for connection reuse
db = None

def lambda_handler(event, context):
    global db

    # Initialize database (cached between invocations)
    if db is None:
        # Use S3 or local /tmp storage
        db = peachbase.connect("s3://my-bucket/peachbase")
        # Or use local: db = peachbase.connect("/tmp/peachbase")

    # Open or create collection
    try:
        collection = db.open_collection("documents")
    except (KeyError, FileNotFoundError):
        collection = db.create_collection("documents", dimension=384)

    # Handle different operations
    operation = event.get("operation", "search")

    if operation == "add":
        # Add documents
        documents = event.get("documents", [])
        collection.add(documents)
        collection.save()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"Added {len(documents)} documents",
                "collection_size": collection.size
            })
        }

    elif operation == "search":
        # Search documents
        query_vector = event.get("query_vector")
        query_text = event.get("query_text")
        mode = event.get("mode", "semantic")
        limit = event.get("limit", 10)
        filter_query = event.get("filter")

        # Build search parameters
        search_params = {"mode": mode, "limit": limit}
        if query_vector:
            search_params["query_vector"] = query_vector
        if query_text:
            search_params["query_text"] = query_text
        if filter_query:
            search_params["filter"] = filter_query

        results = collection.search(**search_params)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "results": [
                    {
                        "id": r["id"],
                        "text": r.get("text", ""),
                        "score": r["score"],
                        "metadata": r.get("metadata", {})
                    }
                    for r in results.to_list()
                ]
            })
        }

    elif operation == "get":
        # Get document by ID
        doc_id = event.get("id")
        document = collection.get(doc_id)

        if document:
            return {
                "statusCode": 200,
                "body": json.dumps({"document": document})
            }
        else:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "Document not found"})
            }

    elif operation == "delete":
        # Delete document by ID
        doc_id = event.get("id")
        collection.delete(doc_id)
        collection.save()

        return {
            "statusCode": 200,
            "body": json.dumps({"message": f"Deleted document {doc_id}"})
        }

    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Unknown operation: {operation}"})
        }
```

Add your function to the package:

```bash
zip -g lambda_function.zip lambda_function.py
```

### Step 3: Deploy to Lambda

```bash
# Create Lambda function
aws lambda create-function \
    --function-name peachbase-search \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda_function.zip \
    --timeout 30 \
    --memory-size 1024 \
    --architecture x86_64

# Or update existing function
aws lambda update-function-code \
    --function-name peachbase-search \
    --zip-file fileb://lambda_function.zip
```

### Step 4: Lambda Configuration

**Recommended Settings:**

| Setting | Value | Notes |
|---------|-------|-------|
| Memory | 512-1024 MB | Adjust based on collection size |
| Timeout | 10-30 seconds | First cold start may be slower |
| Runtime | Python 3.11/3.12 | Use latest stable version |
| Architecture | x86_64 | Required for AVX2 SIMD support |
| Ephemeral Storage | 512 MB - 10 GB | For caching in /tmp |

**Environment Variables:**

```bash
PEACHBASE_S3_BUCKET=my-bucket           # S3 bucket for collections
PEACHBASE_CACHE_DIR=/tmp/peachbase        # Cache directory
PEACHBASE_LOG_LEVEL=INFO                # Logging level
```

### Step 5: S3 Permissions

Add S3 permissions to your Lambda execution role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket/*",
        "arn:aws:s3:::my-bucket"
      ]
    }
  ]
}
```

### Step 6: Test Your Function

```bash
# Test search operation
aws lambda invoke \
    --function-name peachbase-search \
    --payload '{"operation":"search","query_text":"machine learning","limit":5}' \
    response.json

cat response.json
```

### Lambda Optimization Tips

1. **Use /tmp for caching**: Lambda preserves /tmp between invocations
   ```python
   db = peachbase.connect("/tmp/peachbase")
   ```

2. **Initialize outside handler**: Reuse database connections
   ```python
   db = peachbase.connect("s3://bucket/peachbase")  # Outside handler

   def lambda_handler(event, context):
       collection = db.open_collection("docs")  # Fast access
   ```

3. **Minimize package size**:
   - Build without OpenMP if not needed
   - Strip debug symbols: `strip *.so`
   - Remove unnecessary dependencies

4. **Use provisioned concurrency**: For sub-second cold starts
   ```bash
   aws lambda put-provisioned-concurrency-config \
       --function-name peachbase-search \
       --provisioned-concurrent-executions 2
   ```

5. **Monitor performance**:
   ```bash
   aws cloudwatch get-metric-statistics \
       --namespace AWS/Lambda \
       --metric-name Duration \
       --dimensions Name=FunctionName,Value=peachbase-search \
       --start-time 2024-01-01T00:00:00Z \
       --end-time 2024-01-02T00:00:00Z \
       --period 3600 \
       --statistics Average,Maximum
   ```

---

## Docker Deployment

Deploy PeachBase in a Docker container for consistent, portable deployments.

### Dockerfile

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy PeachBase wheel (if building locally)
COPY dist/peachbase-*.whl .
RUN pip install --no-cache-dir peachbase-*.whl

# Copy application code
COPY app.py .
COPY collections/ ./collections/

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

### Example Application (Flask)

Create `app.py`:

```python
from flask import Flask, request, jsonify
import peachbase

app = Flask(__name__)

# Initialize database
db = peachbase.connect("./collections")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    collection_name = data.get("collection", "documents")

    try:
        collection = db.open_collection(collection_name)
    except (KeyError, FileNotFoundError):
        return jsonify({"error": "Collection not found"}), 404

    results = collection.search(
        query_vector=data.get("query_vector"),
        query_text=data.get("query_text"),
        mode=data.get("mode", "semantic"),
        limit=data.get("limit", 10),
        filter=data.get("filter")
    )

    return jsonify({
        "results": [
            {"id": r["id"], "text": r.get("text"), "score": r["score"]}
            for r in results.to_list()
        ]
    })

@app.route("/add", methods=["POST"])
def add_documents():
    data = request.json
    collection_name = data.get("collection", "documents")
    documents = data.get("documents", [])

    collection = db.open_collection(collection_name)
    collection.add(documents)
    collection.save()

    return jsonify({"message": f"Added {len(documents)} documents"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

### Build and Run

```bash
# Build image
docker build -t peachbase-api .

# Run container
docker run -d \
    -p 8000:8000 \
    -v $(pwd)/data:/app/collections \
    --name peachbase-api \
    peachbase-api

# Test
curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"query_text":"machine learning","limit":5}'
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  peachbase-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/collections
    environment:
      - PEACHBASE_LOG_LEVEL=INFO
    restart: unless-stopped

  # Optional: Add nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - peachbase-api
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

---

## Traditional Server Deployment

Deploy PeachBase on a traditional server (VPS, EC2, bare metal).

### Prerequisites

- Python 3.11+ installed
- systemd (for service management)
- nginx (optional, for reverse proxy)

### Step 1: Install PeachBase

```bash
# Clone repository
git clone https://github.com/PeachstoneAI/peachbase.git
cd peachbase

# Build with OpenMP for best performance
python -m build

# Install
pip install dist/peachbase-*.whl

# Or install from PyPI
pip install peachbase
```

### Step 2: Create Application

Create `/opt/peachbase/app.py`:

```python
# Your application code (similar to Docker example)
```

### Step 3: Create systemd Service

Create `/etc/systemd/system/peachbase.service`:

```ini
[Unit]
Description=PeachBase API Service
After=network.target

[Service]
Type=simple
User=peachbase
Group=peachbase
WorkingDirectory=/opt/peachbase
Environment="PATH=/opt/peachbase/venv/bin"
ExecStart=/opt/peachbase/venv/bin/python app.py
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/peachbase/data

[Install]
WantedBy=multi-user.target
```

### Step 4: Start Service

```bash
# Create user
sudo useradd -r -s /bin/false peachbase

# Set permissions
sudo chown -R peachbase:peachbase /opt/peachbase

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable peachbase
sudo systemctl start peachbase

# Check status
sudo systemctl status peachbase
```

### Step 5: Configure nginx (Optional)

Create `/etc/nginx/sites-available/peachbase`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for large collections
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/peachbase /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Environment Variables

PeachBase supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PEACHBASE_CACHE_DIR` | Directory for caching collections | `/tmp/peachbase` |
| `PEACHBASE_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `PEACHBASE_DISABLE_OPENMP` | Disable OpenMP at runtime | `false` |
| `PEACHBASE_MAX_THREADS` | Maximum OpenMP threads | System CPU count |
| `PEACHBASE_S3_ENDPOINT` | Custom S3 endpoint (for MinIO, etc.) | AWS default |

Set in your deployment:

```bash
# Lambda
export PEACHBASE_CACHE_DIR=/tmp/peachbase

# Docker
docker run -e PEACHBASE_LOG_LEVEL=DEBUG ...

# systemd
Environment="PEACHBASE_LOG_LEVEL=INFO"
```

---

## Performance Tuning

### Lambda Performance

1. **Memory allocation**: Higher memory = more CPU
   - 512 MB: Basic workloads (< 1K docs)
   - 1024 MB: Medium workloads (1K-10K docs)
   - 2048 MB: Large workloads (10K-100K docs)

2. **Timeout settings**:
   - Cold start: 10-15 seconds
   - Warm requests: 2-5 seconds
   - Set timeout to 30 seconds minimum

3. **Caching strategy**:
   ```python
   # Load to /tmp on first invocation
   if not os.path.exists("/tmp/peachbase"):
       db = peachbase.connect("s3://bucket/peachbase")
       # Cache is preserved between invocations
   ```

### Server Performance

1. **OpenMP threads**:
   ```bash
   export OMP_NUM_THREADS=4  # Set to CPU core count
   ```

2. **Process workers** (for Flask/FastAPI):
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
   ```

3. **Memory limits**:
   ```ini
   # In systemd service
   MemoryMax=2G
   MemoryHigh=1.5G
   ```

---

## Troubleshooting

### Common Issues

**"Module not found: _simd"**
- C extensions not compiled
- Solution: Rebuild with `python -m build`

**Lambda cold starts > 10 seconds**
- Package too large (> 50 MB)
- Solution: Build without OpenMP, strip symbols

**"Permission denied" on S3**
- Missing IAM permissions
- Solution: Add S3 read/write permissions to role

**Slow searches (> 1 second)**
- Collection too large for brute force
- Solution: Consider splitting collections or using filters

**OpenMP errors on Lambda**
- libgomp.so.1 not found
- Solution: Build with `PEACHBASE_DISABLE_OPENMP=1`

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check Lambda logs:

```bash
aws logs tail /aws/lambda/peachbase-search --follow
```

Monitor system resources:

```bash
# Docker
docker stats peachbase-api

# Server
htop
journalctl -u peachbase -f
```

---

## Next Steps

- See [Performance Optimizations](performance.md) for detailed tuning
- Check [Building Guide](building.md) for compilation options
- Read [API Reference](../reference/api.md) for complete API docs

---

**Need Help?**
- Open an issue: https://github.com/PeachstoneAI/peachbase/issues
- Check examples: `examples/` directory
