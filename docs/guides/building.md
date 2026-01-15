# PeachBase Build Options

## Quick Reference

### Default Build (Recommended for Most Users)
```bash
python -m build
pip install dist/peachbase-*.whl
```
✅ OpenMP enabled - multi-core acceleration
✅ 2-4x faster for collections >1K vectors
⚠️ Requires libgomp (~1.2MB, usually pre-installed)

### Lambda Build (AWS Lambda Deployments)
```bash
PEACHBASE_DISABLE_OPENMP=1 python -m build
pip install dist/peachbase-*.whl
```
✅ No external dependencies
✅ Minimal 43KB package
✅ Perfect for AWS Lambda
⚠️ Single-threaded only

---

## Dependency Comparison

### Default Build
```
peachbase._simd.so
├── SIMD: AVX2/AVX-512 ✅
├── OpenMP: Enabled ✅
└── Requires: libgomp.so.1 (~1.2MB)
```

**Check dependency:**
```bash
ldd $(python -c "import peachbase._simd; print(peachbase._simd.__file__)") | grep gomp
# Shows: libgomp.so.1 => /path/to/libgomp.so.1
```

### Lambda Build
```
peachbase._simd.so
├── SIMD: AVX2/AVX-512 ✅
├── OpenMP: Disabled
└── Requires: Nothing (only standard libc)
```

**Check dependency:**
```bash
ldd $(python -c "import peachbase._simd; print(peachbase._simd.__file__)") | grep gomp
# Shows nothing
```

---

## Performance Comparison

### Small Collections (< 1,000 vectors)
| Build | Performance | Notes |
|-------|-------------|-------|
| **Lambda** | ~87 QPS | ✅ Slightly faster (no thread overhead) |
| Default | ~85 QPS | Thread overhead minimal |

**Recommendation**: Either build works great

### Medium Collections (1,000-10,000 vectors)
| Build | Performance | Notes |
|-------|-------------|-------|
| Lambda | ~87 QPS | Single-threaded |
| **Default** | ~150-200 QPS | ✅ **1.7-2.3x faster** |

**Recommendation**: Use **default build** for best performance

### Large Collections (10,000+ vectors)
| Build | Performance | Notes |
|-------|-------------|-------|
| Lambda | ~8 QPS | Single-threaded |
| **Default** | ~25-35 QPS | ✅ **3-4x faster** |

**Recommendation**: Use **default build** for best performance

---

## When to Use Each Build

### ✅ Use Default Build (OpenMP) For:

1. **Local Development**
   - Maximum performance during testing
   - libgomp usually already installed

2. **Server Deployments**
   - EC2, DigitalOcean, dedicated servers
   - Docker containers with full OS

3. **Large Collections**
   - Any collection ≥ 1,000 vectors
   - Batch processing workloads

4. **Multi-Core Environments**
   - Servers with 2+ CPU cores
   - HPC clusters

### ✅ Use Lambda Build For:

1. **AWS Lambda**
   - Serverless deployments
   - Minimal cold start time
   - No dependency bundling needed

2. **Minimal Dependencies**
   - Restricted environments
   - Alpine Linux containers
   - Embedded systems

3. **Maximum Portability**
   - Systems without libgomp
   - Cross-compilation targets

---

## Installing libgomp (if needed)

Most Linux systems already have libgomp. If not:

### Ubuntu/Debian
```bash
sudo apt-get install libgomp1
```

### CentOS/RHEL/Fedora
```bash
sudo yum install libgomp
# or
sudo dnf install libgomp
```

### Alpine Linux (Docker)
```bash
apk add libgomp
```

### macOS
```bash
# Usually installed with Xcode Command Line Tools
# If needed:
brew install gcc
```

---

## CI/CD Examples

### GitHub Actions - Default Build
```yaml
- name: Build PeachBase
  run: |
    python -m build
    pip install dist/peachbase-*.whl
```

### GitHub Actions - Lambda Build
```yaml
- name: Build PeachBase for Lambda
  run: |
    PEACHBASE_DISABLE_OPENMP=1 python -m build
    pip install dist/peachbase-*.whl
  env:
    PEACHBASE_DISABLE_OPENMP: "1"
```

### Docker - Default Build
```dockerfile
FROM python:3.11
RUN apt-get update && apt-get install -y libgomp1
COPY . /app
WORKDIR /app
RUN python -m build && pip install dist/peachbase-*.whl
```

### Docker - Lambda Build
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN PEACHBASE_DISABLE_OPENMP=1 python -m build && \
    pip install dist/peachbase-*.whl
```

---

## Verifying Your Build

### Check if OpenMP is Enabled
```python
import peachbase
import subprocess

result = subprocess.run(
    ["ldd", peachbase._simd.__file__],
    capture_output=True,
    text=True
)

if "libgomp" in result.stdout:
    print("✅ OpenMP build - multi-core enabled")
else:
    print("📦 Lambda build - single-threaded")
```

### Test Multi-Core Usage
```bash
# Terminal 1: Run benchmark
python -c "
import peachbase
db = peachbase.connect('./test_db')
col = db.create_collection('test', dimension=384)
import random
docs = [{'id': f'd{i}', 'text': f'doc {i}',
         'vector': [random.random() for _ in range(384)]}
        for i in range(5000)]
col.add(docs)
for _ in range(100):
    col.search(query_vector=[random.random() for _ in range(384)], limit=10).to_list()
"

# Terminal 2: Monitor CPU usage
htop  # or top

# With OpenMP: Multiple cores at 100%
# Without OpenMP: One core at 100%
```

---

## Troubleshooting

### "libgomp.so.1: cannot open shared object file"

**Problem**: OpenMP build but libgomp not installed

**Solution 1**: Install libgomp
```bash
sudo apt-get install libgomp1  # Ubuntu/Debian
```

**Solution 2**: Rebuild without OpenMP
```bash
PEACHBASE_DISABLE_OPENMP=1 python -m build
pip install dist/peachbase-*.whl --force-reinstall
```

### Performance Not Improving with OpenMP

**Check 1**: Collection size
```python
# OpenMP only helps with >1,000 vectors
print(f"Collection size: {collection.size}")
```

**Check 2**: Thread count
```bash
# Make sure OpenMP can use multiple threads
export OMP_NUM_THREADS=$(nproc)
python your_script.py
```

**Check 3**: Verify OpenMP is actually linked
```bash
ldd $(python -c "import peachbase._simd; print(peachbase._simd.__file__)") | grep gomp
# Should show libgomp.so.1
```

---

## Summary

| Use Case | Build Command | Why |
|----------|--------------|-----|
| Local dev | `python -m build` | Max performance, libgomp available |
| Server prod | `python -m build` | Utilize all cores |
| AWS Lambda | `PEACHBASE_DISABLE_OPENMP=1 python -m build` | Minimal deps |
| Docker (full OS) | `python -m build` | libgomp available |
| Docker (Alpine) | `PEACHBASE_DISABLE_OPENMP=1 python -m build` | Minimal deps |
| Small collections | Either | Both work great |
| Large collections | `python -m build` | 2-4x faster |

**Default recommendation**: Use standard `python -m build` for best performance. Only use `PEACHBASE_DISABLE_OPENMP=1` for Lambda or minimal dependency requirements.

---

Made with 🍑 by the PeachBase team
