# Installation Guide

Get PeachBase up and running in under a minute.

---

## Quick Install (Recommended)

From pre-built wheel (fastest):

```bash
pip install dist/peachbase-*.whl
```

That's it! ✅

---

## Verify Installation

Run the quick test:

```bash
cd examples
python quick_test.py
```

Expected output:
```
================================================================================
🍑 PeachBase Quick Test
================================================================================

✓ PeachBase imported successfully!
  Version: 0.1.0

📦 Testing C Extensions:
  ✓ SIMD module loaded (CPU: AVX2)
  ✓ SIMD cosine similarity works: 0.9938
  ✓ BM25 module loaded

...

================================================================================
✅ All Tests Passed!
================================================================================
```

---

## Build from Source

### Standard Build (OpenMP Enabled)

**Recommended for**: Development, servers, production

```bash
# Build
python -m build

# Install
pip install dist/peachbase-*.whl
```

**Features**:
- ✅ Multi-core acceleration (OpenMP)
- ✅ 3-4x faster for collections >1K vectors
- ✅ Uses all CPU cores
- ⚠️ Requires libgomp (~1.2MB, usually pre-installed)

### Lambda Build (No OpenMP)

**Recommended for**: AWS Lambda, minimal dependencies

```bash
# Build without OpenMP
PEACHBASE_DISABLE_OPENMP=1 python -m build

# Install
pip install dist/peachbase-*.whl
```

**Features**:
- ✅ No external dependencies
- ✅ Minimal 43KB package
- ✅ Perfect for AWS Lambda
- ⚠️ Single-threaded only

See [Building Guide](../guides/building.md) for detailed options.

---

## Requirements

### Runtime Requirements

**Python**: 3.11 or higher

**Dependencies**:
- `boto3 >= 1.34.0` (for S3 support)

**Optional**:
- `libgomp.so.1` (for multi-core, usually pre-installed on Linux)

### Build Requirements

- `setuptools >= 68`
- `wheel`
- C compiler (gcc or clang)
- AVX2-capable CPU (for SIMD, falls back to standard code otherwise)

---

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install build dependencies (if building from source)
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Install libgomp (for OpenMP, usually already installed)
sudo apt-get install libgomp1

# Build and install
python -m build
pip install dist/peachbase-*.whl
```

### macOS

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# libgomp is included with Xcode tools

# Build and install
python -m build
pip install dist/peachbase-*.whl
```

### Windows

```bash
# Install Microsoft Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Build and install
python -m build
pip install dist/peachbase-*.whl
```

**Note**: OpenMP may not be available on Windows with some compilers.

### Docker

**Standard Image** (with OpenMP):
```dockerfile
FROM python:3.11

# Install libgomp
RUN apt-get update && apt-get install -y libgomp1

# Copy and install PeachBase
COPY dist/peachbase-*.whl /tmp/
RUN pip install /tmp/peachbase-*.whl

# Your app
COPY . /app
WORKDIR /app
```

**Minimal Image** (without OpenMP):
```dockerfile
FROM python:3.11-slim

# Copy Lambda build
COPY dist/peachbase-*.whl /tmp/
RUN pip install /tmp/peachbase-*.whl

# Your app
COPY . /app
WORKDIR /app
```

---

## Troubleshooting

### "No module named 'peachbase'"

**Problem**: PeachBase not installed

**Solution**:
```bash
pip install dist/peachbase-*.whl
```

### "Cannot import _simd"

**Problem**: C extensions not compiled

**Solution**: Rebuild from source:
```bash
# Clean previous build
rm -rf build dist/*.whl

# Rebuild
python -m build

# Reinstall
pip install dist/peachbase-*.whl --force-reinstall
```

### "libgomp.so.1: cannot open shared object file"

**Problem**: OpenMP library not installed (only affects OpenMP builds)

**Solution 1**: Install libgomp
```bash
# Ubuntu/Debian
sudo apt-get install libgomp1

# CentOS/RHEL
sudo yum install libgomp

# Alpine
apk add libgomp
```

**Solution 2**: Use Lambda build (no OpenMP)
```bash
PEACHBASE_DISABLE_OPENMP=1 python -m build
pip install dist/peachbase-*.whl --force-reinstall
```

### Build fails with "error: command 'gcc' failed"

**Problem**: Missing C compiler or development headers

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# macOS
xcode-select --install
```

---

## Next Steps

Now that PeachBase is installed:

1. **[Quick Start](quick-start.md)** - Your first search in 5 minutes
2. **[Basic Concepts](basic-concepts.md)** - Understand core concepts
3. **[Examples](../examples/)** - See working code examples

---

## Additional Dependencies

For running examples, you may want:

```bash
# For Wikipedia RAG examples
pip install sentence-transformers wikipedia-api datasets tqdm

# For development
pip install pytest pytest-benchmark black ruff mypy
```

See individual example READMEs for specific requirements.

---

[← Back to Documentation Index](../README.md) | [Next: Quick Start →](quick-start.md)
