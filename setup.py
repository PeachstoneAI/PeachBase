"""Setup script for PeachBase C extensions."""

from setuptools import setup, Extension
import platform
import sys

# Detect platform and set SIMD flags
extra_compile_args = ["-O3"]
extra_link_args = []

# OpenMP for multi-threading (ENABLED by default)
# Set PEACHBASE_DISABLE_OPENMP=1 to build without OpenMP for Lambda
# Note: OpenMP adds ~1.2MB libgomp dependency
import os
if os.environ.get("PEACHBASE_DISABLE_OPENMP", "0") != "1":
    # OpenMP enabled by default
    try:
        # Try to detect if OpenMP is available
        import subprocess
        result = subprocess.run(
            ["gcc", "-fopenmp", "-x", "c", "-", "-o", "/dev/null"],
            input=b"int main(){return 0;}",
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            extra_compile_args.append("-fopenmp")
            extra_link_args.append("-fopenmp")
            print("🔥 OpenMP ENABLED (default) - multi-core acceleration")
            print("   • 2-4x faster for collections >1K vectors")
            print("   • Dependency: libgomp (~1.2MB)")
            print("   • For Lambda: PEACHBASE_DISABLE_OPENMP=1 python -m build")
        else:
            print("⚠️  OpenMP not available - building single-threaded")
    except Exception:
        print("⚠️  Could not detect OpenMP - building single-threaded")
else:
    print("📦 OpenMP DISABLED - Lambda-friendly build")
    print("   • Minimal dependencies (43KB)")
    print("   • Single-threaded mode")
    print("   • Perfect for AWS Lambda")

# Platform-specific optimizations
if platform.machine() in ["x86_64", "AMD64", "x86-64"]:
    if sys.platform == "darwin":  # macOS
        # macOS with Apple Silicon or Intel
        if platform.processor() == "arm":
            # Apple Silicon - use NEON (ARM SIMD)
            extra_compile_args.extend(["-O3"])
        else:
            # Intel Mac - use AVX2
            extra_compile_args.extend(["-mavx2", "-mfma"])
    elif sys.platform.startswith("linux") or sys.platform.startswith("win"):
        # Linux or Windows x86_64 - use AVX2
        extra_compile_args.extend(["-mavx2", "-mfma"])
        # Optionally enable AVX-512 if needed:
        # extra_compile_args.extend(["-mavx512f"])

    # Add position-independent code for shared libraries
    if sys.platform.startswith("linux"):
        extra_compile_args.append("-fPIC")
elif platform.machine() in ["aarch64", "arm64"]:
    # ARM 64-bit - enable NEON
    extra_compile_args.extend(["-O3"])

# SIMD extension module
simd_extension = Extension(
    "peachbase._simd",
    sources=["csrc/peachbase_simd.c"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c",
)

# BM25 extension module
bm25_extension = Extension(
    "peachbase._bm25",
    sources=["csrc/peachbase_bm25.c"],
    extra_compile_args=["-O3"] + (["-fPIC"] if sys.platform.startswith("linux") else []),
    extra_link_args=extra_link_args,
    language="c",
)

# Setup configuration
setup(
    ext_modules=[simd_extension, bm25_extension],
)
