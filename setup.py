"""Setup script for PeachBase C extensions."""

import os
import platform
import subprocess
import sys

from setuptools import Extension, setup

# Detect platform
is_windows = sys.platform.startswith("win")
is_linux = sys.platform.startswith("linux")
is_macos = sys.platform == "darwin"

# Platform-specific compiler flags
if is_windows:
    # MSVC compiler flags
    extra_compile_args = ["/O2"]
    extra_link_args = []
    openmp_compile_flag = "/openmp"
    openmp_link_flag = None
    avx2_flags = ["/arch:AVX2"]
else:
    # GCC/Clang compiler flags
    extra_compile_args = ["-O3"]
    extra_link_args = []
    openmp_compile_flag = "-fopenmp"
    openmp_link_flag = "-fopenmp"
    avx2_flags = ["-mavx2", "-mfma"]

# OpenMP for multi-threading (ENABLED by default)
# Set PEACHBASE_DISABLE_OPENMP=1 to build without OpenMP for Lambda
# Note: OpenMP adds ~1.2MB libgomp dependency on Linux
if os.environ.get("PEACHBASE_DISABLE_OPENMP", "0") != "1":
    openmp_available = False

    if is_windows:
        # MSVC has OpenMP support built-in
        openmp_available = True
    else:
        # Try to detect if OpenMP is available with gcc/clang
        try:
            result = subprocess.run(
                ["gcc", "-fopenmp", "-x", "c", "-", "-o", os.devnull],
                input=b"int main(){return 0;}",
                capture_output=True,
                timeout=5,
            )
            openmp_available = result.returncode == 0
        except Exception:
            pass

    if openmp_available:
        extra_compile_args.append(openmp_compile_flag)
        if openmp_link_flag:
            extra_link_args.append(openmp_link_flag)
        print("[PeachBase] OpenMP ENABLED - multi-core acceleration")
    else:
        print("[PeachBase] OpenMP not available - building single-threaded")
else:
    print("[PeachBase] OpenMP DISABLED - Lambda-friendly build")

# Architecture-specific optimizations
machine = platform.machine().lower()
if machine in ["x86_64", "amd64", "x86-64"]:
    extra_compile_args.extend(avx2_flags)
    if is_linux:
        extra_compile_args.append("-fPIC")
elif machine in ["aarch64", "arm64"]:
    # ARM 64-bit - NEON is enabled by default
    pass

# SIMD extension module
simd_extension = Extension(
    "peachbase._simd",
    sources=["csrc/peachbase_simd.c"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c",
)

# BM25 extension module
bm25_compile_args = ["/O2"] if is_windows else ["-O3"]
if is_linux:
    bm25_compile_args.append("-fPIC")

bm25_extension = Extension(
    "peachbase._bm25",
    sources=["csrc/peachbase_bm25.c"],
    extra_compile_args=bm25_compile_args,
    extra_link_args=extra_link_args,
    language="c",
)

# Setup configuration
setup(
    ext_modules=[simd_extension, bm25_extension],
)
