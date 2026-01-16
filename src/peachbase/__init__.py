"""
PeachBase - Lightweight vector database optimized for AWS Lambda.

PeachBase provides lexical (BM25), semantic (SIMD-accelerated), and hybrid search
capabilities with minimal dependencies and fast cold starts for serverless environments.
"""

from peachbase._version import __version__
from peachbase.collection import Collection
from peachbase.database import Database, connect

# Query will be implemented in Phase 5
# from peachbase.query import Query

__all__ = [
    "__version__",
    "Database",
    "connect",
    "Collection",
    # "Query",
]
