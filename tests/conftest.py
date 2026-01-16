"""Pytest configuration and shared fixtures."""

import random
import shutil
import tempfile

import pytest


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    temp_dir = tempfile.mkdtemp(prefix="peachbase_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents():
    """Generate sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "Machine learning is a subset of artificial intelligence",
            "vector": [0.1 + i * 0.01 for i in range(384)],
            "metadata": {"category": "ai", "year": 2024},
        },
        {
            "id": "doc2",
            "text": "Deep learning uses neural networks",
            "vector": [0.2 + i * 0.01 for i in range(384)],
            "metadata": {"category": "ai", "year": 2023},
        },
        {
            "id": "doc3",
            "text": "Natural language processing enables text understanding",
            "vector": [0.3 + i * 0.01 for i in range(384)],
            "metadata": {"category": "nlp", "year": 2024},
        },
        {
            "id": "doc4",
            "text": "Computer vision analyzes images",
            "vector": [0.4 + i * 0.01 for i in range(384)],
            "metadata": {"category": "cv", "year": 2023},
        },
        {
            "id": "doc5",
            "text": "Reinforcement learning trains agents",
            "vector": [0.5 + i * 0.01 for i in range(384)],
            "metadata": {"category": "rl", "year": 2024},
        },
    ]


@pytest.fixture
def random_vectors():
    """Generate random vectors for performance testing."""

    def _generate(n_vectors=100, dimension=384):
        return [[random.random() for _ in range(dimension)] for _ in range(n_vectors)]

    return _generate


@pytest.fixture
def query_vector():
    """Generate a query vector."""
    return [0.15 + i * 0.01 for i in range(384)]
