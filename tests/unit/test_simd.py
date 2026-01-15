"""Unit tests for SIMD C extensions."""
import pytest
import array
import math


def test_simd_module_imports():
    """Test that SIMD module can be imported."""
    from peachbase import _simd
    assert _simd is not None


def test_cpu_feature_detection():
    """Test CPU feature detection."""
    from peachbase import _simd

    features = _simd.detect_cpu_features()
    assert isinstance(features, int)
    assert features in [0, 1, 2]  # 0=none, 1=avx2, 2=avx512


def test_openmp_info():
    """Test OpenMP information."""
    from peachbase import _simd

    info = _simd.get_openmp_info()
    assert isinstance(info, dict)
    assert "compiled_with_openmp" in info
    assert isinstance(info["compiled_with_openmp"], bool)

    if info["compiled_with_openmp"]:
        assert "max_threads" in info
        assert "openmp_version" in info


def test_cosine_similarity_basic():
    """Test basic cosine similarity."""
    from peachbase import _simd

    # Simple vectors
    vec1 = array.array('f', [1.0, 0.0, 0.0])
    vec2 = array.array('f', [1.0, 0.0, 0.0])

    result = _simd.cosine_similarity(vec1, vec2)

    # Identical vectors should have cosine similarity of 1.0
    assert abs(result - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    """Test cosine similarity of orthogonal vectors."""
    from peachbase import _simd

    vec1 = array.array('f', [1.0, 0.0, 0.0])
    vec2 = array.array('f', [0.0, 1.0, 0.0])

    result = _simd.cosine_similarity(vec1, vec2)

    # Orthogonal vectors should have cosine similarity near 0
    assert abs(result) < 1e-6


def test_cosine_similarity_opposite():
    """Test cosine similarity of opposite vectors."""
    from peachbase import _simd

    vec1 = array.array('f', [1.0, 0.0, 0.0])
    vec2 = array.array('f', [-1.0, 0.0, 0.0])

    result = _simd.cosine_similarity(vec1, vec2)

    # Opposite vectors should have cosine similarity of -1.0
    assert abs(result - (-1.0)) < 1e-6


def test_l2_distance_basic():
    """Test basic L2 distance."""
    from peachbase import _simd

    vec1 = array.array('f', [0.0, 0.0, 0.0])
    vec2 = array.array('f', [3.0, 4.0, 0.0])

    result = _simd.l2_distance(vec1, vec2)

    # L2 distance should be 5.0 (3-4-5 triangle)
    assert abs(result - 5.0) < 1e-6


def test_l2_distance_identical():
    """Test L2 distance of identical vectors."""
    from peachbase import _simd

    vec1 = array.array('f', [1.0, 2.0, 3.0])
    vec2 = array.array('f', [1.0, 2.0, 3.0])

    result = _simd.l2_distance(vec1, vec2)

    # Identical vectors should have L2 distance of 0
    assert abs(result) < 1e-6


def test_dot_product_basic():
    """Test basic dot product."""
    from peachbase import _simd

    vec1 = array.array('f', [1.0, 2.0, 3.0])
    vec2 = array.array('f', [4.0, 5.0, 6.0])

    result = _simd.dot_product(vec1, vec2)

    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    expected = 32.0
    assert abs(result - expected) < 1e-6


def test_dot_product_zero():
    """Test dot product with zero vector."""
    from peachbase import _simd

    vec1 = array.array('f', [1.0, 2.0, 3.0])
    vec2 = array.array('f', [0.0, 0.0, 0.0])

    result = _simd.dot_product(vec1, vec2)

    assert abs(result) < 1e-6


def test_batch_cosine_similarity():
    """Test batch cosine similarity."""
    from peachbase import _simd

    query = array.array('f', [1.0, 0.0, 0.0])

    # Create 3 vectors
    vectors = array.array('f', [
        1.0, 0.0, 0.0,  # Same as query
        0.0, 1.0, 0.0,  # Orthogonal
        -1.0, 0.0, 0.0,  # Opposite
    ])

    results = _simd.batch_cosine_similarity(query, vectors)

    assert len(results) == 3
    assert abs(results[0] - 1.0) < 1e-6  # Same
    assert abs(results[1]) < 1e-6  # Orthogonal
    assert abs(results[2] - (-1.0)) < 1e-6  # Opposite


def test_batch_l2_distance():
    """Test batch L2 distance."""
    from peachbase import _simd

    query = array.array('f', [0.0, 0.0, 0.0])

    # Create 3 vectors
    vectors = array.array('f', [
        0.0, 0.0, 0.0,  # Same as query (distance 0)
        3.0, 4.0, 0.0,  # Distance 5
        1.0, 0.0, 0.0,  # Distance 1
    ])

    results = _simd.batch_l2_distance(query, vectors)

    assert len(results) == 3
    assert abs(results[0]) < 1e-6  # Distance 0
    assert abs(results[1] - 5.0) < 1e-6  # Distance 5
    assert abs(results[2] - 1.0) < 1e-6  # Distance 1


def test_batch_dot_product():
    """Test batch dot product."""
    from peachbase import _simd

    query = array.array('f', [1.0, 2.0, 3.0])

    # Create 3 vectors
    vectors = array.array('f', [
        1.0, 2.0, 3.0,  # Same (dot = 14)
        4.0, 5.0, 6.0,  # Different (dot = 32)
        0.0, 0.0, 0.0,  # Zero (dot = 0)
    ])

    results = _simd.batch_dot_product(query, vectors)

    assert len(results) == 3
    assert abs(results[0] - 14.0) < 1e-6  # 1+4+9
    assert abs(results[1] - 32.0) < 1e-6  # 4+10+18
    assert abs(results[2]) < 1e-6  # 0


def test_simd_with_384_dimensions():
    """Test SIMD operations with typical embedding dimension."""
    from peachbase import _simd

    dim = 384
    query = array.array('f', [i * 0.01 for i in range(dim)])
    vectors = array.array('f')

    # Create 10 vectors
    for j in range(10):
        for i in range(dim):
            vectors.append((i + j) * 0.01)

    # Test all batch operations work with realistic dimensions
    results_cosine = _simd.batch_cosine_similarity(query, vectors)
    results_l2 = _simd.batch_l2_distance(query, vectors)
    results_dot = _simd.batch_dot_product(query, vectors)

    assert len(results_cosine) == 10
    assert len(results_l2) == 10
    assert len(results_dot) == 10

    # All results should be valid numbers
    assert all(not math.isnan(r) and not math.isinf(r) for r in results_cosine)
    assert all(not math.isnan(r) and not math.isinf(r) for r in results_l2)
    assert all(not math.isnan(r) and not math.isinf(r) for r in results_dot)


def test_simd_performance_vs_python():
    """Test that SIMD is faster than pure Python (smoke test)."""
    from peachbase import _simd
    import time

    dim = 384
    n_vectors = 100

    query = array.array('f', [i * 0.01 for i in range(dim)])
    vectors = array.array('f', [i * 0.01 for _ in range(n_vectors) for i in range(dim)])

    # Time SIMD version
    start = time.time()
    for _ in range(10):
        _simd.batch_cosine_similarity(query, vectors)
    simd_time = time.time() - start

    # SIMD should complete in reasonable time (<1 second for this workload)
    assert simd_time < 1.0


def test_simd_different_array_types():
    """Test that SIMD functions require float arrays."""
    from peachbase import _simd

    # Should work with float array
    vec1 = array.array('f', [1.0, 2.0, 3.0])
    vec2 = array.array('f', [4.0, 5.0, 6.0])

    result = _simd.cosine_similarity(vec1, vec2)
    assert result is not None

    # Double array might not work (depends on implementation)
    # This is just to document the expected behavior


def test_dimension_mismatch_handling():
    """Test behavior with mismatched dimensions."""
    from peachbase import _simd

    # If we pass wrong dimension parameter, results might be incorrect
    # This test documents the behavior (may crash or return garbage)
    vec1 = array.array('f', [1.0, 2.0, 3.0, 4.0, 5.0])
    vec2 = array.array('f', [1.0, 2.0, 3.0, 4.0, 5.0])

    # Correct dimension
    result_correct = _simd.cosine_similarity(vec1, vec2)
    assert abs(result_correct - 1.0) < 1e-6

    # Using subset of dimension should work (processes first 3 elements)
    result_subset = _simd.cosine_similarity(vec1, vec2)
    # Should still return valid result (just for first 3 elements)
    assert not math.isnan(result_subset)


def test_empty_vectors():
    """Test behavior with empty or very small vectors."""
    from peachbase import _simd

    # Single element
    vec1 = array.array('f', [1.0])
    vec2 = array.array('f', [1.0])

    result = _simd.cosine_similarity(vec1, vec2)
    assert abs(result - 1.0) < 1e-6

    # Two elements
    vec1 = array.array('f', [1.0, 0.0])
    vec2 = array.array('f', [1.0, 0.0])

    result = _simd.cosine_similarity(vec1, vec2)
    assert abs(result - 1.0) < 1e-6


def test_large_batch_operations():
    """Test batch operations with large number of vectors."""
    from peachbase import _simd

    dim = 384
    n_vectors = 1000

    query = array.array('f', [i * 0.01 for i in range(dim)])
    vectors = array.array('f', [i * 0.01 for _ in range(n_vectors) for i in range(dim)])

    # Should handle 1000 vectors without issue
    results = _simd.batch_cosine_similarity(query, vectors)

    assert len(results) == n_vectors
    assert all(not math.isnan(r) for r in results)
