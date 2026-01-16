"""Semantic (vector) search implementation using SIMD acceleration.

Uses brute-force search with SIMD optimizations for fast similarity computation.
Suitable for collections up to ~100K vectors.
"""

import array
import heapq
from typing import Literal

try:
    from peachbase import _simd

    SIMD_AVAILABLE = True
except ImportError:
    SIMD_AVAILABLE = False
    _simd = None


def semantic_search(
    query_vector: list[float],
    vectors: list[list[float]],
    metric: Literal["cosine", "l2", "dot"] = "cosine",
    limit: int = 10,
    candidate_indices: set[int] | None = None,
    flat_vectors: array.array | None = None,
) -> list[tuple[int, float]]:
    """Perform semantic search using vector similarity.

    Args:
        query_vector: Query embedding vector
        vectors: List of document vectors
        metric: Distance metric ("cosine", "l2", or "dot")
        limit: Maximum number of results
        candidate_indices: Optional set of candidate indices to search (for filtering)
        flat_vectors: Pre-flattened array for fast SIMD (avoids overhead)

    Returns:
        List of (doc_index, similarity_score) tuples, sorted by score

    Note:
        - For cosine similarity, higher is better (range: -1 to 1)
        - For L2 distance, lower is better (range: 0 to infinity)
        - For dot product, higher is better (range: -infinity to infinity)
    """
    if not vectors:
        return []

    # Determine which indices to search
    if candidate_indices is not None:
        search_indices = list(candidate_indices)
        search_vectors = [vectors[i] for i in search_indices]
    else:
        search_indices = list(range(len(vectors)))
        search_vectors = vectors

    if not search_vectors:
        return []

    # Compute similarities using SIMD if available
    if SIMD_AVAILABLE:
        if flat_vectors is not None and candidate_indices is None:
            # Use pre-flattened vectors for full search (much faster)
            scores = _compute_similarities_simd_flat(query_vector, flat_vectors, metric)
        elif flat_vectors is not None and candidate_indices is not None:
            # Extract subset of flat vectors for filtered search
            dim = len(query_vector)
            subset_flat = []
            for idx in search_indices:
                subset_flat.extend(flat_vectors[idx * dim : (idx + 1) * dim])
            scores = _compute_similarities_simd_flat(
                query_vector, array.array("f", subset_flat), metric
            )
        else:
            # Fallback: flatten on-the-fly (for backward compatibility)
            scores = _compute_similarities_simd(query_vector, search_vectors, metric)
    else:
        scores = _compute_similarities_python(query_vector, search_vectors, metric)

    # Create result tuples with original indices
    results = [(search_indices[i], scores[i]) for i in range(len(scores))]

    # Use heapq for efficient top-k selection instead of full sort
    if metric == "l2":
        # For L2, lower is better - use nsmallest
        return heapq.nsmallest(limit, results, key=lambda x: x[1])
    else:
        # For cosine and dot, higher is better - use nlargest
        return heapq.nlargest(limit, results, key=lambda x: x[1])


def _compute_similarities_simd(
    query: list[float], vectors: list[list[float]], metric: str
) -> list[float]:
    """Compute similarities using SIMD C extension.

    Args:
        query: Query vector
        vectors: List of vectors
        metric: Distance metric

    Returns:
        List of similarity scores
    """
    # Convert query to float array (for C buffer interface)
    query_array = array.array("f", query)

    # Flatten vectors matrix (single allocation, much faster)
    flat_vectors = []
    for vec in vectors:
        flat_vectors.extend(vec)
    vectors_array = array.array("f", flat_vectors)

    # Call optimized batch SIMD functions
    if metric == "cosine":
        return _simd.batch_cosine_similarity(query_array, vectors_array)
    elif metric == "l2":
        return _simd.batch_l2_distance(query_array, vectors_array)
    elif metric == "dot":
        return _simd.batch_dot_product(query_array, vectors_array)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _compute_similarities_simd_flat(
    query: list[float], flat_vectors: array.array, metric: str
) -> list[float]:
    """Compute similarities using SIMD C extension with pre-flattened vectors.

    Args:
        query: Query vector
        flat_vectors: Pre-flattened array of all vectors
        metric: Distance metric

    Returns:
        List of similarity scores
    """
    # Convert query to float array (for C buffer interface)
    query_array = array.array("f", query)

    # Call optimized batch SIMD functions (vectors already flat!)
    if metric == "cosine":
        return _simd.batch_cosine_similarity(query_array, flat_vectors)
    elif metric == "l2":
        return _simd.batch_l2_distance(query_array, flat_vectors)
    elif metric == "dot":
        return _simd.batch_dot_product(query_array, flat_vectors)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _compute_similarities_python(
    query: list[float], vectors: list[list[float]], metric: str
) -> list[float]:
    """Compute similarities using pure Python (fallback).

    Args:
        query: Query vector
        vectors: List of vectors
        metric: Distance metric

    Returns:
        List of similarity scores
    """
    scores = []

    for vec in vectors:
        if metric == "cosine":
            score = _cosine_similarity_python(query, vec)
        elif metric == "l2":
            score = _l2_distance_python(query, vec)
        elif metric == "dot":
            score = _dot_product_python(query, vec)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    return scores


def _cosine_similarity_python(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity in pure Python."""
    dot = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    denom = norm1 * norm2
    return dot / denom if denom > 1e-8 else 0.0


def _l2_distance_python(vec1: list[float], vec2: list[float]) -> float:
    """L2 distance in pure Python."""
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2, strict=True)) ** 0.5


def _dot_product_python(vec1: list[float], vec2: list[float]) -> float:
    """Dot product in pure Python."""
    return sum(a * b for a, b in zip(vec1, vec2, strict=True))
