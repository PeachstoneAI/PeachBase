"""Hybrid search combining lexical (BM25) and semantic (vector) search.

Implements Reciprocal Rank Fusion (RRF) for combining results.
"""

from typing import Literal


def hybrid_search_rrf(
    lexical_results: list[tuple[str, float]],
    semantic_results: list[tuple[int, float]],
    doc_id_map: dict[int, str],
    alpha: float = 0.5,
    k: int = 60,
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Combine lexical and semantic search results using Reciprocal Rank Fusion (RRF).

    RRF: score = alpha * 1/(k+rank_lex) + (1-alpha) * 1/(k+rank_sem)

    Args:
        lexical_results: BM25 results as list of (doc_id, score)
        semantic_results: Semantic results as list of (doc_index, score)
        doc_id_map: Mapping from doc_index to doc_id
        alpha: Lexical weight (0=semantic, 1=lexical, default 0.5)
        k: RRF parameter (default=60, from literature)
        limit: Maximum number of results

    Returns:
        List of (doc_id, combined_score) tuples, sorted by score descending
    """
    # Convert semantic results to use doc_id
    semantic_results_ids = [(doc_id_map[idx], score) for idx, score in semantic_results]

    # Build rank dictionaries
    lexical_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(lexical_results)}
    semantic_ranks = {
        doc_id: rank for rank, (doc_id, _) in enumerate(semantic_results_ids)
    }

    # Get all unique doc_ids
    all_doc_ids = set(lexical_ranks.keys()) | set(semantic_ranks.keys())

    # Compute RRF scores
    rrf_scores = {}
    for doc_id in all_doc_ids:
        lexical_rrf = 0.0
        if doc_id in lexical_ranks:
            lexical_rrf = 1.0 / (k + lexical_ranks[doc_id])

        semantic_rrf = 0.0
        if doc_id in semantic_ranks:
            semantic_rrf = 1.0 / (k + semantic_ranks[doc_id])

        combined_score = alpha * lexical_rrf + (1.0 - alpha) * semantic_rrf
        rrf_scores[doc_id] = combined_score

    # Sort by combined score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:limit]


def hybrid_search_weighted(
    lexical_results: list[tuple[str, float]],
    semantic_results: list[tuple[int, float]],
    doc_id_map: dict[int, str],
    alpha: float = 0.5,
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Combine lexical and semantic search results using weighted score fusion.

    Normalizes scores to [0, 1] range and combines with alpha weighting.
    Formula: score(d) = alpha * norm_lexical(d) + (1-alpha) * norm_semantic(d)

    Args:
        lexical_results: BM25 results as list of (doc_id, score)
        semantic_results: Semantic results as list of (doc_index, score)
        doc_id_map: Mapping from doc_index to doc_id
        alpha: Lexical weight (0=semantic, 1=lexical, default 0.5)
        limit: Maximum number of results

    Returns:
        List of (doc_id, combined_score) tuples, sorted by score descending
    """
    # Convert semantic results to use doc_id
    semantic_results_ids = [(doc_id_map[idx], score) for idx, score in semantic_results]

    # Normalize lexical scores
    lexical_scores = {}
    if lexical_results:
        max_lexical = (
            max(score for _, score in lexical_results) if lexical_results else 1.0
        )
        min_lexical = (
            min(score for _, score in lexical_results) if lexical_results else 0.0
        )
        range_lexical = max_lexical - min_lexical

        for doc_id, score in lexical_results:
            if range_lexical > 1e-8:
                normalized = (score - min_lexical) / range_lexical
            else:
                normalized = 1.0
            lexical_scores[doc_id] = normalized

    # Normalize semantic scores
    semantic_scores = {}
    if semantic_results_ids:
        max_semantic = max(score for _, score in semantic_results_ids)
        min_semantic = min(score for _, score in semantic_results_ids)
        range_semantic = max_semantic - min_semantic

        for doc_id, score in semantic_results_ids:
            if range_semantic > 1e-8:
                normalized = (score - min_semantic) / range_semantic
            else:
                normalized = 1.0
            semantic_scores[doc_id] = normalized

    # Get all unique doc_ids
    all_doc_ids = set(lexical_scores.keys()) | set(semantic_scores.keys())

    # Compute weighted scores
    combined_scores = {}
    for doc_id in all_doc_ids:
        lex_score = lexical_scores.get(doc_id, 0.0)
        sem_score = semantic_scores.get(doc_id, 0.0)
        combined_scores[doc_id] = alpha * lex_score + (1.0 - alpha) * sem_score

    # Sort by combined score
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:limit]


def hybrid_search(
    lexical_results: list[tuple[str, float]],
    semantic_results: list[tuple[int, float]],
    doc_id_map: dict[int, str],
    method: Literal["rrf", "weighted"] = "rrf",
    alpha: float = 0.5,
    k: int = 60,
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Perform hybrid search combining lexical and semantic results.

    Args:
        lexical_results: BM25 results as list of (doc_id, score)
        semantic_results: Semantic results as list of (doc_index, score)
        doc_id_map: Mapping from doc_index to doc_id
        method: Fusion method ("rrf" or "weighted")
        alpha: Weight for lexical vs semantic (default=0.5)
        k: RRF parameter (only for RRF method, default=60)
        limit: Maximum number of results

    Returns:
        List of (doc_id, combined_score) tuples, sorted by score descending
    """
    if method == "rrf":
        return hybrid_search_rrf(
            lexical_results, semantic_results, doc_id_map, alpha=alpha, k=k, limit=limit
        )
    elif method == "weighted":
        return hybrid_search_weighted(
            lexical_results, semantic_results, doc_id_map, alpha=alpha, limit=limit
        )
    else:
        raise ValueError(f"Unknown hybrid search method: {method}")
