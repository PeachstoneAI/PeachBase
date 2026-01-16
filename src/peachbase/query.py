"""Query builder and result wrapper for PeachBase."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from peachbase.collection import Collection


class Query:
    """Query object for building and executing searches.

    Wraps search parameters and provides methods for retrieving results
    in different formats.

    Args:
        collection: Collection to search
        query_text: Text query for lexical/hybrid search
        query_vector: Vector query for semantic/hybrid search
        mode: Search mode ("lexical", "semantic", or "hybrid")
        metric: Distance metric for semantic search
        limit: Maximum number of results
        filter: Metadata filter (MongoDB-like syntax)
        alpha: Weight for hybrid search
    """

    def __init__(
        self,
        collection: "Collection",
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        mode: Literal["lexical", "semantic", "hybrid"] = "semantic",
        metric: Literal["cosine", "l2", "dot"] = "cosine",
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        alpha: float = 0.5,
    ) -> None:
        self.collection = collection
        self.query_text = query_text
        self.query_vector = query_vector
        self.mode = mode
        self.metric = metric
        self.limit = limit
        self.filter = filter
        self.alpha = alpha

        # Execute search
        self._results = self._execute()

    def _execute(self) -> list[dict[str, Any]]:
        """Execute the search query.

        Returns:
            List of result documents with scores
        """
        from peachbase.search.filters import apply_filter
        from peachbase.search.hybrid import hybrid_search
        from peachbase.search.semantic import semantic_search

        # Apply metadata filter if present
        candidate_indices = None
        if self.filter:
            candidate_indices = apply_filter(self.collection._documents, self.filter)
            if not candidate_indices:
                return []

        # Execute search based on mode
        if self.mode == "semantic":
            if self.query_vector is None:
                raise ValueError("query_vector required for semantic search")

            # Semantic search
            results = semantic_search(
                query_vector=self.query_vector,
                vectors=self.collection._vectors,
                metric=self.metric,
                limit=self.limit,
                candidate_indices=candidate_indices,
                flat_vectors=self.collection._ensure_flat_vectors(),
            )

            # Convert to result format
            return self._format_semantic_results(results)

        elif self.mode == "lexical":
            if self.query_text is None:
                raise ValueError("query_text required for lexical search")

            # Build BM25 index if needed
            self.collection._rebuild_indices()

            if self.collection._bm25_index is None:
                return []

            # BM25 search
            results = self.collection._bm25_index.search(
                self.query_text, limit=self.limit * 2
            )

            # Apply filter if needed
            if candidate_indices:
                filtered_results = []
                for doc_id, score in results:
                    doc_idx = self.collection._doc_index.get(doc_id)
                    if doc_idx is not None and doc_idx in candidate_indices:
                        filtered_results.append((doc_id, score))
                results = filtered_results[: self.limit]

            return self._format_lexical_results(results)

        elif self.mode == "hybrid":
            if self.query_text is None or self.query_vector is None:
                raise ValueError(
                    "Both query_text and query_vector required for hybrid search"
                )

            # Build BM25 index if needed
            self.collection._rebuild_indices()

            if self.collection._bm25_index is None:
                return []

            # Run lexical and semantic searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both searches
                lexical_future = executor.submit(
                    self.collection._bm25_index.search, self.query_text, self.limit * 2
                )
                semantic_future = executor.submit(
                    semantic_search,
                    query_vector=self.query_vector,
                    vectors=self.collection._vectors,
                    metric=self.metric,
                    limit=self.limit * 2,
                    candidate_indices=candidate_indices,
                    flat_vectors=self.collection._ensure_flat_vectors(),
                )

                # Get results from both
                lexical_results = lexical_future.result()
                semantic_results = semantic_future.result()

            # Create doc_id map for hybrid search
            doc_id_map = {
                i: doc["id"] for i, doc in enumerate(self.collection._documents)
            }

            # Combine results
            combined_results = hybrid_search(
                lexical_results=lexical_results,
                semantic_results=semantic_results,
                doc_id_map=doc_id_map,
                method="rrf",
                alpha=self.alpha,
                limit=self.limit,
            )

            return self._format_hybrid_results(combined_results)

        else:
            raise ValueError(f"Unknown search mode: {self.mode}")

    def _format_semantic_results(
        self, results: list[tuple[int, float]]
    ) -> list[dict[str, Any]]:
        """Format semantic search results."""
        formatted = []
        for doc_idx, score in results:
            doc = self.collection._documents[doc_idx].copy()
            doc["score"] = score
            doc["vector"] = self.collection._vectors[doc_idx]
            formatted.append(doc)
        return formatted

    def _format_lexical_results(
        self, results: list[tuple[str, float]]
    ) -> list[dict[str, Any]]:
        """Format lexical search results."""
        formatted = []
        for doc_id, score in results:
            doc_idx = self.collection._doc_index.get(doc_id)
            if doc_idx is not None:
                doc = self.collection._documents[doc_idx].copy()
                doc["score"] = score
                doc["vector"] = self.collection._vectors[doc_idx]
                formatted.append(doc)
        return formatted

    def _format_hybrid_results(
        self, results: list[tuple[str, float]]
    ) -> list[dict[str, Any]]:
        """Format hybrid search results."""
        formatted = []
        for doc_id, score in results:
            doc_idx = self.collection._doc_index.get(doc_id)
            if doc_idx is not None:
                doc = self.collection._documents[doc_idx].copy()
                doc["score"] = score
                doc["vector"] = self.collection._vectors[doc_idx]
                formatted.append(doc)
        return formatted

    def to_list(self) -> list[dict[str, Any]]:
        """Return results as a list of dictionaries.

        Returns:
            List of result documents with scores
        """
        return self._results

    def to_dict(self) -> dict[str, Any]:
        """Return results as a dictionary with metadata.

        Returns:
            Dictionary with results and query metadata
        """
        return {
            "results": self._results,
            "count": len(self._results),
            "mode": self.mode,
            "limit": self.limit,
        }

    def __iter__(self):
        """Iterate over results."""
        return iter(self._results)

    def __len__(self) -> int:
        """Get number of results."""
        return len(self._results)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get result by index."""
        return self._results[index]

    def __repr__(self) -> str:
        """String representation."""
        return f"Query(mode='{self.mode}', results={len(self._results)})"
