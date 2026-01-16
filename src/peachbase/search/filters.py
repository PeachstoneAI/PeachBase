"""Metadata filtering with MongoDB-like query syntax.

Supports:
- Exact match: {"field": "value"}
- Comparison: {"field": {"$gt": 10, "$lte": 100}}
- Array membership: {"field": {"$in": ["a", "b"]}}
- Logical operators: {"$and": [...], "$or": [...]}
"""

from typing import Any


def apply_filter(
    documents: list[dict[str, Any]], filter_query: dict[str, Any]
) -> set[int]:
    """Apply metadata filter to documents and return matching indices.

    Args:
        documents: List of documents with 'metadata' field
        filter_query: MongoDB-like filter query

    Returns:
        Set of document indices that match the filter

    Examples:
        >>> docs = [
        ...     {"id": "1", "metadata": {"category": "tech", "year": 2023}},
        ...     {"id": "2", "metadata": {"category": "news", "year": 2024}},
        ... ]
        >>> apply_filter(docs, {"category": "tech"})
        {0}
        >>> apply_filter(docs, {"year": {"$gte": 2024}})
        {1}
    """
    matching_indices = set()

    for idx, doc in enumerate(documents):
        metadata = doc.get("metadata", {})
        if _match_document(metadata, filter_query):
            matching_indices.add(idx)

    return matching_indices


def _match_document(metadata: dict[str, Any], filter_query: dict[str, Any]) -> bool:
    """Check if a document's metadata matches the filter query.

    Args:
        metadata: Document metadata
        filter_query: Filter query

    Returns:
        True if document matches, False otherwise
    """
    if not filter_query:
        return True

    # Handle logical operators
    if "$and" in filter_query:
        return all(_match_document(metadata, q) for q in filter_query["$and"])

    if "$or" in filter_query:
        return any(_match_document(metadata, q) for q in filter_query["$or"])

    if "$not" in filter_query:
        return not _match_document(metadata, filter_query["$not"])

    # Handle field queries
    for field, value in filter_query.items():
        if field.startswith("$"):
            # Skip logical operators already handled
            continue

        if not _match_field(metadata.get(field), value):
            return False

    return True


def _match_field(field_value: Any, query_value: Any) -> bool:
    """Check if a field value matches the query value.

    Args:
        field_value: Actual field value from document
        query_value: Query value (can be value or operator dict)

    Returns:
        True if matches, False otherwise
    """
    # If query_value is a dict, it contains operators
    if isinstance(query_value, dict):
        return _match_operators(field_value, query_value)
    else:
        # Exact match
        return field_value == query_value


def _match_operators(field_value: Any, operators: dict[str, Any]) -> bool:
    """Match field value against operator expressions.

    Args:
        field_value: Actual field value
        operators: Dict of operators and their values

    Returns:
        True if all operators match, False otherwise
    """
    for op, op_value in operators.items():
        if op == "$eq":
            if field_value != op_value:
                return False

        elif op == "$ne":
            if field_value == op_value:
                return False

        elif op == "$gt":
            if not (field_value is not None and field_value > op_value):
                return False

        elif op == "$gte":
            if not (field_value is not None and field_value >= op_value):
                return False

        elif op == "$lt":
            if not (field_value is not None and field_value < op_value):
                return False

        elif op == "$lte":
            if not (field_value is not None and field_value <= op_value):
                return False

        elif op == "$in":
            if field_value not in op_value:
                return False

        elif op == "$nin":
            if field_value in op_value:
                return False

        elif op == "$exists":
            exists = field_value is not None
            if exists != op_value:
                return False

        else:
            raise ValueError(f"Unknown operator: {op}")

    return True
