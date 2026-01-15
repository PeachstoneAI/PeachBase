# Unit Tests

This directory contains unit tests for PeachBase that test individual components in isolation without external dependencies.

## Test Categories

### Collection Tests (`test_collection.py`)
Tests for the Collection class:
- Adding documents
- Getting and deleting documents
- Document validation
- Collection properties

### Database Tests (`test_database.py`)
Tests for the Database class:
- Creating and opening databases
- Collection management
- Listing and dropping collections

### Filter Tests (`test_filters.py`)
Tests for metadata filtering:
- Exact match filtering
- Comparison operators ($gte, $lte, $gt, $lt)
- Set operations ($in)
- Logical operators ($and, $or)
- Complex nested queries

### Search Tests
- **`test_search_semantic.py`** - Semantic (vector) search with SIMD
- **`test_search_lexical.py`** - Lexical (BM25) text search
- **`test_search_hybrid.py`** - Hybrid search combining semantic and lexical

### SIMD Tests (`test_simd.py`)
Tests for C extension SIMD operations:
- Cosine similarity
- L2 distance
- Dot product
- Batch operations
- CPU feature detection

### Storage Tests (`test_storage.py`)
Tests for persistence layer:
- Saving and loading collections
- Binary format serialization
- Multiple save/load cycles

## Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_collection.py

# Run with verbose output
pytest tests/unit/ -v

# Run only unit tests (exclude integration)
pytest tests/unit/
```

## Characteristics

Unit tests should:
- Run fast (no I/O, network, or external services)
- Be isolated (no dependencies on other tests)
- Test single components in isolation
- Use mocking for external dependencies if needed
