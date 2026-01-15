# PeachBase Tests

Comprehensive test suite for PeachBase using pytest.

---

## Test Organization

Tests are organized into two main categories:

### Unit Tests (`tests/unit/`)
Fast, isolated tests that test individual components without external dependencies:
- 114 tests covering core functionality
- Run in ~0.5 seconds
- No network or I/O dependencies

### Integration Tests (`tests/integration/`)
Tests that verify integration with external services:
- 13 tests for S3 integration
- Use `moto` to mock AWS services
- Run in ~3.5 seconds

See `tests/unit/README.md` and `tests/integration/README.md` for detailed documentation.

---

## Running Tests

### Run All Tests (127 tests)
```bash
pytest tests/
```

### Run Only Unit Tests (114 tests)
```bash
pytest tests/unit/
```

### Run Only Integration Tests (13 tests)
```bash
pytest tests/integration/
```

### Run Specific Test File
```bash
pytest tests/unit/test_collection.py
pytest tests/unit/test_search_semantic.py
pytest tests/integration/test_s3.py
```

### Run Specific Test Function
```bash
pytest tests/unit/test_database.py::test_create_collection
pytest tests/unit/test_search_lexical.py::test_lexical_search_basic
```

### Run Tests with Verbose Output
```bash
pytest -v
```

### Run Tests with Coverage
```bash
pip install pytest-cov
pytest --cov=src/peachbase --cov-report=html
```

---

## Quick Reference

### `conftest.py`
Shared pytest fixtures used across all tests:
- `temp_db_path`: Temporary database directory
- `sample_documents`: 5 sample documents for testing
- `random_vectors`: Generator for random test vectors
- `query_vector`: Sample query vector

### Unit Test Modules

**Database** (`unit/test_database.py`)
- Creating/getting/deleting collections
- Collection listing
- Multiple collections
- Error handling

**Collection** (`unit/test_collection.py`)
- Adding documents (single, batch, duplicates)
- Getting documents
- Updating documents
- Deleting documents
- Size and dimension properties
- Field validation

**Semantic Search** (`unit/test_search_semantic.py`)
- All distance metrics (cosine, L2, dot)
- Limit parameter
- Result sorting and structure
- Empty collections
- Large collections
- Iterator and length support

**Lexical Search** (`unit/test_search_lexical.py`)
- Keyword matching
- Case insensitivity
- Punctuation handling
- Term frequency effects
- Multiple terms
- Empty queries and no matches

**Hybrid Search** (`unit/test_search_hybrid.py`)
- Combined semantic + lexical search
- Alpha parameter tuning (0.0 to 1.0)
- Reciprocal rank fusion
- Required parameters validation
- Different metrics

**Metadata Filters** (`unit/test_filters.py`)
- Exact match filters
- Comparison operators ($gte, $lte, $gt, $lt)
- $in operator
- Logical operators ($and, $or)
- Complex nested queries
- Filter with all search modes

**Storage** (`unit/test_storage.py`)
- Save and load operations
- Multiple save/load cycles
- Index preservation
- Document updates/deletes
- Empty collections
- Large collections
- Metadata type preservation

**SIMD Extensions** (`unit/test_simd.py`)
- CPU feature detection
- OpenMP information
- Cosine similarity (basic, orthogonal, opposite)
- L2 distance calculations
- Dot product calculations
- Batch operations
- 384-dimension vectors
- Performance characteristics

### Integration Test Modules

**S3 Integration** (`integration/test_s3.py`)
- Lazy loading of boto3
- Listing collections from S3
- Downloading and caching from S3
- Deleting S3 objects
- Database operations with S3 URIs

---

## Test Coverage

Current test coverage:

| Module | Category | Tests | Runtime |
|--------|----------|-------|---------|
| Database | Unit | 12 | < 0.1s |
| Collection | Unit | 14 | < 0.1s |
| Search (Semantic) | Unit | 14 | < 0.1s |
| Search (Lexical) | Unit | 15 | < 0.1s |
| Search (Hybrid) | Unit | 13 | < 0.1s |
| Filters | Unit | 17 | < 0.1s |
| Storage | Unit | 12 | < 0.1s |
| SIMD | Unit | 17 | < 0.1s |
| S3 Integration | Integration | 13 | ~3.5s |

**Total: 127 tests (114 unit + 13 integration)**

---

## Writing New Tests

### Test File Naming
- Name test files `test_<module>.py`
- Place in `tests/unit/` for unit tests
- Place in `tests/integration/` for integration tests

### Test Function Naming
- Name test functions `test_<what_it_tests>`
- Use descriptive names: `test_filter_exact_match` not `test_filter1`

### Choosing Unit vs Integration
- **Unit tests**: Test single components in isolation, no external dependencies
- **Integration tests**: Test integration with external services (S3, APIs, etc.)

### Using Fixtures
```python
def test_example(temp_db_path, sample_documents):
    """Test with fixtures from conftest.py."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)
    # ... test logic
```

### Assertions
```python
# Good: Specific assertions
assert collection.size == 5
assert result["score"] > 0.5
assert "doc1" in doc_ids

# Bad: Vague assertions
assert collection
assert result
```

### Error Testing
```python
# Test that errors are raised
with pytest.raises(ValueError, match="dimension"):
    collection.add([{...}])
```

### Parametrize for Multiple Inputs
```python
@pytest.mark.parametrize("metric", ["cosine", "l2", "dot"])
def test_all_metrics(temp_db_path, metric):
    # Test runs 3 times, once for each metric
    ...
```

---

## Continuous Integration

Tests are automatically run on:
- Every commit (via pre-commit hook)
- Pull requests
- Before releases

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pytest
          pip install dist/*.whl
      - name: Run tests
        run: pytest
```

---

## Debugging Failed Tests

### Run Failed Tests Only
```bash
pytest --lf  # last failed
pytest --ff  # failed first
```

### Run with Print Statements
```bash
pytest -s  # don't capture output
```

### Run with Debugger
```bash
pytest --pdb  # drop into pdb on failure
```

### Show Full Traceback
```bash
pytest -v --tb=long
```

---

## Performance Testing

Some tests check performance characteristics:

```bash
# Run only slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"
```

Mark slow tests:
```python
@pytest.mark.slow
def test_large_collection():
    # Test with 10K+ documents
    ...
```

---

## Test Data

### Sample Documents
The `sample_documents` fixture provides 5 documents covering:
- Different categories (ai, nlp, cv, rl)
- Different years (2023, 2024)
- Varied text content
- 384-dimensional vectors

### Temporary Directories
The `temp_db_path` fixture:
- Creates a new temp directory for each test
- Automatically cleaned up after test completes
- Ensures test isolation

---

## Best Practices

1. **Isolation**: Each test should be independent
2. **Clear Names**: Test names should describe what they test
3. **One Concept**: Each test should verify one thing
4. **Fast**: Keep tests fast (use small datasets)
5. **Deterministic**: Tests should give same result every time

---

## Common Issues

### Import Errors
```bash
# Make sure PeachBase is installed
pip install dist/*.whl

# Or install in development mode
pip install -e .
```

### Temp Directory Issues
```bash
# Clear temp directories manually if needed
rm -rf /tmp/peachbase_test_*
```

### SIMD Tests Failing
- Check that C extensions are compiled correctly
- Run `python -c "from peachbase import _simd; print(_simd)"`

---

## Contributing

When adding new features:

1. Write tests FIRST (TDD)
2. Run tests to ensure they fail
3. Implement feature
4. Run tests to ensure they pass
5. Check coverage: `pytest --cov`

Target coverage: >90% for all modules

---

Made with 🍑 by the PeachBase team
