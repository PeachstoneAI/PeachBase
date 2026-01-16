# Contributing to PeachBase

Thank you for your interest in contributing to PeachBase! This document provides guidelines and instructions for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [Pull Request Process](#pull-request-process)
8. [Reporting Bugs](#reporting-bugs)
9. [Suggesting Features](#suggesting-features)
10. [Project Structure](#project-structure)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment. We expect:

- **Be respectful**: Treat all contributors with respect
- **Be collaborative**: Work together towards common goals
- **Be constructive**: Provide helpful feedback
- **Be patient**: Remember that everyone is learning

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- C compiler (gcc, clang, or MSVC)
- Git
- Basic understanding of vector databases and embeddings

### Find an Issue

1. Browse [open issues](https://github.com/PeachstoneAI/peachbase/issues)
2. Look for issues labeled `good first issue` or `help wanted`
3. Comment on the issue to let others know you're working on it
4. Wait for maintainer approval before starting significant work

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/PeachBase.git
cd peachbase

# Add upstream remote
git remote add upstream https://github.com/PeachstoneAI/PeachBase.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### 3. Install Development Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev,test]"

# Or manually install dependencies
pip install -r requirements-test.txt
pip install ruff mypy pytest pytest-cov
```

### 4. Build C Extensions

```bash
# Build with OpenMP (recommended for development)
python -m build

# Install the built wheel
pip install dist/peachbase-*.whl --force-reinstall

# Or build without OpenMP
PEACHBASE_DISABLE_OPENMP=1 python -m build
```

### 5. Verify Installation

```bash
# Run quick test
python examples/quick_test.py

# Run test suite
pytest tests/ -v
```

---

## Making Changes

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions/improvements
- `refactor/` - Code refactoring
- `perf/` - Performance improvements

### 2. Make Your Changes

Follow these guidelines:

#### Python Code
- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for all function signatures
- Add docstrings for public functions/classes
- Keep functions focused and small
- Write self-documenting code

#### C Code
- Follow existing code style
- Add comments for complex algorithms
- Ensure memory safety (no leaks, buffer overflows)
- Test on multiple platforms if possible

#### Documentation
- Update relevant `.md` files
- Add docstrings to new functions
- Update API reference if needed
- Add examples for new features

### 3. Write Tests

All changes must include tests:

```python
# tests/test_your_feature.py
import pytest
import peachbase

def test_your_new_feature(temp_db_path):
    """Test description."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Your test code
    assert collection is not None
```

### 4. Run Tests Locally

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_collection.py -v

# Run with coverage
pytest tests/ --cov=peachbase --cov-report=html

# Check coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 5. Format and Lint

```bash
# Format code with ruff
ruff format src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type check with mypy
mypy src/
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_collection.py

# Specific test function
pytest tests/test_collection.py::test_add_documents

# With verbose output
pytest tests/ -v

# With coverage
pytest tests/ --cov=peachbase
```

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── README.md                # Testing documentation
├── unit/                    # Unit tests
│   ├── test_collection.py   # Collection operations
│   ├── test_database.py     # Database operations
│   ├── test_filters.py      # Metadata filtering
│   ├── test_search.py       # Search operations
│   ├── test_simd.py         # SIMD C extensions
│   └── test_storage.py      # Save/load persistence
└── integration/             # Integration tests
    └── test_s3.py           # S3 operations
```

### Writing Good Tests

```python
def test_feature_with_clear_name(temp_db_path, sample_documents):
    """
    Test that does X when Y happens.

    This test verifies that...
    """
    # Arrange: Set up test data
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Act: Perform the action
    collection.add(sample_documents)

    # Assert: Verify the results
    assert collection.size == len(sample_documents)
```

### Test Coverage Goals

- **Minimum**: 80% code coverage
- **Target**: 90% code coverage
- All new features must have tests
- Bug fixes must include regression tests

---

## Code Style

### Python Style (PEP 8)

```python
# Good
def search_documents(
    collection: Collection,
    query_vector: list[float],
    limit: int = 10
) -> list[dict]:
    """Search for similar documents.

    Args:
        collection: PeachBase collection to search
        query_vector: Query embedding vector
        limit: Maximum number of results

    Returns:
        List of matching documents with scores
    """
    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=limit
    )
    return results.to_list()
```

### C Style

```c
// Good
/**
 * Calculate cosine similarity between two vectors.
 *
 * @param vec1 First vector
 * @param vec2 Second vector
 * @param dim Vector dimension
 * @return Cosine similarity score (-1 to 1)
 */
float cosine_similarity(const float *vec1, const float *vec2, int dim) {
    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (int i = 0; i < dim; i++) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    return dot / (sqrtf(norm1) * sqrtf(norm2));
}
```

### Configuration

We use automated tools:

```toml
# pyproject.toml

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "C4", "SIM"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
disallow_untyped_defs = true
```

---

## Pull Request Process

### 1. Commit Your Changes

```bash
# Stage changes
git add src/peachbase/your_file.py tests/test_your_feature.py

# Commit with clear message
git commit -m "Add feature: description of what you added

- Detailed point 1
- Detailed point 2
- Fixes #issue_number"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- First line: concise summary (50 chars or less)
- Body: detailed explanation if needed
- Reference issues: "Fixes #123" or "Relates to #456"

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to [PeachBase repository](https://github.com/PeachstoneAI/peachbase)
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added/updated
- [ ] All tests pass
```

### 4. Code Review

- Respond to reviewer feedback
- Make requested changes
- Push updates to your branch (PR updates automatically)
- Request re-review when ready

### 5. Merge

Once approved, a maintainer will merge your PR. Your changes will be included in the next release!

---

## Reporting Bugs

### Before Reporting

1. Check [existing issues](https://github.com/PeachstoneAI/peachbase/issues)
2. Verify you're using the latest version
3. Try to reproduce with minimal code

### Bug Report Template

```markdown
**Describe the Bug**
Clear description of what's wrong

**To Reproduce**
Steps to reproduce:
1. Create collection with...
2. Add documents...
3. Search with...
4. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Code Example**
```python
# Minimal code to reproduce
import peachbase
db = peachbase.connect("./test_db")
# ...
```

**Environment**
- PeachBase version: 0.1.0
- Python version: 3.11.5
- OS: Ubuntu 22.04
- CPU: Intel i7 (AVX2)

**Additional Context**
Error messages, logs, screenshots
```

---

## Suggesting Features

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why is this feature needed?
What problem does it solve?

**Proposed Solution**
How should this work?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Examples, mockups, related features
```

### Feature Discussion

- Open an issue with `feature request` label
- Discuss with maintainers before implementing
- Wait for approval before starting work
- Large features may need a design document

---

## Project Structure

```
peachbase/
├── src/peachbase/          # Python package
│   ├── __init__.py       # Package exports
│   ├── database.py       # Database class
│   ├── collection.py     # Collection class
│   ├── query.py          # Query builder
│   ├── search/           # Search implementations
│   │   ├── semantic.py   # Semantic search
│   │   ├── bm25.py       # BM25 lexical search
│   │   ├── hybrid.py     # Hybrid search (RRF)
│   │   └── filters.py    # Metadata filtering
│   ├── storage/          # Storage layer
│   │   ├── format.py     # Binary format
│   │   ├── writer.py     # Serialization
│   │   └── reader.py     # Deserialization
│   └── text/             # Text processing
│       └── tokenizer.py  # BM25 tokenizer
├── csrc/                 # C extensions
│   ├── peachbase_simd.c    # SIMD operations
│   ├── peachbase_simd.h    # SIMD header
│   ├── peachbase_bm25.c    # BM25 scoring
│   └── peachbase_bm25.h    # BM25 header
├── tests/                # Test suite
├── examples/             # Example scripts
├── docs/                 # Documentation
├── pyproject.toml        # Package configuration
├── setup.py              # C extension build
└── README.md             # Project README
```

---

## Releasing (Maintainers)

### Creating a Release

1. **Update version** in `src/peachbase/_version.py` and `pyproject.toml`

2. **Update changelog** in `docs/changelog.md`

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Bump version to X.Y.Z"
   git push
   ```

4. **Create a git tag**:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

5. **Create GitHub Release**:
   - Go to https://github.com/PeachstoneAI/PeachBase/releases/new
   - Select the tag you just created
   - Title: `vX.Y.Z`
   - Description: Copy relevant section from changelog
   - Click "Publish release"

6. **Automatic publishing**: GitHub Actions will build wheels and publish to PyPI

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

---

## Questions?

- Open a [discussion](https://github.com/PeachstoneAI/peachbase/discussions)
- Ask in issue comments
- Check [documentation](docs/README.md)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Recognition

Contributors will be:
- Listed in CHANGELOG.md
- Mentioned in release notes
- Added to CONTRIBUTORS.md (if significant contributions)

Thank you for contributing to PeachBase! 🍑
