# Tests

This directory contains tests for the mdn-typo-proofreading project.

## Running Tests

### Install dev dependencies

```bash
uv sync --all-extras
```

Or:

```bash
uv pip install -e ".[dev]"
```

### Run all tests

```bash
uv run pytest
```

### Run specific test file

```bash
uv run pytest tests/test_issue_to_pr.py
```

### Run specific test

```bash
uv run pytest tests/test_issue_to_pr.py::TestNormalizeDiff::test_removes_code_fences
```

### Run with coverage

```bash
uv run pytest --cov=.github/scripts --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

### Run with verbose output

```bash
uv run pytest -v
```

### Run with output capture disabled (see print statements)

```bash
uv run pytest -s
```

## Test Structure

- `test_issue_to_pr.py` - Tests for the GitHub Actions workflow script
  - `TestNormalizeDiff` - Tests for diff normalization
  - `TestComputeResultHash` - Tests for hash computation

## Adding New Tests

1. Create a new test file: `tests/test_<module>.py`
2. Import the module to test
3. Create test classes (prefixed with `Test`)
4. Create test functions (prefixed with `test_`)
5. Use pytest fixtures for common setup

Example:

```python
import pytest
from module import function_to_test

class TestMyFunction:
    """Tests for my_function."""

    @pytest.fixture
    def setup_data(self):
        """Common setup for tests."""
        return {"key": "value"}

    def test_basic_functionality(self, setup_data):
        """Should do something."""
        result = function_to_test(setup_data)
        assert result == expected_value
```

## Continuous Integration

Tests should be run in CI before merging. Consider adding a GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v7
      - run: uv sync --all-extras
      - run: uv run pytest
```

## Test Coverage Goals

- Core functions should have >90% coverage
- Critical paths (diff generation, git operations) should have 100% coverage
- Edge cases (empty input, invalid data) should be tested

## Maintenance

- Keep tests up to date with code changes
- Add tests for new features
- Update fixtures when data formats change
- Remove tests for deprecated functionality
