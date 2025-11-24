# Test Suite Organization

This directory contains all tests for the Software-AsianOdds betting system.

## Structure

```
tests/
├── conftest.py          # Shared pytest fixtures and configuration
├── unit/                # Unit tests (fast, isolated)
│   └── test_*.py
├── integration/         # Integration tests (multiple components)
│   └── test_*.py
└── e2e/                 # End-to-end tests (full workflows)
    └── test_*.py
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# E2E tests only
pytest tests/e2e/
```

### Run with markers
```bash
# Run only unit tests (using marker)
pytest -m unit

# Run slow tests
pytest -m slow

# Exclude API tests
pytest -m "not api"
```

### Run with coverage
```bash
pytest --cov=. --cov-report=html
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual functions/classes in isolation
- Fast execution (< 1 second per test)
- No external dependencies (APIs, databases)
- Use mocking for external calls

**Example:**
```python
@pytest.mark.unit
def test_calculate_ev():
    result = calculate_ev(prob=0.6, odds=2.0)
    assert result > 0
```

### Integration Tests (`tests/integration/`)
- Test interaction between components
- May use test databases or API mocks
- Moderate execution time

**Example:**
```python
@pytest.mark.integration
def test_ai_pipeline_with_api():
    pipeline = AIPipeline(config=test_config)
    result = pipeline.analyze_match(mock_match_data)
    assert result.confidence > 0
```

### E2E Tests (`tests/e2e/`)
- Test complete workflows
- May require actual API access
- Slower execution
- Mark with `@pytest.mark.slow`

**Example:**
```python
@pytest.mark.e2e
@pytest.mark.slow
def test_full_betting_workflow():
    # Test complete flow from data fetch to betting recommendation
    pass
```

## Best Practices

1. **Naming Convention**: `test_<feature>.py` for files, `test_<action>` for functions
2. **One assertion per test**: Keep tests focused and clear
3. **Use fixtures**: Share common setup via fixtures in `conftest.py`
4. **Mark tests appropriately**: Use pytest markers for organization
5. **Mock external calls**: Don't make real API calls in unit tests
6. **Test edge cases**: Include tests for error conditions and boundaries

## Fixtures

Common fixtures are available in `conftest.py`:

- `project_root_path`: Path to project root
- `test_data_dir`: Path to test data directory
- `sample_match_data`: Sample match data for testing
- `sample_odds_data`: Sample odds data for testing

## Migration Status

Original test files from root are being migrated to this organized structure.
Legacy tests can still be found in the project root (test_*.py files).

## TODO
- [ ] Migrate all 50+ test files from root to appropriate directories
- [ ] Add coverage reporting
- [ ] Set up CI/CD pipeline for automated testing
- [ ] Add performance benchmarks
