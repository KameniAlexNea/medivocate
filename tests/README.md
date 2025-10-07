# Test Suite for Medivocate RAG System

This directory contains comprehensive test cases for the `src` package of the Medivocate RAG (Retrieval-Augmented Generation) system.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_config.py          # Configuration class tests
├── test_factory.py         # Factory function tests
├── test_vector_store.py    # Vector store manager tests
├── test_rag_system.py      # RAG system functionality tests
├── test_utilities.py       # Utility function tests
├── test_preprocessing.py   # Document preprocessing tests
└── __init__.py
```

## Running Tests

### Run All Tests
```bash
# From project root
python run_tests.py
# or
pytest tests/
```

### Run Specific Test File
```bash
# Run configuration tests
python run_tests.py config

# Run RAG system tests
python run_tests.py rag_system
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Markers
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### Unit Tests (`-m unit`)
- Test individual functions and methods in isolation
- Use mocks and stubs for external dependencies
- Focus on logic and edge cases

### Integration Tests (`-m integration`)
- Test component interactions
- May require actual dependencies (marked as slow)
- Verify end-to-end functionality

### Slow Tests (`-m slow`)
- Tests that take significant time to run
- Typically integration tests with real dependencies
- Can be skipped for quick development cycles

## Key Test Areas

### 1. Configuration Testing
- **RAGConfig**: Main system configuration
- **VectorStoreConfig**: Vector database settings
- **ChunkingConfig**: Document chunking parameters
- **Validation**: Parameter validation and defaults

### 2. Factory Functions
- **create_rag_system()**: RAG system instantiation
- **create_vector_store_manager()**: Vector store creation
- **Parameter handling**: Different input types and validation

### 3. Vector Store Managers
- **BaseVectorStoreManager**: Abstract base class functionality
- **VectorStoreManager**: Single vector store (Chroma only)
- **EnsembleVectorStoreManager**: Ensemble (Chroma + BM25)
- **Inheritance**: Proper inheritance and method overriding
- **Document processing**: Batch processing and initialization

### 4. RAG System
- **Initialization**: Proper setup with dependencies
- **Document loading**: File loading and processing
- **Vector store integration**: Store initialization and management
- **Chain setup**: Retrieval and generation chain creation
- **Query processing**: End-to-end query handling
- **Error handling**: Graceful failure management

### 5. Utilities
- **Data loading**: Google Drive download and extraction
- **Embedding models**: CPU/CUDA device selection
- **Model initialization**: Proper model setup and error handling

### 6. Preprocessing
- **Document processing**: Loading and chunking
- **Text splitting**: Different chunking strategies
- **Metadata handling**: Document metadata preservation

## Test Fixtures

### Shared Fixtures (`conftest.py`)
- `temp_directory`: Temporary directory for file operations
- `sample_config`: Standard RAG configuration
- `custom_chunking_config`: Custom chunking settings
- `vector_store_config`: Vector store configuration
- `sample_documents`: Sample document objects for testing
- `mock_llm`: Mock language model
- `mock_vector_store_manager`: Mock vector store manager

## Mocking Strategy

The tests use comprehensive mocking to:
- Isolate unit tests from external dependencies
- Speed up test execution
- Test error conditions safely
- Verify correct API usage

Key mocked components:
- LangChain components (LLMs, retrievers, chains)
- File system operations
- External API calls
- Database connections

## Coverage Goals

- **Minimum Coverage**: 80% overall
- **Critical Paths**: 90%+ coverage for core functionality
- **Error Handling**: All error paths tested
- **Edge Cases**: Boundary conditions and unusual inputs

## Adding New Tests

### Test File Structure
```python
"""Tests for [component_name]."""
import pytest
from unittest.mock import patch, MagicMock

from src.[module] import [Component]


class Test[Component]:
    """Test [Component] functionality."""

    def test_[functionality_name](self):
        """Test [specific functionality]."""
        # Arrange
        # Act
        # Assert
        pass
```

### Test Naming Conventions
- `test_[functionality_name]`: Unit test for specific functionality
- `test_[component]_[action]`: Component action testing
- `test_[error_condition]`: Error condition testing
- `test_[edge_case]`: Edge case testing

### Mocking Best Practices
- Use `MagicMock` for complex objects
- Patch at the source (where imported)
- Verify call signatures and return values
- Test both success and failure paths

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- No external dependencies required for unit tests
- Fast execution (< 30 seconds for unit tests)
- Deterministic results
- Clear pass/fail criteria

## Debugging Tests

### Running Failed Tests
```bash
pytest tests/test_failed.py::TestClass::test_method -v -s
```

### Debugging with PDB
```bash
pytest tests/ --pdb
```

### Verbose Output
```bash
pytest tests/ -v --tb=long
```

## Test Maintenance

### Updating Tests After Refactoring
1. Update import paths
2. Modify mock targets if APIs change
3. Add new test cases for new functionality
4. Remove obsolete tests

### Performance Monitoring
- Track test execution time
- Identify slow tests for optimization
- Monitor coverage trends

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure 80%+ coverage for new code
3. Add appropriate markers (`unit`, `integration`, `slow`)
4. Update this README if needed
5. Run full test suite before submitting

## Troubleshooting

### Common Issues
1. **Import Errors**: Check Python path and package structure
2. **Mock Errors**: Verify patch targets and import paths
3. **Coverage Issues**: Ensure test files are in correct location
4. **Fixture Errors**: Check fixture scope and dependencies

### Getting Help
- Check existing test patterns
- Review LangChain documentation for mocking
- Consult pytest documentation
- Run tests with maximum verbosity for debugging