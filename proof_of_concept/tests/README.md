# STAGED Model Tests

This directory contains unit and integration tests for the STAGED model implementation.

## Test Organization

- **test_graph_constructor.py**: Tests for the graph construction utilities
- **test_model.py**: Tests for the STAGED neural network model
- **test_data_utils.py**: Tests for data loading and preprocessing utilities
- **test_integration.py**: Integration tests that ensure all components work together
- **run_tests.py**: Script to run all tests

## Running Tests

To run all tests:

```bash
python run_tests.py
```

To run tests with more verbose output:

```bash
python run_tests.py --verbose
```

To run specific test files:

```bash
python run_tests.py --pattern "test_model.py"
```

## Test Coverage

The tests cover the following functionality:

1. **Graph Construction**
   - Initialization of the GraphConstructor
   - Base graph construction with gene, receptor, and ligand nodes
   - Graph updates with neighboring cells' information
   - Node feature assignment with appropriate time lags

2. **STAGED Model**
   - Model initialization with correct dimensions
   - Forward pass through the model
   - Time initialization calculation

3. **Data Utilities**
   - Loading and generating synthetic data
   - Data structure validation
   - Data preprocessing and normalization

4. **Integration Tests**
   - End-to-end training process
   - Model predictions validation
   - CUDA support (when available)

## Adding New Tests

To add new tests:

1. Create a new file with the name pattern `test_*.py`
2. Subclass `unittest.TestCase`
3. Add test methods starting with `test_`
4. Run using the `run_tests.py` script

Example:

```python
import unittest

class TestMyComponent(unittest.TestCase):
    def test_my_functionality(self):
        # Test code here
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main() 