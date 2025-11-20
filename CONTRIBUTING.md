# Contributing to Heartbeat Classification Project

Thank you for your interest in contributing to the Heartbeat Classification project! This document provides guidelines and instructions for contributing.

## Contributors

- **Julia Schmidt**
- **Christian Meister**
- **Tzu-Jung Huang**

## Code Style Guidelines

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines with some modifications:

- **Line length**: Maximum 100 characters (enforced by Black)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings
- **Imports**: Organize imports in the following order:
  1. Standard library imports
  2. Third-party imports
  3. Local application/library imports
- **Naming conventions**:
  - Functions and variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

### Code Formatting

We use [Black](https://black.readthedocs.io/) for automatic code formatting:

```bash
# Format all Python files
black src/ tests/

# Check formatting without making changes
black --check src/ tests/
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
# Run flake8
flake8 src/ tests/
```

### Type Hints

We encourage the use of type hints for better code documentation and IDE support:

```python
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

def preprocess_data(
    data: pd.DataFrame,
    target_column: str,
    normalize: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the input data.
    
    Args:
        data: Input DataFrame
        target_column: Name of the target column
        normalize: Whether to normalize the data
        
    Returns:
        Tuple of (features, target)
    """
    # Implementation
    pass
```

## How to Add Notebooks

### Notebook Organization

1. **Naming Convention**: Use descriptive names with numbers for ordering:
   - `01_data_exploration.ipynb`
   - `02_preprocessing.ipynb`
   - `03_A_02_01_baseline_models_randomized_search.ipynb`

2. **Notebook Structure**:
   - Start with a markdown cell describing the notebook's purpose
   - Include clear section headers
   - Add comments and explanations in markdown cells
   - Keep code cells focused and readable

3. **Best Practices**:
   - **Clear outputs**: Clear all outputs before committing (use `jupyter nbconvert --clear-output`)
   - **Modular code**: Import functions from `src/` modules rather than duplicating code
   - **Documentation**: Add markdown cells explaining the methodology and results
   - **Reproducibility**: Set random seeds for reproducible results
   - **Data paths**: Use relative paths from the project root

4. **Example Notebook Template**:
   ```python
   # Cell 1: Markdown
   # # Title
   # Description of what this notebook does
   
   # Cell 2: Imports
   import numpy as np
   import pandas as pd
   from src.utils.preprocessing import preprocess_data
   from src.utils.evaluation import evaluate_model
   
   # Cell 3: Configuration
   RANDOM_SEED = 42
   DATA_PATH = "data/interim/"
   
   # Cell 4: Load Data
   # ... data loading code
   
   # Cell 5: Analysis/Modeling
   # ... main code
   
   # Cell 6: Results
   # ... visualization and results
   ```

5. **Before Committing**:
   ```bash
   # Clear outputs from notebooks
   jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
   
   # Or use nbstripout (if installed)
   nbstripout notebooks/*.ipynb
   ```

### Notebook Categories

- **Data Exploration** (`01_*`): EDA, data quality analysis
- **Preprocessing** (`02_*`): Data cleaning, feature engineering
- **Baseline Models** (`03_*`): Traditional ML models
- **Deep Learning** (`04_*`): Neural network models
- **Interpretability** (`05_*`): SHAP, model explanation
- **Archive** (`archive/`): Old or experimental notebooks

## Testing Requirements

### Test Structure

- Place all tests in the `tests/` directory
- Mirror the `src/` directory structure in `tests/`
- Use descriptive test names: `test_function_name_scenario()`

### Writing Tests

```python
# tests/utils/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.utils.preprocessing import preprocess_data

def test_preprocess_data_basic():
    """Test basic preprocessing functionality."""
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
    
    X, y = preprocess_data(data, target_column='target')
    
    assert X.shape[0] == 5
    assert y.shape[0] == 5
    assert 'target' not in X.columns

def test_preprocess_data_normalization():
    """Test normalization option."""
    # Test implementation
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/utils/test_preprocessing.py

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "preprocessing"
```

### Test Coverage

- Aim for at least 80% code coverage
- Focus on testing core functionality in `src/utils/` and `src/visualization/`
- Test edge cases and error handling

## Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/heartbeat_classification.git
cd heartbeat_classification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 5. Make Changes

- Write clean, documented code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation if needed

### 6. Test Your Changes

```bash
# Run tests
pytest

# Check code style
black --check src/ tests/
flake8 src/ tests/
```

### 7. Commit Your Changes

```bash
git add .
git commit -m "Description of your changes"
```

**Commit Message Guidelines**:
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 72 characters
- Add more details in the body if needed

### 8. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to related issues (if any)
- Screenshots or examples (if applicable)

## Code Review Process

1. All pull requests require at least one approval
2. Code must pass all tests and linting checks
3. Maintainers will review for:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Performance considerations

## Questions?

If you have questions about contributing, please:
- Open an issue on GitHub
- Contact one of the project maintainers
- Check the README.md for project overview

Thank you for contributing to the Heartbeat Classification project!

