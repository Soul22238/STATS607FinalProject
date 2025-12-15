# FastGLM

Fast GLM solver optimized for small-to-medium datasets using IRLS with Cholesky decomposition.

## DEMO - Run Everything

**Quick Start - Run complete demo (environment check + features + benchmarks + tests):**

```bash
python demo.py
```

This single command will:
1. Check Python environment and dependencies
2. Demonstrate all 5 GLM families with quick tests
3. Run full benchmark suite (5-10 minutes)
4. Execute all 37 unit tests

**Quick demo only (skip benchmarks):**
Just press 'n' when prompted, or run individual components:
- Benchmarks only: `python benchmark.py`
- Tests only: `pytest test.py -v`

## Project Structure

```
STATS607FinalProject/
├── fastglm.py           # Core GLM implementation (650 lines)
├── benchmark.py         # Performance benchmarking suite (1320 lines)
├── test.py              # Unit tests with pytest (37 tests)
├── demo.py              # Complete demo runner (run all)
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── results/
    ├── csv/             # Benchmark results in CSV format
    └── plots/           # Performance visualization plots
```

## File Descriptions

**fastglm.py**
- Core implementation of FastGLM class
- Supports 5 GLM families: logistic, poisson, gamma, gaussian, inverse_gaussian
- Uses IRLS with Cholesky decomposition
- Optimized for small datasets (n < 10,000, p < 50)

**benchmark.py**
- Comprehensive performance benchmarking
- Compares FastGLM vs sklearn/statsmodels
- Tests across different data sizes and scenarios
- Generates CSV results and visualization plots

**test.py**
- 37 unit tests covering all functionality
- Tests all 5 GLM families
- Validates input handling and edge cases
- Run with pytest

**demo.py**
- Complete workflow automation
- Checks environment and dependencies
- Runs feature demo, benchmarks, and tests
- Interactive - can skip benchmarks if desired

## Setup

### 1. Create Virtual Environment

```bash
cd STATS607FinalProject
python -m venv .venv
```

### 2. Activate Virtual Environment

On macOS/Linux:
```bash
source .venv/bin/activate
```

On Windows:
```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy==2.3.5
- scipy==1.16.3
- scikit-learn==1.7.2
- statsmodels==0.14.5
- matplotlib==3.10.7
- seaborn==0.13.2
- pandas==2.3.3
- pytest==9.0.2

## Usage Examples

### Example 1: Logistic Regression

```python
from fastglm import FastGLM
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(500, 10)
y = (np.random.rand(500) < 0.5).astype(float)

# Fit model
model = FastGLM(family='logistic')
model.fit(X, y)

# Predictions
predictions = model.predict(X)
accuracy = model.score(X, y)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Iterations: {model.n_iter_}")
print(f"Accuracy: {accuracy:.3f}")
```

### Example 2: Poisson Regression (Count Data)

```python
from fastglm import FastGLM
import numpy as np

# Generate count data
np.random.seed(42)
X = np.random.randn(300, 8)
eta = X @ np.random.randn(8)
y = np.random.poisson(np.exp(eta))

# Fit model
model = FastGLM(family='poisson', max_iter=50)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
r2 = model.score(X, y)

print(f"R2 score: {r2:.3f}")
```

### Example 3: Gamma Regression (Insurance Claims)

```python
from fastglm import FastGLM
import numpy as np

# Generate positive continuous data
np.random.seed(42)
X = np.random.randn(400, 6)
mu = np.exp(X @ np.random.randn(6) * 0.5)
y = np.random.gamma(2, mu/2)

# Fit model
model = FastGLM(family='gamma')
model.fit(X, y)

predictions = model.predict(X)
```

### Example 4: Gaussian Regression (Linear Regression)

```python
from fastglm import FastGLM
import numpy as np

# Generate linear data
X = np.random.randn(200, 5)
y = X @ np.array([1.5, -2.0, 0.5, 0.8, -1.2]) + np.random.randn(200) * 0.5

# Fit model
model = FastGLM(family='gaussian')
model.fit(X, y)

predictions = model.predict(X)
r2 = model.score(X, y)
```

### Example 5: All Model Parameters

```python
model = FastGLM(
    family='logistic',      # GLM family
    fit_intercept=True,     # Include intercept
    max_iter=25,            # Max IRLS iterations
    tol=1e-8,               # Convergence tolerance
    verbose=False,          # Print iteration info
    check_input=True        # Validate input data
)
```

## Running Benchmarks

Run the complete benchmark suite:

```bash
python benchmark.py
```

This will:
- Test all 5 GLM families
- Compare FastGLM vs sklearn/statsmodels
- Test multiple data sizes: [10, 50, 100, 500, 1000, 5000, 10000, 100000]
- Generate results in `results/csv/`
- Create plots in `results/plots/`

Expected runtime: 5-10 minutes

Output files:
```
results/csv/benchmark_logistic.csv
results/csv/benchmark_poisson.csv
results/csv/benchmark_gamma.csv
results/csv/benchmark_gaussian.csv
results/csv/benchmark_inverse_gaussian.csv
results/csv/benchmark_stability.csv
results/csv/benchmark_limitations.csv
results/csv/benchmark_weakness.csv
results/csv/benchmark_accuracy.csv
results/csv/benchmark_all_families.csv

results/plots/01_summary.png
results/plots/02_analysis_bars.png
results/plots/03_family_performance.png
results/plots/04_limitation_scenarios.png
```

## Running Tests

Run all 37 unit tests:

```bash
pytest test.py -v
```

Run tests quietly:

```bash
pytest test.py -q
```

Run specific test class:

```bash
pytest test.py::TestLogisticRegression -v
```

Expected output:
```
37 passed, 1 warning in 1.02s
```

## API Reference

### FastGLM Class

**Methods:**
- `fit(X, y)` - Fit the model
- `predict(X)` - Generate predictions
- `score(X, y)` - Compute accuracy (logistic) or R2 (others)
- `predict_proba(X)` - Predict probabilities (logistic only)

**Attributes:**
- `coef_` - Fitted coefficients (array)
- `intercept_` - Intercept term (float)
- `n_iter_` - Number of iterations to convergence
- `family` - GLM family name

## Performance Notes

FastGLM is faster for:
- Gamma regression: ~8x faster than statsmodels
- Poisson regression: ~5x faster than statsmodels
- Inverse Gaussian: ~5x faster than statsmodels

sklearn/statsmodels are faster for:
- Very large datasets (n > 50,000)
- High-dimensional data (p > 100)
- When regularization is needed (L1/L2)
- Sparse matrices

## Deactivate Virtual Environment

When finished:

```bash
deactivate
```

## Requirements

- Python 3.8 or higher
- NumPy 2.3+
- SciPy 1.16+
- scikit-learn 1.7+
- statsmodels 0.14+
