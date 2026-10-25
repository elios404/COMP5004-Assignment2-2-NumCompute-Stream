# COMP5004-Assignment2-NumCompute

A modular, vectorised scientific computing and machine learning toolkit implemented from scratch using plain Python and NumPy.

## Project Structure

```text
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py              # CSV reader (streaming/chunking) and missing value handling
│   ├── preprocessing.py   # StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
│   ├── pipeline.py        # BaseTransformer, BaseEstimator, Pipeline, FeatureUnion
│   ├── sort_search.py     # Sorts, Top-k, quickselect, binary_search
│   ├── rank.py            # Ranking with tie-breaking and percentiles
│   ├── stats.py           # Descriptive stats, Welford streaming stats, histogram, quantiles
│   ├── optim.py           # Finite-difference gradients and Jacobians
│   ├── metrics.py         # Classification & regression metrics, Confusion Matrix, ROC/AUC
│   ├── utils.py           # Pairwise distances, stable softmax/logsumexp, batch_generator
│   └── benchmarking.py    # Micro-benchmark harness and loop vs. NumPy comparisons
├── tests/                 # Unit test suite covering core functionality and edge cases
├── demo/
│   └── quickstart.ipynb   # End-to-end demonstration notebook
├── benchmark/
│   ├── run_benchmarks.py  # Reproducible benchmark script
│   ├── results.csv        # Recorded benchmark timing outputs
│   └── environment.txt    # Execution environment metadata
├── pyproject.toml         # Package configuration
└── README.md

```

## Requirements

* Python >= 3.11
* NumPy
* pytest (for running unit tests)

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/elios404/COMP5004-Assignment2-NumCompute.git](https://github.com/elios404/COMP5004-Assignment2-NumCompute.git)
cd COMP5004-Assignment2-NumCompute

```

### 2. Create and Activate Virtual Environment

* **macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate

```


* **Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

```



### 3. Install Package

Install `numcompute` in editable mode with development dependencies:

```bash
pip install --upgrade pip
pip install -e ".[dev]"

```

---

## API Overview

| Module | Key Functions / Classes | Purpose |
| --- | --- | --- |
| `io.py` | `load_csv`, `load_csv_chunks` | Loads 2D numerical arrays from CSV files with missing-token parsing and chunk streaming. |
| `preprocessing.py` | `SimpleImputer`, `StandardScaler`, `MinMaxScaler`, `OneHotEncoder` | Imputes missing values, scales numerical features, and one-hot encodes categorical columns. |
| `pipeline.py` | `Pipeline`, `FeatureUnion`, `BaseTransformer`, `BaseEstimator` | Chains sequential transformations or merges parallel preprocessing steps under a unified `fit`/`transform` API. |
| `sort_search.py` | `stable_sort`, `multi_key_sort`, `topk`, `quickselect`, `binary_search` | Performs stable sorting, $k$-th element selection, top-$k$ extraction, and binary search. |
| `rank.py` | `rank`, `percentile` | Computes element ranks with configurable tie-breaking (`average`, `dense`, `ordinal`) and NaN-aware percentiles. |
| `stats.py` | `mean`, `std`, `median`, `minimum`, `maximum`, `histogram`, `quantiles`, `WelfordStats` | Computes axis-aware descriptive statistics and maintains online streaming statistics via Welford's algorithm. |
| `optim.py` | `grad`, `jacobian` | Estimates gradients for scalar functions and Jacobians for vector functions using forward or central finite differences. |
| `metrics.py` | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse`, `roc_curve`, `auc` | Evaluates binary/multiclass classification and regression predictions. |
| `utils.py` | `euclidean_distance`, `cosine_similarity`, `softmax`, `logsumexp`, `batch_generator` | Provides pairwise distance matrices, numerically stable activations, and mini-batch data iteration. |
| `benchmarking.py` | `benchmark`, `compare_implementations` | Measures execution times and computes speedup ratios between loop-based and vectorised implementations. |

---

## Usage Examples

### 1. Data Loading & Preprocessing Pipeline

```python
import numpy as np
from numcompute.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder
from numcompute.pipeline import Pipeline

# Sample numerical data with missing values
X_num = np.array([[10.0, np.nan], [20.0, 4.0], [30.0, 6.0]])

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
X_num_scaled = num_pipe.fit_transform(X_num)

# Categorical encoding
X_cat = np.array([[0], [1], [2]])
encoder = OneHotEncoder(drop_first=True)
X_cat_encoded = encoder.fit_transform(X_cat)

```

### 2. Descriptive & Streaming Statistics

```python
import numpy as np
from numcompute.stats import mean, std, quantiles, WelfordStats

data = np.array([2.0, 4.0, 6.0, 8.0])
print("Mean:", mean(data))
print("Std:", std(data))
print("Quantiles:", quantiles(data, [0.25, 0.5, 0.75]))

# Streaming updates with Welford's algorithm
stream = WelfordStats()
stream.update([2.0, 4.0])
stream.update([6.0, 8.0])
print("Streaming Mean:", stream.mean_)
print("Streaming Variance:", stream.variance)

```

### 3. Sorting, Ranking & Numerical Optimization

```python
import numpy as np
from numcompute.sort_search import topk, binary_search, stable_sort
from numcompute.rank import rank
from numcompute.optim import grad, jacobian

arr = np.array([10.0, 30.0, 20.0, 30.0])
vals, idx = topk(arr, k=2, largest=True)
ranks = rank(arr, method="average")

# Finite-difference Gradient & Jacobian
f = lambda x: x[0]**2 + 3 * x[0] * x[1]
F = lambda x: np.array([x[0]**2 + x[1], x[0] * x[1]])

x0 = np.array([2.0, 3.0])
print("Gradient:", grad(f, x0, method="central"))
print("Jacobian:\n", jacobian(F, x0, method="central"))

```

### 4. Metrics & Numerical Utilities

```python
import numpy as np
from numcompute.metrics import accuracy, f1, mse
from numcompute.utils import euclidean_distance, softmax

y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0])
print("Accuracy:", accuracy(y_true, y_pred))
print("F1 Score:", f1(y_true, y_pred))

scores = np.array([1000.0, 1001.0, 1002.0])
print("Stable Softmax:", softmax(scores))

```

---

## Testing & Demonstration

### Run Unit Tests

Execute the full automated test suite from the project root:

```bash
pytest -v

```

### Run End-to-End Demo

Open and run `demo/quickstart.ipynb` in Jupyter Notebook or VS Code to view the complete workflow from CSV generation to benchmarking:

```bash
jupyter notebook demo/quickstart.ipynb

```

### Run Reproducible Benchmarks

Execute the standalone benchmark script to regenerate `benchmark/results.csv` and `benchmark/environment.txt`:

```bash
python benchmark/run_benchmarks.py

```

---

## Benchmark Performance (Python Loops vs. NumPy)

Measured using `benchmark/run_benchmarks.py` with a fixed random seed (`42`), averaged over 5 repetitions of 10 calls per operation.

| Operation | Input Size | Loop (ms) | NumPy (ms) | Speedup |
| --- | --- | --- | --- | --- |
| **MSE** | 1,000 | 0.3035 | 0.0173 | 17.56x |
| **MSE** | 10,000 | 5.2451 | 0.0682 | 76.92x |
| **MSE** | 100,000 | 27.3947 | 3.0458 | 8.99x |
| **Euclidean Distance** | 20 × 20 (8 features) | 2.0718 | 0.0717 | 28.91x |
| **Euclidean Distance** | 50 × 50 (8 features) | 8.4105 | 0.2541 | 33.10x |
| **Euclidean Distance** | 100 × 100 (8 features) | 35.9366 | 0.9685 | 37.11x |
| **Softmax** | 1,000 | 0.3847 | 0.0219 | 17.54x |
| **Softmax** | 10,000 | 3.0728 | 0.0926 | 33.20x |
| **Softmax** | 100,000 | 32.5353 | 1.5731 | 20.68x |

*Execution environment: Python 3.14.7; NumPy 2.5.4; Windows 11 (10.0.26200); AMD64.*
