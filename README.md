# COMP5004 Assignment 2.2 — NumCompute-Stream

NumCompute-Stream extends the Assignment 2.1 NumCompute toolkit with streaming preprocessing, statistics, decision-tree classification and evaluation. It uses NumPy for computation and Matplotlib for visualisation.

## Project Structure

```text
COMP5004-Assignment2-2-NumCompute-Stream/
├── numcompute/                   # Assignment 2.1 package
├── numcompute_stream/            # Assignment 2.2 package
│   ├── __init__.py               # Package initialisation
│   ├── _moments.py               # Shared calculations for running means and variances
│   ├── preprocessing.py          # Incremental scaling, imputation and one-hot encoding
│   ├── stats.py                  # Per-feature streaming statistics
│   ├── tree.py                   # Decision tree and buffered retraining
│   ├── pipeline.py               # Streaming preprocessing and model pipeline
│   ├── metrics.py                # Cumulative and rolling classification metrics
│   ├── stream.py                 # Chunk evaluation, training and logging
│   └── visualise.py              # Metric, model comparison and prediction plots
├── tests/                        # Assignment 2.1 and Assignment 2.2 unit tests
├── demo/
│   ├── stream_demo.ipynb         # Assignment 2.2 demonstration
│   └── quickstart.ipynb          # Assignment 2.1 demonstration
├── benchmark/
│   ├── run_stream_benchmarks.py  # Assignment 2.2 benchmark script
│   ├── stream_split_results.csv  # Assignment 2.2 split-search results
│   ├── stream_buffer_results.csv # Assignment 2.2 buffer-size results
│   ├── stream_environment.txt   # Assignment 2.2 benchmark environment
│   ├── run_benchmarks.py         # Assignment 2.1 benchmark script
│   ├── results.csv              # Assignment 2.1 benchmark results
│   └── environment.txt          # Assignment 2.1 benchmark environment
├── pyproject.toml                # Package configuration
└── README.md
```

## Requirements

- Python 3.10 or later
- NumPy and Matplotlib
- pytest and Jupyter for tests and the demo

## Setup & Installation

### 1. Clone the Repository

```bash
git clone --branch sejun-stream https://github.com/elios404/COMP5004-Assignment2-2-NumCompute-Stream.git
cd COMP5004-Assignment2-2-NumCompute-Stream
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install the Package

Install `numcompute` in editable mode with development dependencies.

```bash
python -m pip install -e ".[dev]"
```

## API Overview

| Module in `numcompute_stream` | Main API | Purpose |
| --- | --- | --- |
| `preprocessing.py` | `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder` | Learn preprocessing state from chunks with `partial_fit()` and apply it with `transform()`. |
| `stats.py` | `StreamingStats` | Update per-feature counts, means and variances. Supports fixed-bin histograms and exact quantiles with retained history. |
| `tree.py` | `DecisionTreeClassifier` | Train with `fit()` or retrain when the `partial_fit()` buffer fills. Predict labels or class probabilities. |
| `pipeline.py` | `Pipeline` | Chain preprocessing and classification, and update the fitted pipeline when enough rows are buffered. |
| `metrics.py` | `StreamingMetrics` | Accumulate classification metrics and optionally report them over a rolling window. |
| `stream.py` | `StreamTrainer` | Evaluate each chunk with the current fitted model before updating the pipeline. Record scores, retraining, update time and peak traced memory. |
| `visualise.py` | `plot_metric_over_time()`, `compare_models()`, `plot_predictions_vs_ground_truth()` | Plot chunk metrics, compare runs and show predictions against true labels. |

## Testing & Demonstration

Run the full test suite from the repository root.

```bash
python -m pytest -v
```

To run the tests for the Assignment 2.2 modules, including tree and visualisation tests, use the following command.

```bash
python -m pytest -v tests/test_tree.py tests/test_stream_preprocessing.py tests/test_stream_stats.py tests/test_stream_pipeline.py tests/test_stream_metrics.py tests/test_stream_trainer.py tests/test_visualise.py
```

Open the Assignment 2.2 notebook from the repository root and run all cells using the project Python kernel.

```bash
jupyter notebook demo/stream_demo.ipynb
```

The notebook generates a reproducible CSV with missing feature values and a fixed label rule. It reads the CSV in 40-row chunks and feeds the same chunks to two pipelines with 80- and 160-row buffers. The output shows when each tree retrains, the evaluation and update logs, chunk accuracy for both buffers, and the latest chunk's predictions against its true labels.

## Streaming Benchmarks

Run the benchmark from the repository root to regenerate both results CSVs and the environment record in `benchmark/`.

```bash
python benchmark/run_stream_benchmarks.py
```

Both experiments use a synthetic binary stream of 1,280 rows with six features and seed 42. The decision boundary changes halfway through the stream. The recorded environment is macOS (ARM64), Python 3.11.15 and NumPy 2.4.6. See [stream_environment.txt](benchmark/stream_environment.txt) for the full settings.

### Gini Split Search (Python Loops vs NumPy)

The script verifies that the implementations agree on the selected feature, threshold and gain before timing. Both use NumPy for sorting. The loop implementation evaluates candidate splits one at a time, and the vectorised implementation evaluates them in arrays. Times are mean milliseconds per search at one tree node, measured over five repetitions of ten calls per window. Speedup is the loop time divided by the vectorised time.

| Rows per window | Windows | Python loop (ms) | Vectorised NumPy (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 16 | 0.559430 | 0.394883 | 1.417× |
| 160 | 8 | 0.677354 | 0.330304 | 2.051× |
| 320 | 4 | 1.352802 | 0.458624 | 2.950× |

The measured speedup increased with window size, reaching 2.950× at 320 rows. [Raw split-search results](benchmark/stream_split_results.csv)

### Buffer Size Comparison

Each pipeline receives 40-row chunks and uses a tree with maximum depth 4. Accuracy is measured on the same 960 rows for all three settings, covering chunks 9 to 32. Fit counts include the initial fit. Update time is the median across updates that fitted a tree over five stream runs. It includes preprocessing and buffering with memory tracing enabled. Evaluation runs before the timed update.

| Buffer (rows) | First evaluated chunk | Fits per run | Common evaluated rows | Accuracy | Median fit update (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 3 | 16 | 960 | 87.29% | 10.530271 |
| 160 | 5 | 8 | 960 | 87.50% | 14.934229 |
| 320 | 9 | 4 | 960 | 81.67% | 18.510812 |

Larger buffers delayed the first evaluation, reduced the number of fits and increased the median fit update time. The 80- and 160-row buffers differed by two correct predictions over the same 960 rows. The 320-row buffer had the lowest accuracy on this stream. [Raw buffer-size results](benchmark/stream_buffer_results.csv)
