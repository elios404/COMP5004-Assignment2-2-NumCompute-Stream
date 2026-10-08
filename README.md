# COMP5004-Assignment2-NumCompute

A modular scientific computing toolkit implemented in plain Python and NumPy.

## Requirements
* Python 3.11


## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/elios404/COMP5004-Assignment2-NumCompute.git
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

### 4. Run Tests
Verify the environment setup:
```bash
pytest
```