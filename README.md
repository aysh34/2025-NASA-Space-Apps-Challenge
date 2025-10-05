# Exoplanet Detection - Local Workspace Instructions

This repository contains an improved 1D CNN model for exoplanet detection.

Key changes made to support local runs (instead of Colab):

- All absolute Colab paths (e.g. `/content/...`) were replaced with relative
  directories inside the project: `data/`, `models/`, `results/`, and
  `results/plots/`.
- `main.py` now resolves these relative paths to absolute paths based on the
  script location and creates the directories at runtime.

Quick start (PowerShell):

```powershell
# (1) Create a virtual environment (recommended)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# (2) Install dependencies
pip install -r requirements.txt

# (3) Run the script
python main.py
```

After running, you'll find:

- models/: saved scaler, best model, and final_model.h5
- results/: metrics.json and report.txt
- results/plots/: results.png
