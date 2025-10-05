# ExoNova AI — Exploring exoplanets with AI

One‑sentence pitch: Detect exoplanets from transit signals using a high‑accuracy 1D CNN trained on NASA Kepler/K2 open data.

Note: Uses NASA open data. Not affiliated with or endorsed by NASA.

---

## Contents
- [Problem](#problem)
- [Solution](#solution)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [NASA Data and APIs](#nasa-data-and-apis)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Acknowledgments and License](#acknowledgments-and-license)

---

## Problem
- Challenge: Exoplanet Detection Challenge 2025
- Who: Astronomers, researchers, citizen scientists
- Why: Prioritize telescope time and study potentially habitable planets
- Success criteria: Accuracy ≥90%; precision/recall ≥85%; low‑latency predictions

---

## Solution
- Approach: 1D CNN with residual blocks and attention on realistic transit simulations
- Novelty: Realistic false‑positive modeling + high‑performance CNN + Streamlit UI
- MVP: Local training pipeline and interactive inference app
- Dependencies: Python 3.11+, TensorFlow ≥2.13, scikit‑learn ≥1.2, optional Docker

---

## Key Features
- Ingest simulated or NASA Kepler/K2 light curves
- Robust preprocessing and outlier handling
- 1D CNN with attention for transit classification
- Streamlit UI with probability and signal visualizations
- Export metrics/reports (JSON, PNG) and run reproducibly

---

## Architecture
- Data: CSV/light curve arrays (Kepler/K2, simulated)
- Processing: pandas, numpy, scikit‑learn
- Model: TensorFlow/Keras 1D CNN
- UI: Streamlit
- Storage: data/, models/, results/
- Deployment: Local or Docker

---

## NASA Data and APIs
- Kepler/K2 mission data: https://archive.stsci.edu/kepler/
- Kepler Object of Interest (KOI) catalog: https://exoplanetarchive.ipac.caltech.edu/

---

## Quick Start

Windows (PowerShell):
```powershell
# 1) Clone and create venv
git clone <repo-url>
cd <repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Train and evaluate
python main.py

# 4) Launch UI
streamlit run app.py
```

macOS/Linux (bash):
```bash
# 1) Clone and create venv
git clone <repo-url>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Train and evaluate
python main.py

# 4) Launch UI
streamlit run app.py
```

---

## Project Structure
```
.
├─ app.py               # Streamlit UI
├─ exoplanet_detection.py # Train/evaluate 
predict_exoplanet.py
├─ data/                # Raw/processed data
├─ models/              # Saved models
├─ results/             # Metrics/reports/figures
└─ requirements.txt
```

---

## Responsible AI, Privacy, Accessibility
- Documented limitations and uncertainty estimates
- Reproducible training/evaluation with fixed seeds
- No personal data; only public scientific datasets
- Accessible UI: keyboard navigation and alt text for charts

---

## Acknowledgments and License
- Data courtesy of NASA missions and archives listed above
- Not affiliated with or endorsed by NASA
- License: see LICENSE file