# StackGP Data Modeling Studio

A professional-grade GUI for symbolic regression data science workflows using StackGP.  
Built with **Vite + React + TypeScript** (frontend) and **FastAPI + Python** (backend).

---

## Features

| Page | Capabilities |
|------|-------------|
| **Import** | Upload CSV / Excel files; configure target & feature columns; preview data with sorting |
| **Process** | Missing-value handling (drop / mean / median / zero); Min-Max or Z-score normalization; configurable train/test split |
| **Explore** | Descriptive statistics table; distribution histograms; correlation heatmap; scatter plots; bar chart of feature-target correlations |
| **Model** | Full StackGP `evolve()` configuration (generations, population, operators, mutation/crossover/spawn rates, time limit, alignment, subsampling, early termination); live training progress bar + log; training chart |
| **Analysis** | Pareto front scatter (accuracy vs complexity); sortable model table; symbolic expression viewer; predicted vs actual; residual plot; variable importance; ensemble builder with uncertainty quantification |
| **Predict** | Single-point prediction form; batch CSV/tab input; ensemble predictions with ± uncertainty; max-uncertainty point finder (active learning); downloadable results CSV |

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18

### 1 · Install backend dependencies

```bash
cd gui/backend
pip install -r requirements.txt
```

### 2 · Install frontend dependencies

```bash
cd gui/frontend
npm install
```

### 3 · Start development servers

```bash
# From the repo root:
bash gui/start-dev.sh
```

Then open **http://localhost:5173** in your browser.

The Vite dev server proxies all `/api/*` requests to the FastAPI backend at `http://localhost:8000`.

### API docs

FastAPI auto-generates interactive docs at **http://localhost:8000/docs**.

---

## Architecture

```
gui/
├── backend/
│   ├── main.py          # FastAPI app — all REST endpoints
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/client.ts        # Typed Axios API client
│   │   ├── components/
│   │   │   ├── ui.tsx           # Shared UI primitives (Card, Button, …)
│   │   │   ├── DataTable.tsx    # Sortable / paginated data table
│   │   │   └── Layout.tsx       # Sidebar navigation
│   │   └── pages/
│   │       ├── ImportPage.tsx
│   │       ├── ProcessPage.tsx
│   │       ├── ExplorePage.tsx
│   │       ├── ModelPage.tsx
│   │       ├── AnalysisPage.tsx
│   │       └── PredictPage.tsx
│   ├── tailwind.config.js
│   └── vite.config.ts
├── start-dev.sh    # Launch both servers (development)
└── start-prod.sh   # Build frontend + launch backend only
```

---

## Typical Workflow

1. **Import** — upload a CSV, select your target `Y` and feature columns `X`
2. **Process** — handle missing values, optionally normalize, set the train/test split
3. **Explore** — examine distributions, correlations, and relationships between variables
4. **Model** — configure and run StackGP symbolic regression evolution
5. **Analysis** — inspect the Pareto front of discovered models, check residuals and variable importance, build a model ensemble
6. **Predict** — enter new data points to get predictions and uncertainty estimates; use the active learning tool to find the most informative next experiment
