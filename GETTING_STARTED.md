# Getting Started with StackGP Web App

This guide walks you through setting up and running the StackGP browser-based
evolution interface.

---

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.9 |
| pip | 21 |

No browser extensions or plugins are required. All front-end assets (Bootstrap,
Chart.js, Bootstrap Icons) are vendored inside `frontend/vendor/` so the app
works fully offline.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hoolagans/StackGP.git
cd StackGP
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Starting the server

```bash
python api.py
```

You should see output similar to:

```
 * Running on http://0.0.0.0:5000
 * Running on http://127.0.0.1:5000
```

The server runs on **port 5000** by default and is accessible from any network
interface. To restrict it to localhost only, edit the last line of `api.py`:

```python
app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
```

---

## Accessing the UI

Open your browser and navigate to:

```
http://localhost:5000
```

---

## Quick-start walkthrough

### Step 1 — Upload a dataset

1. In the **Dataset** panel, click **Choose File** and select a CSV file.
   - The first row must be a header row (column names).
   - All columns must be numeric.
   - Headerless CSVs are also accepted; columns will be auto-named `x0`, `x1`, …
2. Click **Upload**.
3. Choose the **Response column** (the variable you want StackGP to model).
4. Optionally click the table icon (👁) to preview the first five rows.

Previously uploaded datasets persist on disk and can be reloaded from the
**or load existing** dropdown at any time.

### Step 2 — Configure evolution settings

Use the **Evolution Settings** panel to tune the run. Sensible defaults are
already filled in. Key parameters:

| Setting | Description |
|---|---|
| Evolution method | `evolve` (single-process) or `parallelEvolve` (multi-core) |
| Generations | Total number of evolution generations |
| Population size | Number of candidate programs per generation |
| Mutation / Crossover / Spawn | Must sum to 100 % |
| Test split % | Hold-out percentage for out-of-sample evaluation (0 = none) |
| Seed population | Optionally warm-start from a previously saved model set |

**Settings presets** let you save and restore named configurations using the
controls on the right side of the Settings panel.

### Step 3 — Run evolution

Click **Run Evolution**. The **Live Tracking** chart updates in real time,
showing the best fitness (1 − R²) per generation. You can cancel a run at any
time with the **Cancel** button.

### Step 4 — Inspect results

When evolution completes, the **Results & Model Sets** panel shows the
Pareto-front models ranked by fitness. Click any row to jump to that model in
the **Model Explorer**.

- **Save current models as** — type a name and click **Save** to persist the
  Pareto front to disk.
- **Export CSV** — download the summary table as a CSV file.

### Step 5 — Explore and plot

Use the **Model Explorer** panel to visualise the current (or any saved) model
set:

- Choose a **Model set** source from the dropdown (current run or any saved set,
  including combined sets).
- Pick a **Plot type** from population-level or per-model options.
- For per-model plots, select the **Model #** (or click a row in the Results
  table to auto-select it).
- Click **Generate**.

### Step 6 — Combine model sets (optional)

Open the **Combine Saved Model Sets** section inside the Results card to merge
two or more saved sets into one. The combined set is saved to disk and
immediately available as a seed source or in the Model Explorer.

### Step 7 — Use saved models as seeds

In the **Evolution Settings** panel, select a saved model set from the
**Seed population** dropdown. The chosen set will be used to warm-start the next
evolution run, letting you continue or refine a previous search.

---

## Persistent storage

| Directory | Contents |
|---|---|
| `saved_datasets/` | Uploaded CSV files |
| `saved_models/` | Saved model sets (dill-serialised) |

Both directories are created automatically on first run and are preserved across
server restarts.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Browser shows "Cannot connect" | Server not running | Run `python api.py` |
| Upload fails with HTTP 400 | Filename contains special characters | Rename the file using only letters, numbers, hyphens, and underscores |
| Evolution is slow | Large dataset or high population | Reduce **Population size** or switch to `parallelEvolve` with **n_jobs = -1** |
| Plot returns an error | Per-model plot selected but no dataset loaded | Load the dataset that was used during evolution |
| "Models not available" when reloading history | Server was restarted; in-memory jobs were lost | Save model sets to disk immediately after a run completes |
