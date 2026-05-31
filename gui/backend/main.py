#!/usr/bin/env python3
"""
StackGP Data Modeling GUI - FastAPI Backend
"""

import sys
import os
import json
import uuid
import threading
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import io

import StackGP as sgp

app = FastAPI(title="StackGP GUI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store (single-user tool)
# ---------------------------------------------------------------------------
session: Dict[str, Any] = {
    "raw_df": None,
    "processed_df": None,
    "target_col": None,
    "feature_cols": [],
    "train_indices": [],
    "test_indices": [],
    "models": [],
    "ensemble": [],
    "training_log": [],
    "training_status": "idle",  # idle | running | done | error
    "training_progress": 0,
    "training_total": 0,
    "training_thread": None,
    "training_lock": threading.Lock(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_records(df: pd.DataFrame, max_rows: int = 200) -> dict:
    sample = df.head(max_rows)
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "rows": sample.where(pd.notna(sample), None).values.tolist(),
        "total_rows": len(df),
        "shape": list(df.shape),
    }


def get_xy(df: pd.DataFrame, target: str, features: List[str]):
    x = df[features].values.T.astype(float)
    y = df[target].values.astype(float)
    return x, y


def model_to_dict(model, idx: int, inputData, response, varNames) -> dict:
    try:
        pred = sgp.evaluateGPModel(model, inputData)
        rmse_val = float(sgp.rmse(model, inputData, response))
        fit_val = float(sgp.fitness(model, inputData, response))
        complexity = int(sgp.stackGPModelComplexity(model))
        expr = sgp.printGPModel(model, varNames)
        residuals = (np.array(pred) - np.array(response)).tolist()
    except Exception as e:
        rmse_val = None
        fit_val = None
        complexity = None
        expr = "error evaluating expression"
        residuals = []
        pred = []

    metrics = model[2] if len(model) > 2 else []
    safe_metrics = []
    for m in metrics:
        try:
            safe_metrics.append(float(m))
        except Exception:
            safe_metrics.append(None)

    return {
        "id": idx,
        "expression": str(expr),
        "rmse": rmse_val,
        "fitness": fit_val,
        "complexity": complexity,
        "metrics": safe_metrics,
        "predictions": [float(v) if np.isfinite(float(v)) else None for v in (pred if hasattr(pred, '__len__') else [])],
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ColumnConfig(BaseModel):
    target_col: str
    feature_cols: List[str]


class ProcessConfig(BaseModel):
    fill_missing: str = "drop"  # drop | mean | median | zero
    normalize: str = "none"    # none | minmax | zscore
    train_split: float = 0.8
    random_seed: int = 42


class TrainConfig(BaseModel):
    generations: int = 100
    pop_size: int = 300
    max_complexity: int = 100
    max_length: int = 10
    mutation_rate: int = 79
    crossover_rate: int = 11
    spawn_rate: int = 10
    time_limit: int = 300
    ops_set: str = "default"   # default | all | boolean
    align: bool = True
    data_subsample: bool = False
    allow_early_termination: bool = False
    early_termination_threshold: float = 0.0


class PredictRequest(BaseModel):
    rows: List[List[float]]


class EnsembleRequest(BaseModel):
    num_clusters: int = 10


# ---------------------------------------------------------------------------
# Data Import
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    name = file.filename or ""
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    session["raw_df"] = df
    session["processed_df"] = None
    session["target_col"] = None
    session["feature_cols"] = []
    session["models"] = []
    session["ensemble"] = []
    session["training_log"] = []
    session["training_status"] = "idle"
    return df_to_records(df)


@app.get("/api/data/raw")
def get_raw_data():
    df = session["raw_df"]
    if df is None:
        raise HTTPException(404, "No data loaded")
    return df_to_records(df)


@app.post("/api/data/configure")
def configure_columns(cfg: ColumnConfig):
    df = session["raw_df"]
    if df is None:
        raise HTTPException(404, "No data loaded")
    if cfg.target_col not in df.columns:
        raise HTTPException(400, f"Column '{cfg.target_col}' not found")
    bad = [c for c in cfg.feature_cols if c not in df.columns]
    if bad:
        raise HTTPException(400, f"Columns not found: {bad}")
    session["target_col"] = cfg.target_col
    session["feature_cols"] = cfg.feature_cols
    return {"ok": True, "target_col": cfg.target_col, "feature_cols": cfg.feature_cols}


# ---------------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------------

@app.post("/api/data/process")
def process_data(cfg: ProcessConfig):
    df = session["raw_df"]
    if df is None:
        raise HTTPException(404, "No data loaded")
    target = session["target_col"]
    features = session["feature_cols"]
    if not target or not features:
        raise HTTPException(400, "Configure target and feature columns first")

    cols = features + [target]
    df2 = df[cols].copy()

    # Fill missing
    if cfg.fill_missing == "drop":
        df2 = df2.dropna()
    elif cfg.fill_missing == "mean":
        df2 = df2.fillna(df2.mean(numeric_only=True))
    elif cfg.fill_missing == "median":
        df2 = df2.fillna(df2.median(numeric_only=True))
    elif cfg.fill_missing == "zero":
        df2 = df2.fillna(0)

    # Normalize features (not target)
    if cfg.normalize == "minmax":
        for c in features:
            mn, mx = df2[c].min(), df2[c].max()
            df2[c] = (df2[c] - mn) / (mx - mn + 1e-12)
    elif cfg.normalize == "zscore":
        for c in features:
            mu, sd = df2[c].mean(), df2[c].std()
            df2[c] = (df2[c] - mu) / (sd + 1e-12)

    df2 = df2.reset_index(drop=True)

    # Train/test split
    np.random.seed(cfg.random_seed)
    n = len(df2)
    idx = np.random.permutation(n)
    split = int(n * cfg.train_split)
    train_idx = idx[:split].tolist()
    test_idx = idx[split:].tolist()

    session["processed_df"] = df2
    session["train_indices"] = train_idx
    session["test_indices"] = test_idx

    return {
        "ok": True,
        "total_rows": n,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "data": df_to_records(df2),
    }


@app.get("/api/data/processed")
def get_processed_data():
    df = session["processed_df"]
    if df is None:
        raise HTTPException(404, "No processed data. Run /api/data/process first.")
    return df_to_records(df)


# ---------------------------------------------------------------------------
# Data Exploration
# ---------------------------------------------------------------------------

@app.get("/api/explore/stats")
def get_stats():
    df = session["processed_df"] or session["raw_df"]
    if df is None:
        raise HTTPException(404, "No data loaded")
    numeric_df = df.select_dtypes(include=[np.number])
    desc = numeric_df.describe().T
    result = {}
    for col in desc.index:
        result[col] = {
            "count": float(desc.loc[col, "count"]),
            "mean": float(desc.loc[col, "mean"]),
            "std": float(desc.loc[col, "std"]),
            "min": float(desc.loc[col, "min"]),
            "q25": float(desc.loc[col, "25%"]),
            "median": float(desc.loc[col, "50%"]),
            "q75": float(desc.loc[col, "75%"]),
            "max": float(desc.loc[col, "max"]),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique()),
            "skewness": float(df[col].dropna().skew()),
            "kurtosis": float(df[col].dropna().kurtosis()),
        }
    return result


@app.get("/api/explore/correlation")
def get_correlation():
    df = session["processed_df"] or session["raw_df"]
    if df is None:
        raise HTTPException(404, "No data loaded")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    return {
        "columns": list(corr.columns),
        "matrix": corr.where(pd.notna(corr), None).values.tolist(),
    }


@app.get("/api/explore/column/{col}")
def get_column_data(col: str):
    df = session["processed_df"] or session["raw_df"]
    if df is None:
        raise HTTPException(404, "No data loaded")
    if col not in df.columns:
        raise HTTPException(404, f"Column {col} not found")
    series = df[col].dropna()
    return {
        "values": series.tolist(),
        "histogram": _compute_histogram(series),
    }


def _compute_histogram(series: pd.Series, bins: int = 30) -> dict:
    arr = series.dropna().values.astype(float)
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "bin_centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
    }


@app.get("/api/explore/scatter")
def get_scatter(x_col: str, y_col: str, max_points: int = 1000):
    df = session["processed_df"] or session["raw_df"]
    if df is None:
        raise HTTPException(404, "No data loaded")
    sub = df[[x_col, y_col]].dropna()
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)
    return {
        "x": sub[x_col].tolist(),
        "y": sub[y_col].tolist(),
    }


@app.get("/api/explore/pairplot")
def get_pairplot(max_points: int = 300):
    df = session["processed_df"] or session["raw_df"]
    if df is None:
        raise HTTPException(404, "No data loaded")
    cols = session["feature_cols"] or list(df.select_dtypes(include=[np.number]).columns)[:8]
    target = session["target_col"]
    if target and target not in cols:
        cols = cols + [target]
    result = {}
    sample = df[cols].dropna()
    if len(sample) > max_points:
        sample = sample.sample(max_points, random_state=42)
    for c in cols:
        result[c] = sample[c].tolist()
    return {"columns": cols, "data": result}


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def _ops_set(name: str):
    if name == "all":
        return sgp.allOps()
    if name == "boolean":
        return sgp.booleanOps()
    return sgp.defaultOps()


def _run_training(cfg: TrainConfig):
    df = session["processed_df"]
    target = session["target_col"]
    features = session["feature_cols"]
    train_idx = session["train_indices"]

    if df is None or not target or not features:
        session["training_status"] = "error"
        return

    train_df = df.iloc[train_idx] if train_idx else df
    x, y = get_xy(train_df, target, features)
    var_names = [str(n) for n in features]

    ops = _ops_set(cfg.ops_set)

    log = []
    session["training_log"] = log
    session["training_progress"] = 0
    session["training_total"] = cfg.generations
    session["training_status"] = "running"

    def track_cb(gen, models):
        session["training_progress"] = gen
        if models:
            best = models[0]
            try:
                fit = float(best[2][0]) if best[2] else None
                cplx = int(best[2][1]) if len(best[2]) > 1 else None
            except Exception:
                fit, cplx = None, None
            log.append({"gen": gen, "fitness": fit, "complexity": cplx})

    try:
        models = sgp.evolve(
            x, y,
            generations=cfg.generations,
            ops=ops,
            variableNames=var_names,
            mutationRate=cfg.mutation_rate,
            crossoverRate=cfg.crossover_rate,
            spawnRate=cfg.spawn_rate,
            popSize=cfg.pop_size,
            maxComplexity=cfg.max_complexity,
            maxLength=cfg.max_length,
            align=cfg.align,
            timeLimit=cfg.time_limit,
            dataSubsample=cfg.data_subsample,
            allowEarlyTermination=cfg.allow_early_termination,
            earlyTerminationThreshold=cfg.early_termination_threshold,
            tracking=True,
            liveTracking=False,
        )
        session["models"] = models
        session["training_status"] = "done"
        session["training_progress"] = cfg.generations
    except Exception as e:
        session["training_status"] = "error"
        session["training_log"].append({"error": "Training failed — check server logs"})


@app.post("/api/train/start")
def start_training(cfg: TrainConfig, background_tasks: BackgroundTasks):
    if session["training_status"] == "running":
        raise HTTPException(400, "Training already running")
    session["models"] = []
    session["ensemble"] = []
    session["training_log"] = []
    session["training_status"] = "starting"
    session["training_progress"] = 0

    t = threading.Thread(target=_run_training, args=(cfg,), daemon=True)
    session["training_thread"] = t
    t.start()
    return {"ok": True, "status": "started"}


@app.get("/api/train/status")
def get_training_status():
    return {
        "status": session["training_status"],
        "progress": session["training_progress"],
        "total": session["training_total"],
        "log": session["training_log"][-50:],
        "model_count": len(session["models"]),
    }


@app.post("/api/train/stop")
def stop_training():
    # There's no clean interrupt, but we set status; thread will finish current generation
    session["training_status"] = "idle"
    return {"ok": True}


# ---------------------------------------------------------------------------
# Model Analysis
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models(max_models: int = 50):
    models = session["models"]
    if not models:
        return {"models": []}

    df = session["processed_df"]
    target = session["target_col"]
    features = session["feature_cols"]
    if df is None or not target or not features:
        return {"models": []}

    var_syms = [sgp.symbols(n) for n in features]
    x, y = get_xy(df, target, features)

    result = []
    for i, m in enumerate(models[:max_models]):
        result.append(model_to_dict(m, i, x, y, var_syms))
    return {"models": result}


@app.get("/api/models/{model_id}")
def get_model(model_id: int):
    models = session["models"]
    if model_id >= len(models):
        raise HTTPException(404, "Model not found")

    df = session["processed_df"]
    target = session["target_col"]
    features = session["feature_cols"]
    var_syms = [sgp.symbols(n) for n in features]
    x, y = get_xy(df, target, features)

    return model_to_dict(models[model_id], model_id, x, y, var_syms)


@app.get("/api/models/pareto")
def get_pareto():
    models = session["models"]
    if not models:
        return {"pareto": []}

    df = session["processed_df"]
    target = session["target_col"]
    features = session["feature_cols"]
    var_syms = [sgp.symbols(n) for n in features]
    x, y = get_xy(df, target, features)

    result = []
    for i, m in enumerate(models):
        try:
            fit = float(m[2][0]) if m[2] else None
            cplx = int(m[2][1]) if len(m[2]) > 1 else None
            expr = str(sgp.printGPModel(m, var_syms))
            result.append({"id": i, "fitness": fit, "complexity": cplx, "expression": expr})
        except Exception:
            pass

    return {"pareto": result}


@app.get("/api/models/{model_id}/residuals")
def get_residuals(model_id: int):
    models = session["models"]
    if model_id >= len(models):
        raise HTTPException(404, "Model not found")

    df = session["processed_df"]
    target = session["target_col"]
    features = session["feature_cols"]
    x, y = get_xy(df, target, features)

    train_idx = session["train_indices"]
    test_idx = session["test_indices"]

    model = models[model_id]
    pred = sgp.evaluateGPModel(model, x)
    if not hasattr(pred, '__len__'):
        pred = np.full(len(y), float(pred))
    pred = np.array(pred, dtype=float)
    resid = pred - y

    def safe_list(arr):
        return [float(v) if np.isfinite(v) else None for v in arr]

    return {
        "actual": safe_list(y),
        "predicted": safe_list(pred),
        "residuals": safe_list(resid),
        "train_rmse": float(sgp.rmse(model, x[:, train_idx] if train_idx else x, y[train_idx] if train_idx else y)),
        "test_rmse": float(sgp.rmse(model, x[:, test_idx] if test_idx else x, y[test_idx] if test_idx else y)) if test_idx else None,
        "train_fitness": float(sgp.fitness(model, x[:, train_idx] if train_idx else x, y[train_idx] if train_idx else y)),
        "test_fitness": float(sgp.fitness(model, x[:, test_idx] if test_idx else x, y[test_idx] if test_idx else y)) if test_idx else None,
    }


@app.get("/api/models/variable_importance")
def get_variable_importance(max_models: int = 20):
    models = session["models"]
    features = session["feature_cols"]
    if not models or not features:
        return {"importance": {}}

    var_usage: Dict[str, int] = {f: 0 for f in features}
    for m in models[:max_models]:
        try:
            usage = sgp.stackVarUsage(m[0])
            for i, count in enumerate(usage):
                if i < len(features):
                    var_usage[features[i]] += count
        except Exception:
            pass

    total = sum(var_usage.values()) or 1
    return {"importance": {k: v / total for k, v in var_usage.items()}}


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@app.post("/api/ensemble/build")
def build_ensemble(req: EnsembleRequest):
    models = session["models"]
    if not models:
        raise HTTPException(400, "No models trained yet")

    df = session["processed_df"]
    target = session["target_col"]
    features = session["feature_cols"]
    x, y = get_xy(df, target, features)

    try:
        ensemble = sgp.ensembleSelect(models, x, y, numberOfClusters=min(req.num_clusters, len(models)))
        session["ensemble"] = ensemble
        return {"ok": True, "ensemble_size": len(ensemble)}
    except Exception:
        raise HTTPException(500, "Failed to build ensemble")
def get_ensemble_info():
    ensemble = session["ensemble"]
    features = session["feature_cols"]
    if not ensemble:
        return {"ensemble": []}

    var_syms = [sgp.symbols(n) for n in features]
    result = []
    for i, m in enumerate(ensemble):
        try:
            expr = str(sgp.printGPModel(m, var_syms))
            complexity = int(sgp.stackGPModelComplexity(m))
        except Exception:
            expr = "?"
            complexity = None
        result.append({"id": i, "expression": expr, "complexity": complexity})
    return {"ensemble": result}


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@app.post("/api/predict/model/{model_id}")
def predict_model(model_id: int, req: PredictRequest):
    models = session["models"]
    if model_id >= len(models):
        raise HTTPException(404, "Model not found")

    x_new = np.array(req.rows).T.astype(float)
    model = models[model_id]
    try:
        pred = sgp.evaluateGPModel(model, x_new)
        if not hasattr(pred, '__len__'):
            pred = [float(pred)] * len(req.rows)
        return {"predictions": [float(v) if np.isfinite(float(v)) else None for v in pred]}
    except Exception:
        raise HTTPException(500, "Prediction failed")


@app.post("/api/predict/ensemble")
def predict_ensemble(req: PredictRequest):
    ensemble = session["ensemble"]
    if not ensemble:
        raise HTTPException(400, "No ensemble built yet")

    x_new = np.array(req.rows).T.astype(float)
    try:
        pred = sgp.evaluateModelEnsemble(ensemble, x_new)
        unc = sgp.evaluateModelEnsembleUncertainty(ensemble, x_new)
        if not hasattr(pred, '__len__'):
            pred = [float(pred)] * len(req.rows)
            unc = [float(unc)] * len(req.rows)
        return {
            "predictions": [float(v) if np.isfinite(float(v)) else None for v in pred],
            "uncertainty": [float(v) if np.isfinite(float(v)) else None for v in unc],
        }
    except Exception:
        raise HTTPException(500, "Ensemble prediction failed")


@app.post("/api/predict/uncertainty_point")
def find_max_uncertainty():
    ensemble = session["ensemble"]
    features = session["feature_cols"]
    df = session["processed_df"]
    if not ensemble or df is None:
        raise HTTPException(400, "Need ensemble and processed data")

    try:
        bounds = [[float(df[f].min()), float(df[f].max())] for f in features]
        point = sgp.maximizeUncertainty(ensemble, len(features), bounds=bounds)
        return {"point": point.tolist() if hasattr(point, 'tolist') else list(point)}
    except Exception:
        raise HTTPException(500, "Could not compute uncertainty point")


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

@app.get("/api/session/state")
def get_session_state():
    return {
        "has_raw_data": session["raw_df"] is not None,
        "has_processed_data": session["processed_df"] is not None,
        "target_col": session["target_col"],
        "feature_cols": session["feature_cols"],
        "train_rows": len(session["train_indices"]),
        "test_rows": len(session["test_indices"]),
        "model_count": len(session["models"]),
        "ensemble_size": len(session["ensemble"]),
        "training_status": session["training_status"],
    }


@app.delete("/api/session/reset")
def reset_session():
    for k in ["raw_df", "processed_df", "target_col", "feature_cols",
              "train_indices", "test_indices", "models", "ensemble",
              "training_log"]:
        session[k] = None if k in ["raw_df", "processed_df", "target_col"] else []
    session["training_status"] = "idle"
    session["training_progress"] = 0
    session["training_total"] = 0
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
