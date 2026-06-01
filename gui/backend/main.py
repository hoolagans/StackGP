#!/usr/bin/env python3
"""
StackGP Data Modeling GUI - FastAPI Backend
"""

import sys
import os
import json
import uuid
import threading
import time
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
import io

import StackGP as sgp

app = FastAPI(title="StackGP GUI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    "stop_event": None,
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

    model_dict = {
        "id": idx,
        "expression": str(expr),
        "rmse": rmse_val,
        "fitness": fit_val,
        "complexity": complexity,
        "metrics": safe_metrics,
        "residuals": residuals,
    }
    if hasattr(pred, '__len__'):
        safe_pred = []
        for v in pred:
            try:
                fv = float(v)
            except Exception:
                safe_pred.append(None)
                continue
            safe_pred.append(fv if np.isfinite(fv) else None)
        model_dict["predictions"] = safe_pred
    else:
        model_dict["predictions"] = []
    return model_dict


def active_df() -> Optional[pd.DataFrame]:
    df = session["processed_df"]
    if df is not None:
        return df
    return session["raw_df"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ColumnConfig(BaseModel):
    target_col: str
    feature_cols: List[str]


class ProcessConfig(BaseModel):
    fill_missing: Literal["drop", "mean", "median", "zero"] = "drop"
    normalize: Literal["none", "minmax", "zscore"] = "none"
    train_split: float = Field(default=0.8, gt=0.0, lt=1.0)
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
    ops_set: Literal["default", "all", "boolean"] = "default"   # default | all | boolean
    align: bool = True
    data_subsample: bool = False
    allow_early_termination: bool = False
    early_termination_threshold: float = 0.0

    @model_validator(mode="after")
    def validate_rates(self):
        if self.mutation_rate < 0 or self.crossover_rate < 0 or self.spawn_rate < 0:
            raise ValueError("Mutation/crossover/spawn rates must be non-negative")
        if self.mutation_rate + self.crossover_rate + self.spawn_rate != 100:
            raise ValueError("Mutation/crossover/spawn rates must sum to 100")
        return self


class PredictRequest(BaseModel):
    rows: List[List[float]]


class EnsembleRequest(BaseModel):
    num_clusters: int = 10


# ---------------------------------------------------------------------------
# Data Import
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
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
    if cfg.target_col in cfg.feature_cols:
        raise HTTPException(400, "Target column cannot also be a feature column")
    if not cfg.feature_cols:
        raise HTTPException(400, "At least one feature column is required")
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
    if session["training_status"] in {"starting", "running"}:
        raise HTTPException(400, "Stop training before re-processing data")

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
    if not train_idx or not test_idx:
        raise HTTPException(400, "Train/test split produced an empty set. Adjust split or dataset size.")

    session["processed_df"] = df2
    session["train_indices"] = train_idx
    session["test_indices"] = test_idx
    session["models"] = []
    session["ensemble"] = []
    session["training_log"] = []
    session["training_status"] = "idle"
    session["training_progress"] = 0
    session["training_total"] = 0

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
    df = active_df()
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
    df = active_df()
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
    df = active_df()
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
    df = active_df()
    if df is None:
        raise HTTPException(404, "No data loaded")
    missing = [c for c in (x_col, y_col) if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    if max_points <= 0:
        raise HTTPException(400, "max_points must be > 0")
    sub = df[[x_col, y_col]].dropna()
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)
    return {
        "x": sub[x_col].tolist(),
        "y": sub[y_col].tolist(),
    }


@app.get("/api/explore/pairplot")
def get_pairplot(max_points: int = 300):
    df = active_df()
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


_TRAINING_CHUNK = 10  # generations per progress/stop-check interval


def _run_training(cfg: TrainConfig, stop_event: threading.Event):
    with session["training_lock"]:
        df = session["processed_df"]
        target = session["target_col"]
        features = list(session["feature_cols"])
        train_idx = list(session["train_indices"])

    if df is None or not target or not features:
        with session["training_lock"]:
            session["training_status"] = "error"
        return

    train_df = df.iloc[train_idx] if train_idx else df
    x, y = get_xy(train_df, target, features)
    var_names = [str(n) for n in features]

    ops = _ops_set(cfg.ops_set)

    log = []
    with session["training_lock"]:
        session["training_log"] = log
        session["training_progress"] = 0
        session["training_total"] = cfg.generations
        session["training_status"] = "running"

    current_models: list = []
    gens_done = 0
    start_time = time.perf_counter()

    while gens_done < cfg.generations and not stop_event.is_set():
        chunk = min(_TRAINING_CHUNK, cfg.generations - gens_done)
        elapsed = time.perf_counter() - start_time
        if elapsed >= cfg.time_limit:
            break
        remaining_time = cfg.time_limit - elapsed
        is_last_chunk = (gens_done + chunk >= cfg.generations)

        try:
            result = sgp.evolve(
                x, y,
                generations=chunk,
                ops=ops,
                variableNames=var_names,
                mutationRate=cfg.mutation_rate,
                crossoverRate=cfg.crossover_rate,
                spawnRate=cfg.spawn_rate,
                popSize=cfg.pop_size,
                maxComplexity=cfg.max_complexity,
                align=is_last_chunk and cfg.align,
                timeLimit=remaining_time,
                capTime=True,
                dataSubsample=cfg.data_subsample,
                allowEarlyTermination=cfg.allow_early_termination,
                earlyTerminationThreshold=cfg.early_termination_threshold,
                initialPop=current_models,
                returnTracking=True,
                liveTracking=False,
            )
            current_models, best_fits = result
        except Exception as e:
            with session["training_lock"]:
                session["training_status"] = "error"
            log.append({"error": "Training failed — check server logs"})
            return

        gens_done += chunk
        with session["training_lock"]:
            session["training_progress"] = gens_done

        # Populate per-generation log entries from tracking data.
        # Complexity is only known for the current best model at chunk end.
        cplx = None
        if current_models:
            try:
                cplx = int(sgp.stackGPModelComplexity(current_models[0]))
            except Exception:
                cplx = None
        for j, fit in enumerate(best_fits):
            entry: Dict[str, Any] = {"gen": gens_done - chunk + j + 1}
            try:
                entry["fitness"] = float(fit)
            except Exception:
                entry["fitness"] = None
            if cplx is not None and j == len(best_fits) - 1:
                entry["complexity"] = cplx
            log.append(entry)

    with session["training_lock"]:
        session["models"] = current_models
        if stop_event.is_set():
            session["training_status"] = "idle"
        else:
            session["training_status"] = "done"
            session["training_progress"] = cfg.generations


@app.post("/api/train/start")
def start_training(cfg: TrainConfig):
    with session["training_lock"]:
        old_thread = session.get("training_thread")
        if session["training_status"] in {"starting", "running"} or (old_thread is not None and old_thread.is_alive()):
            raise HTTPException(400, "Training already running")
        session["models"] = []
        session["ensemble"] = []
        session["training_log"] = []
        session["training_status"] = "starting"
        session["training_progress"] = 0
        session["training_total"] = cfg.generations
        stop_event = threading.Event()
        t = threading.Thread(target=_run_training, args=(cfg, stop_event), daemon=True)
        session["stop_event"] = stop_event
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
    stop_event: Optional[threading.Event] = session.get("stop_event")
    thread: Optional[threading.Thread] = session.get("training_thread")
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    with session["training_lock"]:
        status = "running" if thread is not None and thread.is_alive() else "idle"
        session["training_status"] = status
    return {"ok": True, "status": status}


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
        item = model_to_dict(m, i, x, y, var_syms)
        item.pop("predictions", None)
        result.append(item)
    return {"models": result}


@app.get("/api/models/pareto")
def get_pareto():
    models = session["models"]
    if not models:
        return {"pareto": []}

    df = session["processed_df"]
    target = session["target_col"]
    features = session["feature_cols"]
    if df is None or not target or not features:
        return {"pareto": []}
    var_syms = [sgp.symbols(n) for n in features]
    x, y = get_xy(df, target, features)

    result = []
    for i, m in enumerate(models):
        try:
            fit = float(sgp.fitness(m, x, y))
            cplx = int(sgp.stackGPModelComplexity(m))
            expr = str(sgp.printGPModel(m, var_syms))
            result.append({"id": i, "fitness": fit, "complexity": cplx, "expression": expr})
        except Exception:
            pass

    return {"pareto": result}


@app.get("/api/models/variable_importance")
def get_variable_importance(max_models: int = 20):
    models = session["models"]
    features = session["feature_cols"]
    if not models or not features:
        return {"importance": {}}

    var_usage: Dict[str, int] = {f: 0 for f in features}
    var_syms = [sgp.symbols(n) for n in features]
    for m in models[:max_models]:
        try:
            expr = str(sgp.printGPModel(m, var_syms))
            for f in features:
                var_usage[f] += expr.count(f)
        except Exception:
            pass

    total = sum(var_usage.values()) or 1
    return {"importance": {k: v / total for k, v in var_usage.items()}}


@app.get("/api/models/{model_id}")
def get_model(model_id: int):
    models = session["models"]
    if model_id < 0 or model_id >= len(models):
        raise HTTPException(404, "Model not found")

    df = session["processed_df"]
    target = session["target_col"]
    features = session["feature_cols"]
    var_syms = [sgp.symbols(n) for n in features]
    x, y = get_xy(df, target, features)

    return model_to_dict(models[model_id], model_id, x, y, var_syms)


@app.get("/api/models/{model_id}/residuals")
def get_residuals(model_id: int):
    models = session["models"]
    if model_id < 0 or model_id >= len(models):
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


@app.get("/api/ensemble")
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
    if model_id < 0 or model_id >= len(models):
        raise HTTPException(404, "Model not found")
    features = session["feature_cols"]
    if not features:
        raise HTTPException(400, "No feature configuration found")
    if any(len(row) != len(features) for row in req.rows):
        raise HTTPException(400, f"Each row must have exactly {len(features)} values")

    x_new = np.array(req.rows).T.astype(float)
    model = models[model_id]
    try:
        pred = sgp.evaluateGPModel(model, x_new)
        if not hasattr(pred, '__len__'):
            pred = [float(pred)] * len(req.rows)
        safe_pred = []
        for v in pred:
            fv = float(v)
            safe_pred.append(fv if np.isfinite(fv) else None)
        return {"predictions": safe_pred}
    except Exception:
        raise HTTPException(500, "Prediction failed")


@app.post("/api/predict/ensemble")
def predict_ensemble(req: PredictRequest):
    ensemble = session["ensemble"]
    if not ensemble:
        raise HTTPException(400, "No ensemble built yet")

    features = session["feature_cols"]
    if not features:
        raise HTTPException(400, "No feature configuration found")
    if any(len(row) != len(features) for row in req.rows):
        raise HTTPException(400, f"Each row must have exactly {len(features)} values")

    x_new = np.array(req.rows).T.astype(float)
    try:
        pred = sgp.evaluateModelEnsemble(ensemble, x_new)
        unc = sgp.evaluateModelEnsembleUncertainty(ensemble, x_new)
        if not hasattr(pred, '__len__'):
            pred = [float(pred)] * len(req.rows)
            unc = [float(unc)] * len(req.rows)
        safe_pred = []
        safe_unc = []
        for pv, uv in zip(pred, unc):
            fp = float(pv)
            fu = float(uv)
            safe_pred.append(fp if np.isfinite(fp) else None)
            safe_unc.append(fu if np.isfinite(fu) else None)
        return {"predictions": safe_pred, "uncertainty": safe_unc}
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
    stop_event: Optional[threading.Event] = session.get("stop_event")
    thread: Optional[threading.Thread] = session.get("training_thread")
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    for k in ["raw_df", "processed_df", "target_col", "feature_cols",
              "train_indices", "test_indices", "models", "ensemble",
              "training_log"]:
        session[k] = None if k in ["raw_df", "processed_df", "target_col"] else []
    session["training_status"] = "idle"
    session["training_progress"] = 0
    session["training_total"] = 0
    session["training_thread"] = None
    session["stop_event"] = None
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# Serve the React SPA in production (must be defined after all API routes so
# the catch-all does not shadow them).
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    from fastapi.responses import FileResponse as _FileResponse

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        candidate = _static_dir / full_path
        if candidate.is_file():
            return _FileResponse(str(candidate))
        return _FileResponse(str(_static_dir / "index.html"))
