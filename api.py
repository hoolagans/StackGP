#!/usr/bin/env python3
"""
StackGP Web API
Flask backend that exposes StackGP evolution and visualisation functions
to a browser-based front-end.
"""

import copy
import datetime
import io
import json
import math
import os
import queue
import threading
import time
import uuid

import base64
import dill
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import StackGP as sgp

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# Persistent storage directories (created at startup)
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "saved_datasets")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,   exist_ok=True)

# In-memory job registry  {job_id -> dict}
JOBS: dict = {}
JOBS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPS_MAP = {
    "defaultOps":  sgp.defaultOps,
    "allOps":      sgp.allOps,
    "booleanOps":  sgp.booleanOps,
}
CONST_MAP = {
    "defaultConst":  sgp.defaultConst,
    "booleanConst":  sgp.booleanConst,
}
SAMPLING_MAP = {
    "randomSubsample":              sgp.randomSubsample,
    "generationProportionalSample": sgp.generationProportionalSample,
    "ordinalSample":                sgp.ordinalSample,
    "orderedSample":                sgp.orderedSample,
    "ordinalBalancedSample":        sgp.ordinalBalancedSample,
    "balancedSample":               sgp.balancedSample,
}


def _new_job():
    job_id = str(uuid.uuid4())
    job = {
        "id":         job_id,
        "status":     "pending",  # pending | running | complete | error | cancelled
        "tracking":   [],         # list[float] — best fitness per generation
        "models":     None,       # serialised via dill after completion
        "error":      None,
        "queue":      queue.Queue(),   # internal SSE messages
        "generation": 0,
        "total_gens": 0,
        "cancelled":  False,      # set True by cancel endpoint
        "best_expr":  None,       # best model expression from latest batch
        "meta":       {},         # arbitrary metadata (dataset, method, etc.)
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def _model_summary_row(mod, input_data, response_data, var_names,
                       test_input=None, test_response=None):
    """Return a JSON-serialisable dict describing one model."""
    try:
        expr = str(sgp.printGPModel(mod))
    except Exception:
        expr = str(mod)
    try:
        fit  = float(sgp.fitness(mod, input_data, response_data))
        r2   = round(1.0 - fit, 6) if math.isfinite(fit) else None
    except Exception:
        fit, r2 = None, None
    try:
        err  = float(sgp.rmse(mod, input_data, response_data))
        err  = round(err, 6) if math.isfinite(err) else None
    except Exception:
        err = None
    try:
        comp = int(sgp.stackGPModelComplexity(mod))
    except Exception:
        comp = None
    row = {"expression": expr, "r2": r2, "rmse": err, "complexity": comp}
    if test_input is not None and test_response is not None:
        try:
            tfit = float(sgp.fitness(mod, test_input, test_response))
            row["test_r2"] = round(1.0 - tfit, 6) if math.isfinite(tfit) else None
        except Exception:
            row["test_r2"] = None
        try:
            terr = float(sgp.rmse(mod, test_input, test_response))
            row["test_rmse"] = round(terr, 6) if math.isfinite(terr) else None
        except Exception:
            row["test_rmse"] = None
    return row


def _capture_plot(fn, *args, **kwargs):
    """
    Call *fn* which internally calls plt.show(), capture the resulting
    figure as a base64-encoded PNG and return the string.
    """
    original_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        fn(*args, **kwargs)
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode("utf-8")
    finally:
        plt.show = original_show
        plt.close("all")
    return img


def _sanitise_name(name: str) -> str:
    """
    Use werkzeug's secure_filename to strip path separators and special
    characters, then additionally limit to alphanumerics, hyphens, and
    underscores so that user-supplied names cannot escape the storage dirs.
    """
    base = secure_filename(name)
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in base)


def _safe_models_path(name: str) -> str | None:
    """Return the absolute path for a model file, or None if unsafe."""
    safe = _sanitise_name(name)
    if not safe:
        return None
    candidate = os.path.join(MODELS_DIR, f"{safe}.dill")
    real_models = os.path.realpath(MODELS_DIR) + os.sep
    real_candidate = os.path.realpath(candidate)
    if not real_candidate.startswith(real_models):
        return None
    return candidate


def _safe_dataset_path(name: str) -> str | None:
    """Return the absolute path for a dataset file, or None if unsafe."""
    safe = _sanitise_name(name)
    if not safe:
        return None
    candidate = os.path.join(DATASETS_DIR, f"{safe}.csv")
    real_datasets = os.path.realpath(DATASETS_DIR) + os.sep
    real_candidate = os.path.realpath(candidate)
    if not real_candidate.startswith(real_datasets):
        return None
    return candidate


def _load_models(name: str):
    path = _safe_models_path(name)
    if path is None or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return dill.load(f)


def _save_models_to_disk(name: str, models):
    path = _safe_models_path(name)
    if path is None:
        raise ValueError("Invalid model set name")
    with open(path, "wb") as f:
        dill.dump(models, f)


def _read_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV file, auto-detecting whether it has a header row.

    If the first row looks like data (all values can be parsed as numbers)
    the file is treated as headerless and columns are named x0, x1, … x(n-1).
    Otherwise pandas' default header=0 behaviour is used.
    """
    df_default = pd.read_csv(path)
    # Check whether every column name looks numeric (i.e. came from a data row)
    def _is_numeric_str(s: str) -> bool:
        try:
            float(str(s))
            return True
        except ValueError:
            return False

    if df_default.columns.size > 0 and all(
        _is_numeric_str(c) for c in df_default.columns
    ):
        df_no_hdr = pd.read_csv(path, header=None)
        df_no_hdr.columns = [f"x{i}" for i in range(len(df_no_hdr.columns))]
        return df_no_hdr
    return df_default


def _load_dataset(name: str):
    path = _safe_dataset_path(name)
    if path is None or not os.path.exists(path):
        return None
    return _read_csv(path)


# ---------------------------------------------------------------------------
# Background evolution worker
# ---------------------------------------------------------------------------

def _evolution_worker(job: dict, df: pd.DataFrame, response_col: str,
                      settings: dict, initial_models):
    """
    Runs evolve() (or parallelEvolve()) in small generation-batches so the
    front-end receives live tracking updates via SSE.
    """
    q: queue.Queue = job["queue"]
    try:
        job["status"] = "running"

        # ---- Prepare numpy arrays ----------------------------------------
        feature_cols = [c for c in df.columns if c != response_col]
        all_input    = df[feature_cols].values.T.astype(float)   # (n_vars, n_samples)
        all_response = df[response_col].values.astype(float)

        # ---- Optional train/test split -----------------------------------
        test_split_pct = float(settings.pop("testSplit", 0))
        test_input  = None
        test_response = None
        if 0 < test_split_pct < 100:
            n_total = all_input.shape[1]
            n_test  = max(1, int(round(n_total * test_split_pct / 100.0)))
            n_train = n_total - n_test
            rng = np.random.default_rng(42)
            idx = rng.permutation(n_total)
            train_idx, test_idx = idx[:n_train], idx[n_train:]
            input_data    = all_input[:, train_idx]
            response_data = all_response[train_idx]
            test_input    = all_input[:, test_idx]
            test_response = all_response[test_idx]
        else:
            input_data    = all_input
            response_data = all_response

        job["test_input"]    = test_input
        job["test_response"] = test_response

        # ---- Resolve string keys to StackGP objects ----------------------
        ops_key  = settings.pop("ops",  "defaultOps")
        const_key = settings.pop("const", "defaultConst")
        samp_key = settings.pop("samplingMethod", "randomSubsample")

        ops_fn    = OPS_MAP.get(ops_key,    OPS_MAP["defaultOps"])()
        const_fn  = CONST_MAP.get(const_key, CONST_MAP["defaultConst"])()
        samp_fn   = SAMPLING_MAP.get(samp_key, sgp.randomSubsample)

        use_parallel = settings.pop("useParallel", False)
        n_jobs       = int(settings.pop("n_jobs", -1))
        cascades     = bool(settings.pop("cascades", False))
        cascade_count = int(settings.pop("cascadeCount", 10))
        exchange_count = int(settings.pop("exchangeCount", 5))

        total_gens   = int(settings.pop("generations", 100))
        batch_size   = max(1, int(settings.pop("liveTrackingInterval", 10)))
        var_names    = feature_cols

        job["total_gens"] = total_gens

        # ---- Common kwargs -----------------------------------------------
        # Both serial and parallel paths use returnTracking to stream fitness
        # history per batch.  parallelEvolve was fixed to return
        # (sorted_models, combined_tracking) when returnTracking=True.
        base_kwargs = dict(
            ops            = ops_fn,
            const          = const_fn,
            mutationRate   = int(settings.get("mutationRate", 79)),
            crossoverRate  = int(settings.get("crossoverRate", 11)),
            spawnRate      = int(settings.get("spawnRate", 10)),
            extinction     = bool(settings.get("extinction", False)),
            extinctionRate = int(settings.get("extinctionRate", 10)),
            elitismRate    = int(settings.get("elitismRate", 10)),
            popSize        = int(settings.get("popSize", 300)),
            maxComplexity  = int(settings.get("maxComplexity", 100)),
            align          = bool(settings.get("align", True)),
            timeLimit      = int(settings.get("timeLimit", 300)),
            capTime        = bool(settings.get("capTime", False)),
            tourneySize    = int(settings.get("tourneySize", 5)),
            dataSubsample  = bool(settings.get("dataSubsample", False)),
            samplingMethod = samp_fn,
            allowEarlyTermination   = bool(settings.get("allowEarlyTermination", False)),
            earlyTerminationThreshold = float(settings.get("earlyTerminationThreshold", 0)),
            liveTracking   = False,
            tracking       = False,
            variableNames  = var_names,
        )
        # Serial path gets returnTracking so we can stream fitness per batch.
        serial_extra   = {"returnTracking": True}
        # Parallel path also uses returnTracking; parallelEvolve now returns
        # (sorted_models, combined_tracking) when returnTracking=True.
        parallel_extra = {"returnTracking": True}

        tracking_all: list = []
        current_models     = list(initial_models) if initial_models else []
        gens_done          = 0

        # ---- Batch loop --------------------------------------------------
        while gens_done < total_gens:
            # Check for server-side cancellation between batches
            if job.get("cancelled"):
                job["status"] = "cancelled"
                q.put(json.dumps({"type": "cancelled"}))
                return

            this_batch = min(batch_size, total_gens - gens_done)

            if use_parallel:
                batch_kwargs = {**base_kwargs, **parallel_extra,
                                "generations": this_batch,
                                "initialPop":  copy.deepcopy(current_models)}
                result = sgp.parallelEvolve(
                    input_data, response_data,
                    n_jobs        = n_jobs,
                    cascades      = cascades,
                    cascadeCount  = cascade_count,
                    exchangeCount = exchange_count,
                    **batch_kwargs,
                )
                # parallelEvolve returns (models, tracking) when returnTracking=True
                if isinstance(result, tuple):
                    current_models, batch_tracking = result
                    batch_tracking = list(batch_tracking)
                else:
                    current_models = result if isinstance(result, list) else list(result)
                    batch_tracking = []
            else:
                batch_kwargs = {**base_kwargs, **serial_extra,
                                "generations": this_batch,
                                "initialPop":  copy.deepcopy(current_models)}
                result = sgp.evolve(input_data, response_data, **batch_kwargs)
                if isinstance(result, tuple):
                    current_models, batch_tracking = result
                else:
                    current_models = result
                    batch_tracking = []

            tracking_all.extend(batch_tracking)
            gens_done += this_batch
            job["generation"] = gens_done
            job["tracking"]   = list(tracking_all)

            # Best model expression for live preview
            best_expr = None
            if current_models:
                try:
                    sorted_now = sgp.sortModels(current_models)
                    best_expr  = str(sgp.printGPModel(sorted_now[0]))
                    job["best_expr"] = best_expr
                except Exception:
                    pass

            best = float(min(batch_tracking)) if batch_tracking else None
            q.put(json.dumps({
                "type":       "progress",
                "generation": gens_done,
                "total":      total_gens,
                "tracking":   tracking_all,
                "best":       best,
                "best_expr":  best_expr,
            }))

        # ---- Finalise ----------------------------------------------------
        current_models = sgp.sortModels(current_models)
        job["models"]  = current_models
        job["status"]  = "complete"
        job["input_data"]    = input_data
        job["response_data"] = response_data
        job["var_names"]     = var_names

        # Build summary rows for the Pareto front models (non-dominated in
        # accuracy vs. complexity).  Mirror what plotModels() does internally:
        # deep-copy, convert to list form, extract front, restore, then sort.
        # We also track each Pareto model's original index in current_models so
        # the frontend can send the correct model_index to the plot endpoint.
        pareto_mods = copy.deepcopy(current_models)
        if pareto_mods:
            # Ensure each model has both fitness and complexity as objectives.
            # mod[2] may be a list, numpy array, or a scalar depending on the
            # evolution path, so handle all three cases safely.
            first_obj = pareto_mods[0][2]
            needs_complexity = (
                not hasattr(first_obj, "__len__") or len(first_obj) < 2
            )
            if needs_complexity:
                for mod in pareto_mods:
                    obj = mod[2]
                    fitness_val = (
                        float(obj[0]) if hasattr(obj, "__len__") else float(obj)
                    )
                    mod[2] = [fitness_val, sgp.stackGPModelComplexity(mod)]
            [sgp.modelToListForm(mod) for mod in pareto_mods]
            pareto_front = sgp.paretoTournament(pareto_mods)
            [sgp.modelRestoreForm(mod) for mod in pareto_front]
            pareto_front.sort(key=lambda m: m[2][0])  # best accuracy first

            # Map each Pareto model back to its position in current_models by
            # matching on the expression string so the plot endpoint receives
            # the correct index into the full sorted list.
            def _expr(mod):
                try:
                    return str(sgp.printGPModel(mod))
                except Exception:
                    return ""

            original_exprs = [_expr(m) for m in current_models]
            pareto_original_indices = []
            for pmod in pareto_front:
                pexpr = _expr(pmod)
                idx = next(
                    (i for i, e in enumerate(original_exprs) if e == pexpr),
                    0,
                )
                pareto_original_indices.append(idx)
        else:
            pareto_front = []
            pareto_original_indices = []

        summary = [
            {**_model_summary_row(m, input_data, response_data, var_names,
                                  test_input, test_response),
             "model_index": orig_idx}
            for m, orig_idx in zip(pareto_front, pareto_original_indices)
        ]

        q.put(json.dumps({
            "type":     "complete",
            "tracking": tracking_all,
            "summary":  summary,
            "has_test_split": test_input is not None,
        }))

    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Evolution worker failed")
        job["status"] = "error"
        job["error"]  = str(exc)
        q.put(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        q.put(None)   # sentinel — SSE generator will close stream


# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ---------------------------------------------------------------------------
# Routes — dataset
# ---------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """Accept a CSV file, persist it, return column names and a preview."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    safe_name = _sanitise_name(os.path.splitext(file.filename)[0])
    save_path = _safe_dataset_path(safe_name)
    if save_path is None:
        return jsonify({"error": "Invalid filename"}), 400
    file.save(save_path)

    df = _read_csv(save_path)
    # If the file was headerless we re-save it with the auto-generated column
    # names so that every subsequent load (including _load_dataset) sees them.
    if df.columns[0] == "x0":
        df.to_csv(save_path, index=False)
    preview = df.head(5).to_dict(orient="records")
    return jsonify({
        "name":    safe_name,
        "columns": list(df.columns),
        "rows":    len(df),
        "preview": preview,
    })


@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    files = [f[:-4] for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]
    return jsonify(files)


@app.route("/api/datasets/<name>", methods=["GET"])
def get_dataset(name):
    df = _load_dataset(name)
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify({
        "name":    name,
        "columns": list(df.columns),
        "rows":    len(df),
        "preview": df.head(5).to_dict(orient="records"),
    })


@app.route("/api/datasets/<name>", methods=["DELETE"])
def delete_dataset(name):
    path = _safe_dataset_path(name)
    if path is None:
        return jsonify({"error": "Invalid name"}), 400
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"deleted": _sanitise_name(name)})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/datasets/<name>/stats", methods=["GET"])
def dataset_stats(name):
    """Return per-column statistics: mean, std, min, max, null_count."""
    df = _load_dataset(name)
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404
    stats = []
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            stats.append({
                "column":     col,
                "mean":       round(float(valid.mean()), 6) if len(valid) else None,
                "std":        round(float(valid.std()),  6) if len(valid) > 1 else None,
                "min":        round(float(valid.min()),  6) if len(valid) else None,
                "max":        round(float(valid.max()),  6) if len(valid) else None,
                "null_count": null_count,
            })
        else:
            stats.append({
                "column":     col,
                "mean":       None,
                "std":        None,
                "min":        None,
                "max":        None,
                "null_count": null_count,
            })
    return jsonify(stats)


# ---------------------------------------------------------------------------
# Routes — evolution
# ---------------------------------------------------------------------------

@app.route("/api/evolve", methods=["POST"])
def start_evolve():
    """
    Body (JSON):
      dataset_name   str   — name of previously uploaded dataset
      response_col   str   — column to use as response variable
      settings       dict  — evolve/parallelEvolve keyword arguments
      seed_models    str?  — name of a saved model set to seed with (optional)
      seed_job_id    str?  — job ID whose in-memory models to use as seed (optional)
    """
    body = request.get_json(force=True)
    dataset_name = body.get("dataset_name")
    response_col = body.get("response_col")
    settings     = dict(body.get("settings", {}))
    seed_name    = body.get("seed_models")
    seed_job_id  = body.get("seed_job_id")

    if not dataset_name or not response_col:
        return jsonify({"error": "dataset_name and response_col are required"}), 400

    df = _load_dataset(dataset_name)
    if df is None:
        return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404
    if response_col not in df.columns:
        return jsonify({"error": f"Column '{response_col}' not in dataset"}), 400

    initial_models = []
    if seed_job_id:
        # Seed from in-memory job (no save required)
        with JOBS_LOCK:
            seed_job = JOBS.get(seed_job_id)
        if seed_job and seed_job.get("models"):
            initial_models = list(seed_job["models"])
    elif seed_name:
        loaded = _load_models(seed_name)
        if loaded:
            initial_models = loaded

    job = _new_job()
    # Store metadata for history panel
    job["meta"] = {
        "dataset":   dataset_name,
        "response":  response_col,
        "method":    "parallelEvolve" if settings.get("useParallel") else "evolve",
        "generations": int(settings.get("generations", 100)),
        "started_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    thread = threading.Thread(
        target=_evolution_worker,
        args=(job, df, response_col, settings, initial_models),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job["id"]})


@app.route("/api/jobs/<job_id>/stream")
def stream_job(job_id):
    """Server-Sent Events stream for live tracking updates."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q: queue.Queue = job["queue"]
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                yield "data: {\"type\":\"heartbeat\"}\n\n"
                continue
            if msg is None:
                break
            yield f"data: {msg}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "id":         job["id"],
        "status":     job["status"],
        "generation": job["generation"],
        "total_gens": job["total_gens"],
        "tracking":   job["tracking"],
        "error":      job["error"],
        "has_models": job["models"] is not None,
        "meta":       job.get("meta", {}),
    })


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Signal the evolution worker to stop after the current batch."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] not in ("pending", "running"):
        return jsonify({"error": "Job is not running"}), 400
    job["cancelled"] = True
    return jsonify({"cancelled": job_id})


# ---------------------------------------------------------------------------
# Routes — model sets
# ---------------------------------------------------------------------------

@app.route("/api/models", methods=["GET"])
def list_model_sets():
    files = [f[:-5] for f in os.listdir(MODELS_DIR) if f.endswith(".dill")]
    return jsonify(files)


@app.route("/api/models/<name>", methods=["POST"])
def save_model_set(name):
    """
    Body (JSON): { "job_id": "..." }
    Saves the models from that job under *name*.
    """
    body    = request.get_json(force=True)
    job_id  = body.get("job_id")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or job["models"] is None:
        return jsonify({"error": "No models available for that job"}), 404
    safe = _sanitise_name(name)
    _save_models_to_disk(safe, job["models"])
    return jsonify({"saved": safe})


@app.route("/api/models/<name>", methods=["DELETE"])
def delete_model_set(name):
    path = _safe_models_path(name)
    if path is None:
        return jsonify({"error": "Invalid name"}), 400
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"deleted": _sanitise_name(name)})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/models/<name>/summary", methods=["GET"])
def model_set_summary(name):
    """Return a JSON summary table for a saved model set (no data required)."""
    models = _load_models(name)
    if models is None:
        return jsonify({"error": "Model set not found"}), 404
    rows = []
    for mod in models[:20]:
        try:
            expr = str(sgp.printGPModel(mod))
        except Exception:
            expr = str(mod)
        try:
            comp = int(sgp.stackGPModelComplexity(mod))
        except Exception:
            comp = None
        rows.append({"expression": expr, "complexity": comp})
    return jsonify(rows)


# ---------------------------------------------------------------------------
# Routes — plots
# ---------------------------------------------------------------------------

@app.route("/api/plot", methods=["POST"])
def generate_plot():
    """
    Body (JSON):
      plot_type      str   — one of the recognised plot function names
      job_id         str?  — use models from a running/complete job
      models_name    str?  — use a saved model set
      dataset_name   str?  — required for per-model plots
      response_col   str?  — required for per-model plots
      model_index    int?  — which model to use (default 0)
      sort           bool? — for plotModelResponseComparison
      variables      list? — variable names for plotVariablePresence
    """
    body         = request.get_json(force=True)
    plot_type    = body.get("plot_type", "plotModels")
    job_id       = body.get("job_id")
    models_name  = body.get("models_name")
    dataset_name = body.get("dataset_name")
    response_col = body.get("response_col")
    model_index  = int(body.get("model_index", 0))
    sort_flag    = bool(body.get("sort", False))
    var_names    = body.get("variables", [])

    # -- Resolve models
    models = None
    if job_id:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job:
            models = job.get("models")
    if models is None and models_name:
        models = _load_models(models_name)
    if models is None:
        return jsonify({"error": "No models found for given job_id or models_name"}), 400

    # -- Resolve data (if needed)
    input_data    = None
    response_data = None
    if dataset_name and response_col:
        df = _load_dataset(dataset_name)
        if df is not None and response_col in df.columns:
            feat_cols     = [c for c in df.columns if c != response_col]
            input_data    = df[feat_cols].values.T.astype(float)
            response_data = df[response_col].values.astype(float)
            if not var_names:
                var_names = feat_cols

    PLOT_FUNCS = {
        "plotModels":                     lambda: sgp.plotModels(models),
        "plotModelComplexityDistribution": lambda: sgp.plotModelComplexityDistribution(models),
        "plotModelAccuracyDistribution":   lambda: sgp.plotModelAccuracyDistribution(models),
        "plotVariablePresence":            lambda: sgp.plotVariablePresence(models, variables=var_names or ["x"+str(i) for i in range(100)]),
        "plotOperatorPresence":            lambda: sgp.plotOperatorPresence(models),
    }
    DATA_PLOT_FUNCS = {
        "plotModelResponseComparison":    lambda m: sgp.plotModelResponseComparison(m, input_data, response_data, sort=sort_flag),
        "plotPredictionResponseCorrelation": lambda m: sgp.plotPredictionResponseCorrelation(m, input_data, response_data),
        "plotModelResiduals":             lambda m: sgp.plotModelResiduals(m, input_data, response_data),
        "plotModelResidualDistribution":  lambda m: sgp.plotModelResidualDistribution(m, input_data, response_data),
    }

    def _sensitivity_plot(model, inp, vnames):
        """Partial-dependence (sensitivity) chart — one series per variable."""
        n_points = 50
        n_vars   = inp.shape[0]
        col_means = inp.mean(axis=1)   # shape (n_vars,)
        fig, ax = plt.subplots(figsize=(9, 5))
        for vi in range(n_vars):
            lo, hi = inp[vi].min(), inp[vi].max()
            if lo == hi:
                continue
            sweep     = np.linspace(lo, hi, n_points)
            x_sweep   = np.tile(col_means[:, np.newaxis], (1, n_points))
            x_sweep[vi] = sweep
            try:
                preds = sgp.evaluateGPModel(model, x_sweep)
                if not isinstance(preds, np.ndarray):
                    preds = np.full(n_points, float(preds))
                preds = np.where(np.isfinite(preds), preds, np.nan)
            except Exception:
                continue
            label = vnames[vi] if vi < len(vnames) else f"x{vi}"
            ax.plot(sweep, preds, label=label)
        ax.set_xlabel("Variable value")
        ax.set_ylabel("Model output")
        ax.set_title("Sensitivity Analysis (partial dependence)")
        ax.legend(loc="best", fontsize="small")
        plt.tight_layout()

    try:
        if plot_type in PLOT_FUNCS:
            img = _capture_plot(PLOT_FUNCS[plot_type])
        elif plot_type in DATA_PLOT_FUNCS:
            if input_data is None or response_data is None:
                return jsonify({"error": "dataset_name and response_col required for this plot"}), 400
            if model_index >= len(models):
                model_index = 0
            img = _capture_plot(DATA_PLOT_FUNCS[plot_type], models[model_index])
        elif plot_type == "sensitivityAnalysis":
            if input_data is None:
                return jsonify({"error": "dataset_name and response_col required for sensitivity analysis"}), 400
            if model_index >= len(models):
                model_index = 0
            img = _capture_plot(_sensitivity_plot, models[model_index], input_data, var_names)
        else:
            return jsonify({"error": f"Unknown plot type '{plot_type}'"}), 400
    except Exception:  # noqa: BLE001
        app.logger.exception("Plot generation failed")
        return jsonify({"error": "Plot generation failed. Check that your data and models are valid."}), 500

    return jsonify({"image": img, "format": "png"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
