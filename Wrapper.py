import numpy as np
from sympy import symbols
# ─── SKLearn Type StackGP Wrapper ──────────────────────────────────────────────────────────

def _stackgp_symbol_vars(n_features, feature_names=None):
    from sympy import Symbol
    names = list(feature_names) if feature_names is not None else None
    return [
        Symbol(str(names[i])) if names and i < len(names) else Symbol(f"x{i}")
        for i in range(n_features or 0)
    ]

class StackGPRegressor:
    """Scikit-learn–compatible wrapper around StackGP's evolve/evaluate API."""

    def __init__(self, generations=30, pop_size=100, time_limit=60, operator_set="default"):
        self.generations = int(generations)
        self.pop_size = int(pop_size)
        self.time_limit = int(time_limit)
        self.operator_set = "allOps" if str(operator_set) == "allOps" else "default"
        self.models_ = None
        self.best_model_ = None
        self.n_features_in_ = None
        self.feature_names_ = None
        self.training_curve_ = []

    def _to_input_list(self, X):
        arr = np.array(X, dtype=float)
        return [arr[:, i] for i in range(arr.shape[1])]

    def fit(self, X, y):
        from .StackGP import evolve, allOps as _allOps, defaultOps as _defaultOps
        if len(X) != len(y):
            print("Warning: Mismatch in number of samples between X and y. Attempting transpose.")
            X=X.T
        self.n_features_in_ = np.array(X).shape[1]
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        input_data = self._to_input_list(X)
        response = np.array(y, dtype=float)
        ops = _allOps() if self.operator_set == "allOps" else _defaultOps()
        self.models_, self.training_curve_ = evolve(
            input_data,
            response,
            generations=self.generations,
            popSize=self.pop_size,
            ops=ops,
            capTime=True,
            timeLimit=self.time_limit,
            returnTracking=True,
        )
        self.best_model_ = self.models_[0] if self.models_ else None
        return self

    def predict(self, X):
        from .StackGP import evaluateGPModel
        if self.best_model_ is None:
            raise ValueError("StackGPRegressor is not fitted yet.")
        input_data = self._to_input_list(X)
        pred = evaluateGPModel(self.best_model_, input_data)
        return np.array(pred, dtype=float)

    def get_params(self, deep=True):
        return {
            "generations": self.generations,
            "pop_size": self.pop_size,
            "time_limit": self.time_limit,
            "operator_set": self.operator_set,
        }

    def get_formula(self, feature_names=None):
        """Return symbolic formula string for the best evolved model."""
        if self.best_model_ is None:
            return "N/A"
        try:
            from .StackGP import printGPModel
            from sympy import simplify
            names = feature_names if feature_names is not None else self.feature_names_
            sym_vars = _stackgp_symbol_vars(self.n_features_in_, names)
            expr = printGPModel(self.best_model_, inputData=sym_vars)
            try:
                expr = simplify(expr)
            except Exception:
                pass
            return str(expr)
        except Exception:
            return "N/A"

    def get_feature_importance(self, feature_names=symbols(["x"+str(i) for i in range(100)])):
        """Approximate importance as normalised variable occurrence count."""
        if self.best_model_ is None:
            return [0.0] * len(feature_names)
        n = len(feature_names)
        try:
            from .StackGP import printGPModel
            sym_vars = _stackgp_symbol_vars(self.n_features_in_, feature_names)
            expr = printGPModel(self.best_model_, inputData=sym_vars)
            counts = []
            for v in sym_vars:
                try:
                    counts.append(float(expr.count(v)))
                except Exception:
                    counts.append(0.0)
        except Exception:
            counts = [0.0] * n
        total = sum(counts)
        if total == 0:
            return counts
        return [c / total for c in counts]


# ─── StackGP Ensemble Wrapper ─────────────────────────────────────────────────

class StackGPEnsembleRegressor:
    """Ensemble of GP models selected via StackGP's ``ensembleSelect``.

    After fitting, ``models_`` contains the selected individuals from the evolved
    population (targeting ``top_k`` members via cluster-based selection).
    Predictions are the mean across all members; uncertainty is the standard
    deviation, giving a natural measure of disagreement within the ensemble.
    """

    def __init__(self, generations=30, pop_size=100, time_limit=60, top_k=5, operator_set="default"):
        self.generations = int(generations)
        self.pop_size = int(pop_size)
        self.time_limit = int(time_limit)
        self.top_k = max(2, int(top_k))
        self.operator_set = "allOps" if str(operator_set) == "allOps" else "default"
        self.models_ = None
        self.n_features_in_ = None
        self.feature_names_ = None
        self._formula_cache_key = None
        self._member_formulas = []
        self.training_curve_ = []

    def _to_input_list(self, X):
        arr = np.array(X, dtype=float)
        return [arr[:, i] for i in range(arr.shape[1])]

    def fit(self, X, y):
        from .StackGP import evolve, allOps as _allOps, defaultOps as _defaultOps
        try:
            from .StackGP import ensembleSelect as _ensembleSelect
        except ImportError:
            _ensembleSelect = None
        if len(X) != len(y):
            print("Warning: Mismatch in number of samples between X and y. Attempting transpose.")
            X=X.T
        self.n_features_in_ = np.array(X).shape[1]
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        input_data = self._to_input_list(X)
        response = np.array(y, dtype=float)
        ops = _allOps() if self.operator_set == "allOps" else _defaultOps()
        all_models, self.training_curve_ = evolve(
            input_data,
            response,
            generations=self.generations,
            popSize=self.pop_size,
            ops=ops,
            capTime=True,
            timeLimit=self.time_limit,
            returnTracking=True,
        )
        candidate_models = all_models or []
        if _ensembleSelect and candidate_models:
            try:
                selected = _ensembleSelect(
                    candidate_models,
                    input_data,
                    response,
                    numberOfClusters=self.top_k,
                )
                self.models_ = (selected or [])[:self.top_k]
            except Exception as exc:
                print("StackGP ensembleSelect failed, falling back to top-k: %s", exc)
                self.models_ = candidate_models[:self.top_k]
        else:
            self.models_ = candidate_models[:self.top_k]
        if not self.models_:
            raise ValueError("StackGP returned no models.")
        self._build_formulas()
        return self

    def _build_formulas(self, feature_names=None):
        """Cache human-readable formulas for all ensemble members."""
        self._member_formulas = []
        try:
            from StackGP.StackGP import printGPModel
            from sympy import simplify
            names = feature_names if feature_names is not None else self.feature_names_
            self._formula_cache_key = tuple(map(str, names)) if names else None
            sym_vars = _stackgp_symbol_vars(self.n_features_in_, names)
            for m in self.models_:
                try:
                    expr = printGPModel(m, inputData=sym_vars)
                    try:
                        expr = simplify(expr)
                    except Exception:
                        pass
                    self._member_formulas.append(str(expr))
                except Exception:
                    self._member_formulas.append("N/A")
        except Exception:
            self._member_formulas = ["N/A"] * len(self.models_)

    def _predict_member(self, model, input_data):
        from .StackGP import evaluateGPModel
        return np.array(evaluateGPModel(model, input_data), dtype=float)

    def predict(self, X):
        """Return ensemble mean prediction."""
        input_data = self._to_input_list(X)
        preds = np.array([self._predict_member(m, input_data) for m in self.models_])
        return np.nanmean(preds, axis=0)

    def predict_with_uncertainty(self, X):
        """Return (mean, std) arrays across ensemble members."""
        input_data = self._to_input_list(X)
        preds = np.array([self._predict_member(m, input_data) for m in self.models_])
        mean = np.nanmean(preds, axis=0)
        std = np.nanstd(preds, axis=0)
        return mean, std, preds.tolist()

    def get_params(self, deep=True):
        return {
            "generations": self.generations,
            "pop_size": self.pop_size,
            "time_limit": self.time_limit,
            "top_k": self.top_k,
            "operator_set": self.operator_set,
        }

    def get_formula(self, feature_names=None):
        if feature_names is not None:
            requested_signature = tuple(map(str, feature_names))
            if requested_signature != self._formula_cache_key:
                self._build_formulas(feature_names)
        elif not self._member_formulas and self.models_:
            self._build_formulas(feature_names)
        if not self._member_formulas:
            return "N/A"
        lines = [f"Member {i+1}: {f}" for i, f in enumerate(self._member_formulas)]
        return "\n".join(lines)

    def get_feature_importance(self, feature_names=symbols(["x"+str(i) for i in range(100)])):
        """Approximate importance as mean normalised variable occurrence across members."""
        n = len(feature_names)
        if not self.models_:
            return [0.0] * n
        try:
            from .StackGP import printGPModel
            sym_vars = _stackgp_symbol_vars(self.n_features_in_, feature_names)
            all_counts = []
            for m in self.models_:
                try:
                    expr = printGPModel(m, inputData=sym_vars)
                    counts = []
                    for v in sym_vars:
                        try:
                            counts.append(float(str(expr).count(str(v))))
                        except Exception:
                            counts.append(0.0)
                    total = sum(counts)
                    if total > 0:
                        counts = [c / total for c in counts]
                    all_counts.append(counts)
                except Exception:
                    all_counts.append([0.0] * n)
            return list(np.mean(all_counts, axis=0))
        except Exception:
            return [0.0] * n