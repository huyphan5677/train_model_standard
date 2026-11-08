"""CatBoost helpers: weighted ensemble, transfer learning (fine-tune), shrinking to the best tree count."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier, sum_models
from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Soft-voting ensemble of pretrained CatBoost models (optionally weighted)."""

    def __init__(self, models: Sequence[CatBoostClassifier], weights: Sequence[float] | None = None):
        self.models = models
        self.weights = weights

    def _norm_weights(self) -> np.ndarray:
        w = np.ones(len(self.models)) if self.weights is None else np.asarray(self.weights, dtype=float)
        return w / w.sum()

    def predict_proba(self, X) -> np.ndarray:
        probas = np.array([m.predict_proba(X) for m in self.models])
        return np.tensordot(self._norm_weights(), probas, axes=(0, 0))

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save_model(self, path: str | Path) -> None:
        joblib.dump({"models": list(self.models), "weights": self.weights}, path)

    @classmethod
    def load_model(cls, path: str | Path) -> CatBoostEnsembleClassifier:
        data = joblib.load(path)
        return cls(data["models"], data["weights"])

    def merge_to_single_model(self) -> CatBoostClassifier:
        """Merge into one CatBoost model with ``catboost.sum_models`` (models must share features).

        Note: summing raw scores is not identical to averaging probabilities; validate the result.
        """
        return sum_models(list(self.models), weights=list(self._norm_weights()))


class CatBoostTransferLearner:
    """Continue training a pretrained CatBoost model on new data (``init_model``)."""

    def __init__(self, model: CatBoostClassifier):
        self.model = model

    def fine_tune(self, X, y, iterations: int = 100, learning_rate: float = 0.01, **fit_kwargs) -> CatBoostClassifier:
        params = self.model.get_params()
        params.update(iterations=iterations, learning_rate=learning_rate, verbose=False)
        params.pop("early_stopping_rounds", None)
        tuned = CatBoostClassifier(**params)
        tuned.fit(X, y, init_model=self.model, use_best_model=False, **fit_kwargs)
        self.model = tuned
        return tuned

    def retrain_with_same_params(self, X, y, **fit_kwargs) -> CatBoostClassifier:
        params = {**self.model.get_params(), "verbose": False}
        return CatBoostClassifier(**params).fit(X, y, **fit_kwargs)

    def save(self, model_path: str | Path = "transfer_model.cbm", wrapper_path: str | Path = "transfer.json") -> None:
        self.model.save_model(str(model_path))
        Path(wrapper_path).write_text(json.dumps({"model_path": str(Path(model_path).resolve())}), encoding="utf-8")

    @staticmethod
    def load(wrapper_path: str | Path = "transfer.json") -> CatBoostTransferLearner:
        info = json.loads(Path(wrapper_path).read_text(encoding="utf-8"))
        model = CatBoostClassifier()
        model.load_model(info["model_path"])
        return CatBoostTransferLearner(model)


class CatBoostShrinker:
    """Keep only the first N trees, where N maximizes (or minimizes) a validation metric."""

    def __init__(self, model: CatBoostClassifier):
        self.model = model
        self.best_ntree_end: int | None = None
        self.scores_: list[tuple[int, float]] = []
        self.shrunk_model: CatBoostClassifier | None = None

    def shrink_by_validation(
        self,
        X_val,
        y_val,
        metric_fn: Callable,
        step: int = 10,
        maximize: bool = True,
    ) -> CatBoostClassifier:
        n_trees = self.model.tree_count_
        candidates = sorted(set(range(step, n_trees + 1, step)) | {n_trees})
        self.scores_ = [
            (n, metric_fn(y_val, self.model.predict(X_val, prediction_type="Probability", ntree_end=n)[:, 1]))
            for n in candidates
        ]
        pick = max if maximize else min
        self.best_ntree_end = pick(self.scores_, key=lambda t: t[1])[0]
        self.shrunk_model = copy.deepcopy(self.model)
        self.shrunk_model.shrink(ntree_end=self.best_ntree_end)
        return self.shrunk_model

    def get_shrunk_model(self) -> CatBoostClassifier:
        if self.shrunk_model is None:
            raise ValueError("Call shrink_by_validation first")
        return self.shrunk_model

    def save(self, path: str | Path) -> None:
        self.get_shrunk_model().save_model(str(path))

    def load(self, path: str | Path) -> CatBoostClassifier:
        self.shrunk_model = CatBoostClassifier()
        self.shrunk_model.load_model(str(path))
        return self.shrunk_model
