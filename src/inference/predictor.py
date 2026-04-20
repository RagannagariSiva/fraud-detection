"""
src/inference/predictor.py
==========================
Production inference wrapper for the fraud detection model.

This class is the single entry point for all scoring — whether called from
the FastAPI endpoint, the simulator, or a batch script. It handles:

  - Loading the trained model, scaler, and feature-name list from disk
  - Applying the same RobustScaler transformation used during training
  - Preserving the exact feature column order the model was fit on
  - Mapping raw probabilities to human-readable risk tiers
  - Threshold calibration via Youden's J statistic

Why this matters in production
-------------------------------
Scikit-learn models are sensitive to feature order. Two bugs that silently
destroy model accuracy are:

  1. Feeding unscaled Amount/Time at inference (training-serving skew)
  2. Passing features in dictionary iteration order instead of training order

Both are addressed here: the scaler is loaded from the same pkl that was
fitted during training, and features are always arranged using the saved
feature_names list before prediction.

Usage
-----
    from src.inference.predictor import FraudPredictor

    predictor = FraudPredictor()  # loads xgboost_model by default
    result = predictor.predict({"V1": -1.35, ..., "Amount": 149.62, "Time": 0.0})
    # {"prediction": "fraud", "probability": 0.923, "risk_tier": "CRITICAL", ...}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


# Risk tier boundaries: (upper_bound_exclusive, tier_label, recommended_action)
# Thresholds were calibrated on the Kaggle val set; adjust for your deployment.
_RISK_TIERS: list[tuple[float, str, str]] = [
    (0.15, "LOW", "Transaction appears normal. No action required."),
    (0.40, "MEDIUM", "Mildly suspicious. Consider a soft review."),
    (0.70, "HIGH", "Likely fraudulent. Manual review recommended."),
    (1.01, "CRITICAL", "High fraud probability. Block and alert immediately."),
]


class FraudPredictor:
    """
    Load a trained model and score transactions for fraud.

    Parameters
    ----------
    model_name : str
        File stem of the model pkl (e.g. ``"xgboost_model"``).
    model_dir : str
        Directory containing the model, scaler, and feature_names pkl files.
    threshold : float
        Probability cutoff for the "fraud" hard label. Default 0.40 is tuned
        for high recall on the Kaggle creditcard dataset.

    Raises
    ------
    FileNotFoundError
        If the model, scaler, or feature_names pkl files are missing.
        Run ``python main.py`` to train the model first.
    """

    def __init__(
        self,
        model_name: str = "xgboost_model",
        model_dir: str = "models",
        threshold: float = 0.40,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model_dir = Path(model_dir)
        self._scale_cols = ("Amount", "Time")

        self.model = self._load_artifact(f"{model_name}.pkl")
        self.scaler = self._load_artifact("scaler.pkl")
        self.feature_names: list[str] = self._load_artifact("feature_names.pkl")

        logger.info(
            "FraudPredictor ready | model=%s | features=%d | threshold=%.4f",
            model_name,
            len(self.feature_names),
            threshold,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, transaction: dict[str, float]) -> dict[str, Any]:
        """
        Score a single transaction and return a structured result.

        Parameters
        ----------
        transaction : dict[str, float]
            Feature name → value mapping. Must include all features in
            ``self.feature_names`` (V1–V28, Amount, Time, plus any engineered
            features added during training).

        Returns
        -------
        dict with keys:
            prediction    — ``"fraud"`` or ``"legitimate"``
            probability   — float [0, 1], the model's fraud confidence
            is_fraud      — bool
            risk_tier     — ``"LOW"`` / ``"MEDIUM"`` / ``"HIGH"`` / ``"CRITICAL"``
            threshold_used — the decision threshold applied
            message       — recommended action for this risk tier
        """
        feature_array = self._build_feature_array(transaction)
        probability = float(self.model.predict_proba(feature_array)[0, 1])
        is_fraud = probability >= self.threshold
        tier, message = _get_risk_tier(probability)

        result: dict[str, Any] = {
            "prediction": "fraud" if is_fraud else "legitimate",
            "probability": round(probability, 6),
            "is_fraud": is_fraud,
            "risk_tier": tier,
            "threshold_used": self.threshold,
            "message": message,
        }
        _log_prediction(result)
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame of transactions in a single vectorised call.

        Accepts a DataFrame with just the 30 raw features (V1-V28, Amount, Time)
        and applies scaling and feature engineering internally before scoring,
        matching what the training pipeline did.

        Preserves all original columns and appends ``probability``,
        ``prediction``, and ``risk_tier``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data with V1-V28, Amount, Time columns.
            Extra columns (e.g. a transaction ID) are kept but not passed to the model.

        Raises
        ------
        ValueError
            If the DataFrame is missing required raw features.
        """
        out = df.copy()

        # Scale Amount and Time with the training-time scaler.
        # Use explicit column order to match how the scaler was fitted.
        cols_in_order = [c for c in self._scale_cols if c in out.columns]
        if cols_in_order:
            out[cols_in_order] = self.scaler.transform(out[cols_in_order])

        # Apply the same feature engineering used during training
        try:
            from src.features.feature_engineering import build_features

            eng_cfg = {
                "add_time_features": True,
                "add_velocity_features": False,
                "add_interactions": True,
            }
            engineered = build_features(out, eng_cfg)
        except Exception as eng_err:
            logger.warning("Feature engineering skipped in batch mode: %s", eng_err)
            engineered = out

        missing = set(self.feature_names) - set(engineered.columns)
        if missing:
            raise ValueError(
                f"Batch DataFrame is missing required features after engineering: {sorted(missing)}"
            )

        probs = self.model.predict_proba(engineered[self.feature_names].values)[:, 1]

        # Add predictions back to the original dataframe (preserves caller's columns)
        out["probability"] = probs
        out["prediction"] = np.where(probs >= self.threshold, "fraud", "legitimate")
        out["risk_tier"] = [_get_risk_tier(p)[0] for p in probs]
        return out

    def set_threshold_youden(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """
        Find the decision threshold that maximises Youden's J statistic.

        Youden's J = Sensitivity + Specificity − 1, which equals the vertical
        distance from the ROC curve to the diagonal. The optimal threshold
        balances recall and specificity without favouring either metric.

        This should be called on a held-out *validation* set after training,
        before the test set is ever evaluated.

        Parameters
        ----------
        X_val : np.ndarray
            Validation features (already scaled and column-aligned).
        y_val : np.ndarray
            True binary labels for the validation set.

        Returns
        -------
        float
            The calibrated threshold. Also updates ``self.threshold`` in place.
        """
        probs = self.model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, probs)

        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(thresholds[best_idx])

        logger.info(
            "Threshold calibrated via Youden's J: %.4f  (Recall=%.4f, Specificity=%.4f)",
            best_threshold,
            tpr[best_idx],
            1 - fpr[best_idx],
        )
        self.threshold = best_threshold
        return best_threshold

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_artifact(self, filename: str) -> Any:
        path = self._model_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}\nRun 'python main.py' to train the model."
            )
        return joblib.load(path)

    def _build_feature_array(self, transaction: dict[str, float]) -> np.ndarray:
        """
        Convert a raw transaction dict to a (1, n_features) numpy array.

        Steps
        -----
        1. Scale Amount/Time using the fitted RobustScaler.
        2. Build a single-row DataFrame.
        3. Apply the same feature engineering used during training
           (time features, interaction terms, etc.).
        4. Select columns in the exact training-time order.

        Callers pass the 30 raw features (V1-V28, Amount, Time). The predictor
        handles feature engineering internally so the API contract stays simple
        and consistent with what the model was actually trained on.
        """
        t = dict(transaction)

        # Scale Amount and Time — must preserve training-column order for the scaler
        cols_to_scale = [c for c in self._scale_cols if c in t]
        if cols_to_scale:
            full_raw = pd.DataFrame(
                [[t.get(col, 0.0) for col in self._scale_cols]],
                columns=list(self._scale_cols)
            )
            scaled_full = self.scaler.transform(full_raw)[0]
            for i, col in enumerate(self._scale_cols):
                if col in t:
                    t[col] = float(scaled_full[i])

        # Verify we have the raw base features before running engineering
        raw_base = [f"V{i}" for i in range(1, 29)] + list(self._scale_cols)
        missing_raw = set(raw_base) - set(t.keys())
        if missing_raw:
            raise ValueError(f"Transaction is missing required raw features: {sorted(missing_raw)}")

        # Apply the same feature engineering pipeline used during training.
        # This is what closes the training-serving gap: the predictor and the
        # training pipeline run identical transformations.
        try:
            from src.features.feature_engineering import build_features

            row_df = pd.DataFrame([t])
            eng_cfg = {
                "add_time_features": True,
                "add_velocity_features": False,  # requires sorted multi-row data
                "add_interactions": True,
            }
            row_df = build_features(row_df, eng_cfg)
            t = row_df.iloc[0].to_dict()
        except Exception as eng_err:
            logger.warning(
                "Feature engineering failed at inference (proceeding with raw features): %s",
                eng_err,
            )

        missing = set(self.feature_names) - set(t.keys())
        if missing:
            raise ValueError(
                f"Transaction is missing features after engineering: {sorted(missing)}"
            )

        return np.array([t[f] for f in self.feature_names]).reshape(1, -1)


# ── Module-level helpers ───────────────────────────────────────────────────────


def _get_risk_tier(probability: float) -> tuple[str, str]:
    """Map a fraud probability to a (tier_label, action_message) pair."""
    for upper, tier, message in _RISK_TIERS:
        if probability < upper:
            return tier, message
    # Fallback — shouldn't be reached unless probability > 1.0
    return _RISK_TIERS[-1][1], _RISK_TIERS[-1][2]


def _log_prediction(result: dict[str, Any]) -> None:
    """Write a structured prediction result to the logger and stdout."""
    flag = "🚨 FRAUD" if result["is_fraud"] else "✅ LEGIT"
    logger.info(
        "%s | prob=%.4f | tier=%-8s | threshold=%.4f",
        flag,
        result["probability"],
        result["risk_tier"],
        result["threshold_used"],
    )
