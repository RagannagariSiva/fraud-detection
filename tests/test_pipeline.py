"""
tests/test_pipeline.py
=======================
Integration tests for the full training pipeline.

These tests run the complete pipeline end-to-end on a tiny synthetic dataset
to verify that all phases produce the expected artifacts and that the trained
predictor can score a transaction.  They are slower than unit tests but catch
integration failures that unit tests miss.

Run with:
    pytest tests/test_pipeline.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def tmp_project(tmp_path_factory) -> dict:
    """
    Build a minimal project layout in a temp directory and return path references.
    """
    base = tmp_path_factory.mktemp("fraud_project")
    dirs = ["models", "reports/figures", "data/raw", "data/processed", "logs"]
    for d in dirs:
        (base / d).mkdir(parents=True)
    return {
        "base": base,
        "models": base / "models",
        "reports": base / "reports" / "figures",
        "data": base / "data" / "raw",
        "logs": base / "logs",
    }


def _minimal_config(paths: dict) -> dict:
    """Return a config dict that runs fast enough for tests."""
    return {
        "data": {
            "raw_path": str(paths["data"] / "creditcard.csv"),
            "processed_path": str(paths["base"] / "data" / "processed" / "data.csv"),
            "test_size": 0.20,
            "val_size": 0.10,
            "random_state": 42,
            "fraud_ratio": 0.05,
        },
        "preprocessing": {
            "scaler": "robust",
            "scale_cols": ["Amount", "Time"],
            "scaler_path": str(paths["models"] / "scaler.pkl"),
            "feature_names_path": str(paths["models"] / "feature_names.pkl"),
        },
        "features": {
            "add_time_features": True,
            "add_velocity_features": False,
            "add_interactions": True,
        },
        "resampling": {
            "strategy": "smote",
            "smote_k_neighbors": 3,
        },
        "models": {
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 4,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "n_jobs": 1,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 20,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "eval_metric": "aucpr",
                "n_jobs": 1,
                "verbosity": 0,
                "random_state": 42,
            },
        },
        "tuning": {"enabled": False},
        "training": {
            "primary_model": "xgboost",
            "model_dir": str(paths["models"]),
            "report_dir": str(paths["reports"]),
            "results_path": str(paths["base"] / "reports" / "model_results.csv"),
            "log_path": str(paths["logs"] / "training.log"),
        },
        "inference": {
            "model_name": "xgboost_model",
            "model_dir": str(paths["models"]),
            "threshold": 0.40,
        },
        "mlflow": {"enabled": False},
        "drift_detection": {
            "baseline_path": str(paths["models"] / "drift_baseline.json"),
        },
    }


class TestPipelineArtifacts:
    """Verify the pipeline produces all expected artifacts."""

    @pytest.fixture(scope="class")
    def pipeline_result(self, tmp_project):
        """Run the pipeline once for the class; cache the result."""
        from src.training.pipeline import run_pipeline

        cfg = _minimal_config(tmp_project)
        df_results = run_pipeline(cfg)
        return {"cfg": cfg, "results": df_results, "paths": tmp_project}

    def test_returns_dataframe_with_model_rows(self, pipeline_result):
        df = pipeline_result["results"]
        assert len(df) >= 1, "Expected at least one model row in results"
        assert "PR_AUC" in df.columns

    def test_scaler_pkl_exists(self, pipeline_result):
        path = Path(pipeline_result["cfg"]["preprocessing"]["scaler_path"])
        assert path.exists(), f"scaler.pkl not found at {path}"

    def test_feature_names_pkl_exists(self, pipeline_result):
        path = Path(pipeline_result["cfg"]["preprocessing"]["feature_names_path"])
        assert path.exists(), f"feature_names.pkl not found at {path}"

    def test_xgboost_model_pkl_exists(self, pipeline_result):
        models_dir = Path(pipeline_result["cfg"]["training"]["model_dir"])
        assert (models_dir / "xgboost_model.pkl").exists()

    def test_random_forest_model_pkl_exists(self, pipeline_result):
        models_dir = Path(pipeline_result["cfg"]["training"]["model_dir"])
        assert (models_dir / "random_forest_model.pkl").exists()

    def test_xgboost_metadata_json_exists(self, pipeline_result):
        models_dir = Path(pipeline_result["cfg"]["training"]["model_dir"])
        meta = models_dir / "xgboost_model_metadata.json"
        assert meta.exists()
        with open(meta) as f:
            data = json.load(f)
        assert "val_pr_auc" in data
        assert float(data["val_pr_auc"]) > 0.0

    def test_drift_baseline_json_exists(self, pipeline_result):
        path = Path(pipeline_result["cfg"]["drift_detection"]["baseline_path"])
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "baseline_stats" in data
        assert len(data["baseline_stats"]) > 0

    def test_model_results_csv_exists(self, pipeline_result):
        path = Path(pipeline_result["cfg"]["training"]["results_path"])
        assert path.exists()

    def test_pr_auc_above_random_baseline(self, pipeline_result):
        """The model must meaningfully outperform a random classifier."""
        df = pipeline_result["results"]
        xgb_row = df[df["Model"].str.contains("XGBoost", case=False)]
        if not xgb_row.empty:
            pr_auc = float(xgb_row["PR_AUC"].iloc[0])
            # Synthetic dataset has 5% fraud → random PR-AUC ≈ 0.05
            assert pr_auc > 0.30, (
                f"XGBoost PR-AUC={pr_auc:.4f} is suspiciously low on synthetic data"
            )


class TestPredictorAfterPipeline:
    """Verify FraudPredictor loads and scores correctly after a pipeline run."""

    @pytest.fixture(scope="class")
    def predictor(self, tmp_project):
        # Run pipeline first
        from src.training.pipeline import run_pipeline

        cfg = _minimal_config(tmp_project)
        run_pipeline(cfg)

        from src.inference.predictor import FraudPredictor

        return FraudPredictor(
            model_name="xgboost_model",
            model_dir=str(tmp_project["models"]),
            threshold=0.40,
        )

    def test_predictor_loads_without_error(self, predictor):
        assert predictor is not None
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert len(predictor.feature_names) > 0

    def test_predict_returns_valid_structure(self, predictor, valid_transaction_dict):
        result = predictor.predict(valid_transaction_dict)
        assert "prediction" in result
        assert "probability" in result
        assert "risk_tier" in result
        assert "threshold_used" in result
        assert result["prediction"] in ("fraud", "legitimate")
        assert 0.0 <= result["probability"] <= 1.0
        assert result["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_predict_batch_returns_dataframe(self, predictor, valid_transaction_dict):
        import pandas as pd

        df = pd.DataFrame([valid_transaction_dict] * 5)
        scored = predictor.predict_batch(df)
        assert "probability" in scored.columns
        assert "prediction" in scored.columns
        assert "risk_tier" in scored.columns
        assert len(scored) == 5

    def test_high_v14_negative_scores_higher(self, predictor, valid_transaction_dict):
        """
        V14 is the strongest fraud signal: large negative values should push
        the fraud probability higher than V14 near zero.
        """
        base = dict(valid_transaction_dict)
        fraud_tx = {**base, "V14": -15.0, "V12": -12.0, "V17": -10.0, "V10": -10.0}
        legit_tx = {**base, "V14":   2.0, "V12":   2.0, "V17":   2.0, "V10":   2.0}

        fraud_prob = predictor.predict(fraud_tx)["probability"]
        legit_prob = predictor.predict(legit_tx)["probability"]
        # Allow tiny floating-point ties on synthetic data by requiring >= not >
        assert fraud_prob >= legit_prob, (
            f"Expected fraud_prob ({fraud_prob:.4f}) >= legit_prob ({legit_prob:.4f})"
        )


class TestFeatureEngineeringConsistency:
    """Verify feature engineering applies identically at training and inference time."""

    def test_log_amount_positive(self):
        import pandas as pd

        from src.features.feature_engineering import _add_time_features

        df = pd.DataFrame({"Amount": [0.0, 1.5, 100.0, 25519.0], "Time": [0.0] * 4})
        result = _add_time_features(df)
        assert (result["log_amount"] >= 0).all(), "log_amount must be non-negative"

    def test_is_night_correct_boundary(self):
        import pandas as pd

        from src.features.feature_engineering import _add_time_features

        # 2am (hour=2) → night; 12pm (hour=12) → day
        df = pd.DataFrame(
            {
                "Amount": [50.0, 50.0],
                "Time": [2 * 3600, 12 * 3600],  # 02:00 and 12:00
            }
        )
        result = _add_time_features(df)
        assert result["is_night"].iloc[0] == 1, "02:00 should be classified as night"
        assert result["is_night"].iloc[1] == 0, "12:00 should not be classified as night"

    def test_feature_count_matches_after_engineering(self):
        import pandas as pd

        from src.features.feature_engineering import build_features

        df = pd.DataFrame({f"V{i}": [0.0] for i in range(1, 29)})
        df["Amount"] = [100.0]
        df["Time"] = [3600.0]

        cfg = {"add_time_features": True, "add_interactions": True}
        result = build_features(df, cfg)
        assert result.shape[1] > df.shape[1], "Feature engineering should add columns"
