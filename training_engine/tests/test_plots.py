"""
Tests for plots.py — metric calculation and artifact saving.

All tests run without GPU, live models, DataLoaders, or real NIfTI files.
Metric tests use synthetic prediction records with known expected values.
Artifact tests write to pytest's tmp_path fixture and verify file contents.
"""

import csv
import math
import os

import pandas as pd
import pytest

from src.plots import calculate_classification_metrics, save_metrics_to_csv, save_predictions_from_detailed_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prediction_record(filename: str, true_label: int, pred_label: int) -> dict:
    """Construct a minimal prediction record dict."""
    return {"filename": filename, "true_label": true_label, "pred_label": pred_label}


# ---------------------------------------------------------------------------
# calculate_classification_metrics
# ---------------------------------------------------------------------------

class TestCalculateClassificationMetrics:
    """Unit tests for calculate_classification_metrics."""

    def test_known_values(self):
        """
        Synthetic predictions with a known confusion matrix.

        Truth:      [1, 1, 1, 0]
        Predicted:  [1, 1, 0, 0]

        TP=2, FN=1, FP=0, TN=1
        Precision = TP / (TP + FP) = 2/2 = 1.0
        Recall    = TP / (TP + FN) = 2/3 ≈ 0.667
        F2        = 5 * P * R / (4*P + R) = 5*1*(2/3)/(4*1 + 2/3) ≈ 0.556
        """
        results = [
            make_prediction_record("a.nii.gz", true_label=1, pred_label=1),
            make_prediction_record("b.nii.gz", true_label=1, pred_label=1),
            make_prediction_record("c.nii.gz", true_label=1, pred_label=0),
            make_prediction_record("d.nii.gz", true_label=0, pred_label=0),
        ]
        metrics = calculate_classification_metrics(results)

        assert metrics["Precision"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["Recall"] == pytest.approx(2 / 3, abs=1e-6)
        # F2 = 5*P*R / (4P + R)
        expected_f2 = 5 * 1.0 * (2 / 3) / (4 * 1.0 + 2 / 3)
        assert metrics["F2-Score"] == pytest.approx(expected_f2, abs=1e-4)

    def test_perfect_predictions(self):
        """All correct predictions → Precision=1, Recall=1, F2=1."""
        results = [
            make_prediction_record("a.nii.gz", 1, 1),
            make_prediction_record("b.nii.gz", 0, 0),
        ]
        metrics = calculate_classification_metrics(results)
        assert metrics["Precision"] == pytest.approx(1.0)
        assert metrics["Recall"] == pytest.approx(1.0)
        assert metrics["F2-Score"] == pytest.approx(1.0)

    def test_single_class_true_labels_returns_zeros(self):
        """
        If all true labels are the same class, metrics cannot be computed
        meaningfully — function must return 0.0 for all values without raising.
        """
        results = [
            make_prediction_record("a.nii.gz", true_label=1, pred_label=0),
            make_prediction_record("b.nii.gz", true_label=1, pred_label=1),
        ]
        metrics = calculate_classification_metrics(results)

        assert metrics["Precision"] == 0.0
        assert metrics["Recall"] == 0.0
        assert metrics["F2-Score"] == 0.0

    def test_empty_list_returns_nan(self):
        """Empty detailed_results → all metrics are NaN, no exception raised."""
        metrics = calculate_classification_metrics([])

        assert math.isnan(metrics["Precision"])
        assert math.isnan(metrics["Recall"])
        assert math.isnan(metrics["F2-Score"])

    def test_returns_three_keys(self):
        """Return value always has exactly the three expected keys."""
        results = [make_prediction_record("a.nii.gz", 1, 1)]
        metrics = calculate_classification_metrics(results)
        assert set(metrics.keys()) == {"Precision", "Recall", "F2-Score"}


# ---------------------------------------------------------------------------
# save_metrics_to_csv
# ---------------------------------------------------------------------------

class TestSaveMetricsToCsv:
    """Unit tests for save_metrics_to_csv."""

    def _make_metrics_history(self) -> list:
        return [
            {"epoch": 0, "train_loss": 0.8, "val_loss": 0.75, "val_acc": 0.60},
            {"epoch": 1, "train_loss": 0.6, "val_loss": 0.55, "val_acc": 0.72},
        ]

    def _make_detailed_results(self) -> list:
        return [
            make_prediction_record("a.nii.gz", 1, 1),
            make_prediction_record("b.nii.gz", 0, 0),
            make_prediction_record("c.nii.gz", 1, 0),
        ]

    def test_csv_is_created(self, tmp_path):
        """The function must create the CSV file."""
        csv_path = str(tmp_path / "metrics.csv")
        save_metrics_to_csv(self._make_metrics_history(), csv_path, self._make_detailed_results())
        assert os.path.exists(csv_path)

    def test_csv_row_count(self, tmp_path):
        """CSV must have one row per epoch plus one final_eval row."""
        csv_path = str(tmp_path / "metrics.csv")
        history = self._make_metrics_history()
        save_metrics_to_csv(history, csv_path, self._make_detailed_results())

        df = pd.read_csv(csv_path)
        # 2 epoch rows + 1 final_eval row
        assert len(df) == len(history) + 1

    def test_final_eval_row_present(self, tmp_path):
        """The last row must have epoch == 'final_eval'."""
        csv_path = str(tmp_path / "metrics.csv")
        save_metrics_to_csv(self._make_metrics_history(), csv_path, self._make_detailed_results())

        df = pd.read_csv(csv_path)
        assert str(df.iloc[-1]["epoch"]) == "final_eval"

    def test_final_eval_row_has_metric_columns(self, tmp_path):
        """The final_eval row must contain Precision, Recall, and F2-Score columns."""
        csv_path = str(tmp_path / "metrics.csv")
        save_metrics_to_csv(self._make_metrics_history(), csv_path, self._make_detailed_results())

        df = pd.read_csv(csv_path)
        final_row = df.iloc[-1]
        assert "Precision" in df.columns
        assert "Recall" in df.columns
        assert "F2-Score" in df.columns
        # Values must be numeric (not NaN) for this synthetic data
        assert not math.isnan(float(final_row["Precision"]))

    def test_empty_history_does_not_create_file(self, tmp_path):
        """Empty metrics_history → no file written (function returns early)."""
        csv_path = str(tmp_path / "metrics.csv")
        save_metrics_to_csv([], csv_path, self._make_detailed_results())
        assert not os.path.exists(csv_path)


# ---------------------------------------------------------------------------
# save_predictions_from_detailed_results
# ---------------------------------------------------------------------------

class TestSavePredictionsFromDetailedResults:
    """Unit tests for save_predictions_from_detailed_results."""

    def _make_results(self) -> list:
        return [
            make_prediction_record("case001.nii.gz", true_label=1, pred_label=1),
            make_prediction_record("case002.nii.gz", true_label=0, pred_label=0),
            make_prediction_record("case003.nii.gz", true_label=1, pred_label=0),
        ]

    def test_csv_is_created(self, tmp_path):
        """The function must create the CSV file at the given path."""
        csv_path = str(tmp_path / "predictions.csv")
        save_predictions_from_detailed_results(self._make_results(), csv_path)
        assert os.path.exists(csv_path)

    def test_csv_has_correct_header(self, tmp_path):
        """CSV header must be exactly file_name, true_label, prediction."""
        csv_path = str(tmp_path / "predictions.csv")
        save_predictions_from_detailed_results(self._make_results(), csv_path)

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ["file_name", "true_label", "prediction"]

    def test_csv_row_count(self, tmp_path):
        """CSV must have one data row per prediction record."""
        results = self._make_results()
        csv_path = str(tmp_path / "predictions.csv")
        save_predictions_from_detailed_results(results, csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == len(results)

    def test_csv_values_are_correct(self, tmp_path):
        """CSV values must match the input prediction records exactly."""
        results = self._make_results()
        csv_path = str(tmp_path / "predictions.csv")
        save_predictions_from_detailed_results(results, csv_path)

        df = pd.read_csv(csv_path)
        assert df.iloc[0]["file_name"] == "case001.nii.gz"
        assert df.iloc[0]["true_label"] == 1
        assert df.iloc[0]["prediction"] == 1

        assert df.iloc[2]["file_name"] == "case003.nii.gz"
        assert df.iloc[2]["true_label"] == 1
        assert df.iloc[2]["prediction"] == 0

    def test_empty_results_writes_header_only(self, tmp_path):
        """Empty results list → CSV exists with header row only, no data rows."""
        csv_path = str(tmp_path / "predictions.csv")
        save_predictions_from_detailed_results([], csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 0
        assert list(df.columns) == ["file_name", "true_label", "prediction"]
