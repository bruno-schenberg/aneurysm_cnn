"""Contract of a final run, end to end on synthetic data.

- EVAL_ON_TEST keeps val predictions.csv and adds test_predictions.csv + a
  "test" block in summary.json
- SPLIT_SEED fixes folds/test while RANDOM_SEED changes
- TEST_IDS_FILE aborts the run when the test split differs
- WARMUP_EPOCHS ramps the LR before the cosine schedule

Run: cd training_engine && python -m pytest tests/test_final_run_contract.py -q
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch.nn as nn

import config as cfg
from dlorchestra.training import build_scheduler
from src import models
from src.experiment import AneurysmExperiment

N_PER_CLASS = 20


def _tiny(num_classes: int, **_) -> nn.Module:
    m = nn.Sequential()
    m.add_module("body", nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten()))
    m.add_module("fc", nn.Linear(4, num_classes))
    return m


@pytest.fixture(scope="module", autouse=True)
def tiny_model():
    if "_TINY_2D" not in models._REGISTRY:
        models._reg("_TINY_2D", _tiny, "fc")


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for label in (0, 1):
        (tmp_path / "data" / str(label)).mkdir(parents=True)
        for i in range(N_PER_CLASS):
            img = rng.normal(size=(3, 32, 32)).astype(np.float32) + label
            np.save(tmp_path / "data" / str(label) / f"BP{label * 100 + i:04d}.npy", img)
    monkeypatch.setitem(cfg.DATASET_PATHS, "_tiny", tmp_path / "data")
    return tmp_path


def _config(name, **over):
    return {**cfg.DEFAULT_CONFIG, "name": name, "model": "_TINY_2D", "data_path_key": "_tiny",
            "SPATIAL_DIMS": 2, "EPOCHS": 2, "BATCH_SIZE": 4, "NUM_WORKERS": 0, "CACHE_RATE": 0.0,
            "USE_AMP": False, "USE_KFOLD": True, "N_SPLITS": 2, "EVAL_ON_TEST": True,
            "balancing": "oversampling", "CHECKPOINT_METRIC": "val_auc", **over}


def _ids(csv_path):
    return set(pd.read_csv(csv_path).image.str.extract(r"(BP\d{4})")[0])


def _run(tmp_path, **over):
    name = over.pop("name", "run")
    AneurysmExperiment(experiments_dir=tmp_path / "exp", device="cpu").run(_config(name, **over))
    return tmp_path / "exp" / name


def test_eval_on_test_keeps_val_and_adds_test(dataset):
    d = _run(dataset)
    summary = json.loads((d / "summary.json").read_text())
    assert summary["num_folds"] == 2 and "test" in summary
    assert set(summary["test"]["per_fold"]) == {"fold_1", "fold_2"}
    test_ids = _ids(d / "fold_1/test_predictions.csv")
    assert test_ids == _ids(d / "fold_2/test_predictions.csv")
    val = [_ids(d / f"fold_{k}/predictions.csv") for k in (1, 2)]
    assert not (val[0] | val[1]) & test_ids, "val predictions must not contain test exams"
    assert val[0].isdisjoint(val[1])


def test_split_seed_fixes_folds_across_training_seeds(dataset):
    a = _run(dataset, name="a", RANDOM_SEED=1, SPLIT_SEED=42)
    b = _run(dataset, name="b", RANDOM_SEED=2, SPLIT_SEED=42)
    c = _run(dataset, name="c", RANDOM_SEED=42)
    for f in ("fold_1/predictions.csv", "fold_1/test_predictions.csv"):
        assert _ids(a / f) == _ids(b / f) == _ids(c / f)


def test_test_ids_guard(dataset, monkeypatch):
    d = _run(dataset, name="ref")
    good = dataset / "good.json"
    good.write_text(json.dumps(sorted(_ids(d / "fold_1/test_predictions.csv"))))
    monkeypatch.setattr("src.experiment.REPO_ROOT", dataset)
    _run(dataset, name="ok", TEST_IDS_FILE="good.json")
    bad = dataset / "bad.json"
    bad.write_text(json.dumps(["BP9999"]))
    with pytest.raises(ValueError, match="Test split differs"):
        _run(dataset, name="bad", TEST_IDS_FILE="bad.json")


def test_warmup_then_cosine():
    import torch
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = build_scheduler(opt, {**cfg.DEFAULT_CONFIG, "WARMUP_EPOCHS": 3, "COSINE_T0": 10})
    lrs = []
    for _ in range(5):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.01) and lrs[0] < lrs[1] < lrs[2] < lrs[3]
    assert lrs[3] == pytest.approx(1.0)  # cosine starts at the base LR
    assert build_scheduler(opt, {**cfg.DEFAULT_CONFIG}).__class__.__name__ == "CosineAnnealingWarmRestarts"


def test_multimodal_2d_end_to_end(dataset, monkeypatch):
    """MIP + age/sex (2D-5) must run: tabular survives the 2D transforms and
    the 2D Sequential head is stripped correctly for late fusion."""
    csv_path = dataset / "meta.csv"
    rows = ["exam,class,location,Age,gender,brand,scanner,field_strength_T"]
    for f in sorted((dataset / "data").rglob("*.npy")):
        rows.append(f"{f.stem},{f.parent.name},,50,1,siemens,skyra,3.0")
    csv_path.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(cfg, "DATASET_CSV", csv_path)
    d = _run(dataset, name="mm", USE_MULTIMODAL=True, FREEZE_BACKBONE_EPOCHS=1)
    assert json.loads((d / "summary.json").read_text())["test"]["mean_auc"] is not None


def test_predict_reproduces_test_predictions(dataset):
    """analysis/predict.py on the saved checkpoints must give the same test
    probabilities the run itself wrote (same model, weights, eval transforms)."""
    from analysis.predict import predict
    d = _run(dataset, name="pred")
    run = pd.concat(pd.read_csv(d / f"fold_{k}/test_predictions.csv").assign(fold=k) for k in (1, 2))
    test_ids = set(run.image.str.extract(r"(BP\d{4})")[0])
    out = predict(d, ids=test_ids).set_index("image")
    for k in (1, 2):
        r = run[run.fold == k].set_index("image").prob
        assert np.allclose(out.loc[r.index, f"prob_fold_{k}"], r, atol=1e-5)
    assert np.allclose(out.prob_mean, out[["prob_fold_1", "prob_fold_2"]].mean(axis=1))


def test_exclusions_and_patient_groups(dataset, monkeypatch):
    """EXCLUSIONS_FILE: excluded exams appear nowhere,
    a patient's exams share a fold, and the test set is exactly TEST_IDS_FILE."""
    monkeypatch.setattr("src.experiment.REPO_ROOT", dataset)
    monkeypatch.setattr("src.split.REPO", dataset)
    test = ["BP0000", "BP0001", "BP0002", "BP0100", "BP0101", "BP0102"]
    (dataset / "test.json").write_text(json.dumps(test))
    (dataset / "excl.json").write_text(json.dumps({
        "exclude": {"BP0003": "dup", "BP0103": "dup"},
        "patient_groups": [["BP0004", "BP0005", "BP0104"], ["BP0010", "BP0110"]],
    }))
    d = _run(dataset, name="excl", EXCLUSIONS_FILE="excl.json", TEST_IDS_FILE="test.json")
    val = [_ids(d / f"fold_{k}/predictions.csv") for k in (1, 2)]
    assert _ids(d / "fold_1/test_predictions.csv") == set(test)
    assert val[0] | val[1] == {f"BP{c * 100 + i:04d}" for c in (0, 1) for i in range(N_PER_CLASS)} \
        - set(test) - {"BP0003", "BP0103"}
    for group in ({"BP0004", "BP0005", "BP0104"}, {"BP0010", "BP0110"}):
        assert group <= val[0] or group <= val[1]
