import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_final_configs as mfc  # noqa: E402


def _finish(exp_dir, name, auc):
    (exp_dir / name).mkdir(parents=True, exist_ok=True)
    (exp_dir / name / "summary.json").write_text(json.dumps({"mean_auc": auc}))


def test_stages_pick_best_on_val(tmp_path, monkeypatch):
    monkeypatch.setattr(mfc, "OUT", tmp_path / "final")
    monkeypatch.setattr(mfc.cfg, "EXPERIMENTS_DIR", tmp_path / "exp")
    run = lambda stage: (monkeypatch.setattr(sys, "argv", ["x", "--stage", str(stage)]), mfc.main())
    run(1)
    for i, w in enumerate(mfc.cfg.MIP_GRID_WINDOWS):  # top60 best, center90 second
        for s in ("", "_s1", "_s2"):  # top120 wins one seed but loses on the mean
            auc = {"top60": 0.8, "center90": 0.75}.get(w, 0.6 - i / 100)
            _finish(tmp_path / "exp", f"2D-1_{w}{s}", 0.95 if (w, s) == ("top120", "_s1") else auc)
    run(2)
    names = {p.stem for p in (tmp_path / "final").glob("2D-2_*")}
    assert names == {f"2D-2_{v}_{w}{s}" for v in ("axial", "latefusion", "pad")
                     for w in ("top60", "center90") for s in ("", "_s1", "_s2")}
    assert json.loads((tmp_path / "final" / "2D-2_pad_top60_s2.json").read_text())[0]["RANDOM_SEED"] == 2
    for n in names | {f"2D-1b_n4_top60{s}" for s in ("", "_s1", "_s2")}:
        _finish(tmp_path / "exp", n, 0.9 if n.startswith("2D-2_pad_center90") else 0.5)
    run(3)
    c = json.loads((tmp_path / "final" / "2D-3_512.json").read_text())[0]
    assert c["data_path_key"] == "mip_grid/center90_pad_512" and c["BATCH_SIZE"] == 18
    mixed = json.loads((tmp_path / "final" / "2D-6_mixed.json").read_text())[0]
    assert mixed["data_path_key"] == "mip_grid_mixed/center90_pad_384"
    laus = json.loads((tmp_path / "final" / "2D-6_lausanne_only.json").read_text())[0]
    assert laus["TEST_IDS_FILE"] is None and laus["EXCLUSIONS_FILE"] == "dataset/exclusions.json"
    assert not list((tmp_path / "final").glob("2D-7_*"))
