import numpy as np

from analysis.thesis_stats import dev_max_f2_threshold, dev_threshold, location_group


def test_max_f2_prefers_recall_over_precision():
    # t=2 flags 2 TP + 0 FP of 3 positives (F2 0.71); t=1 flags 3 TP + 1 FP (F2 0.94)
    y = np.array([1, 1, 1, 0, 0])
    assert dev_max_f2_threshold(y, np.array([2.0, 2.0, 1.0, 1.0, 0.0])) == 1.0


def test_threshold_keeps_target_sensitivity():
    y = np.array([1] * 10 + [0] * 5)
    score = np.r_[np.arange(1.0, 11.0), np.zeros(5)]
    t = dev_threshold(y, score)
    assert t == 2.0
    assert (score[y == 1] >= t).mean() == 0.9


def test_threshold_with_ties_never_drops_below_target():
    y = np.array([1] * 10)
    score = np.array([0.0] * 5 + [1.0] * 5)
    assert (score >= dev_threshold(y, score)).mean() >= 0.9


def test_location_groups():
    assert location_group("ICA") == "ICA"
    assert location_group("VERTEBRAL; BASILAR") == "posterior"
    assert location_group("PCA") == "posterior"
    assert location_group("MCA; ACA") == "multiple"
    assert location_group("missing") is None
    assert location_group(float("nan")) is None
