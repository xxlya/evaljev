from evaljev.metrics import brier_score, distribution_shift, expected_calibration_error


def test_metrics_smoke():
    ps = [0.9, 0.8, 0.2, 0.1]
    ys = [1, 1, 0, 0]
    assert brier_score(ps, ys) < 0.1
    assert expected_calibration_error(ps, ys, bins=5) >= 0
    assert distribution_shift({"a": 1, "b": 0}, {"a": 1, "b": 0}) == 0
