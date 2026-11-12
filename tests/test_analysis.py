"""Checks that do not need the ATUS microdata."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atus_teen.models import fit_wls, weighted_corr, weighted_mean
from atus_teen.prepare import income_group, parent_educ_group


def test_income_brackets():
    assert income_group(1) == "lt25"
    assert income_group(7) == "lt25"
    assert income_group(8) == "25_50"
    assert income_group(11) == "25_50"
    assert income_group(12) == "50_75"
    assert income_group(13) == "50_75"
    assert income_group(14) == "75_100"
    assert income_group(15) == "100plus"
    assert income_group(16) == "100plus"
    assert income_group(None) == "missing"
    assert income_group(-1) == "missing"


def test_parent_education_groups():
    assert parent_educ_group(34) == "lt_hs"
    assert parent_educ_group(38) == "lt_hs"
    assert parent_educ_group(39) == "hs"
    assert parent_educ_group(40) == "some_college"
    assert parent_educ_group(42) == "some_college"
    assert parent_educ_group(43) == "ba"
    assert parent_educ_group(46) == "grad"
    assert parent_educ_group(None) == "missing"


def test_weighted_mean_and_correlation():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 2.0])
    assert abs(weighted_mean(values, weights) - 2.25) < 1e-12
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = -2 * x
    w = np.array([1.0, 1.0, 1.0, 3.0])
    assert abs(weighted_corr(x, y, w) + 1) < 1e-12


def test_wls_recovers_slope_with_weights():
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.normal(size=n)
    weight = rng.uniform(0.2, 3.0, size=n)
    y = 1.5 - 0.4 * x + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"reading_min": y, "screen_hours": x, "weight": weight})
    result = fit_wls("reading_min ~ screen_hours", frame, "sim")
    slope = result.term("screen_hours")
    assert abs(slope["coef"] + 0.4) < 0.05
    assert slope["ci_low"] < -0.4 < slope["ci_high"]
