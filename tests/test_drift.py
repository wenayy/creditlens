"""The drift monitor must stay quiet on an unchanged population and flag a shifted one."""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "monitoring"))
from drift import drift_report  # noqa: E402


@pytest.fixture(scope="module")
def batch() -> pd.DataFrame:
    with sqlite3.connect(ROOT / "data" / "creditlens.db") as con:
        return pd.read_sql("SELECT * FROM features ORDER BY RANDOM() LIMIT 20000", con)


def status_of(report: pd.DataFrame, feature: str) -> str:
    return report.set_index("feature").loc[feature, "status"]


def test_unchanged_population_is_stable(batch):
    report = drift_report(batch)
    assert (report.status == "DRIFT").sum() == 0


def test_shifted_population_is_flagged(batch):
    shifted = batch.copy()
    # A new channel brings in applicants with ~40% higher stated income and weaker bureau scores.
    shifted["AMT_INCOME_TOTAL"] *= 1.4
    shifted["EXT_SOURCE_2"] *= 0.7

    report = drift_report(shifted)

    assert status_of(report, "AMT_INCOME_TOTAL") == "DRIFT"
    assert status_of(report, "EXT_SOURCE_2") == "DRIFT"
    assert status_of(report, "EXT_SOURCE_3") == "stable"
