"""Stage 5: drift monitor.

Compares a new batch of applicants against the training distribution with the
Population Stability Index (PSI) per feature. Industry convention:
  PSI < 0.10 stable · 0.10-0.25 watch · > 0.25 drifted (retrain / review policy).

Usage:
  python monitoring/drift.py                 # holdout-style sample from the feature table
  python monitoring/drift.py --batch new.csv # any CSV with the same feature columns
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "models" / "drift_reference.json").read_text())
EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e, a = expected + EPS, actual + EPS
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "DRIFT" if value > 0.25 else "WATCH" if value > 0.10 else "stable"


def drift_report(batch: pd.DataFrame) -> pd.DataFrame:
    """PSI and status per monitored feature, most drifted first."""
    rows = []
    for col, ref in REFERENCE.items():
        if col not in batch:
            continue
        values = batch[col].dropna()
        edges = np.array(ref["edges"])
        counts, _ = np.histogram(values.clip(edges[0], edges[-1]), bins=edges)
        share = counts / max(counts.sum(), 1)
        value = psi(np.array(ref["share"]), share)
        rows.append({"feature": col, "psi": round(value, 4), "status": status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, help="CSV of new applicants with feature columns")
    parser.add_argument("--sample", type=int, default=20_000)
    args = parser.parse_args()

    if args.batch:
        batch = pd.read_csv(args.batch)
    else:
        with sqlite3.connect(ROOT / "data" / "creditlens.db") as con:
            batch = pd.read_sql(f"SELECT * FROM features ORDER BY RANDOM() LIMIT {int(args.sample)}", con)

    report = drift_report(batch)
    out = ROOT / "reports" / "drift_report.csv"
    out.parent.mkdir(exist_ok=True)
    report.to_csv(out, index=False)

    print(report.head(15).to_string(index=False))
    print(f"\n{(report.status == 'DRIFT').sum()} drifted, {(report.status == 'WATCH').sum()} to watch, "
          f"{(report.status == 'stable').sum()} stable of {len(report)} features -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
