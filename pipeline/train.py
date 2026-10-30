"""Stage 3: train a logistic regression baseline and an XGBoost default model,
assign risk tiers, and write the artefacts the API and drift monitor read."""

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42
TIERS = ["A", "B", "C", "D", "E"]  # A = lowest predicted risk, E = early-warning watchlist
DRIFT_BINS = 10


def load_features() -> pd.DataFrame:
    with sqlite3.connect(ROOT / "data" / "creditlens.db") as con:
        return pd.read_sql("SELECT * FROM features", con)


def encode(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.get_dummies(df.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore"), dummy_na=False, dtype=float)
    return X.reindex(columns=columns, fill_value=0.0) if columns is not None else X


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks_stat(y, p), 4)}


def drift_reference(X: pd.DataFrame, numeric: list[str]) -> dict:
    """Decile edges and per-bin share for each numeric feature, used for PSI later."""
    ref = {}
    for col in numeric:
        values = X[col].dropna()
        if values.nunique() < 3:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, DRIFT_BINS + 1)))
        counts, _ = np.histogram(values.clip(edges[0], edges[-1]), bins=edges)
        ref[col] = {"edges": edges.tolist(), "share": (counts / counts.sum()).tolist()}
    return ref


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = load_features()
    y = df["TARGET"].to_numpy()
    X = encode(df)
    numeric = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET") and pd.api.types.is_numeric_dtype(df[c])]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

    logreg = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    logreg.fit(X_train, y_train)
    p_logreg = logreg.predict_proba(X_test)[:, 1]

    xgb = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        eval_metric="auc",
        early_stopping_rounds=100,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    p_xgb = xgb.predict_proba(X_test)[:, 1]

    # Tier cut-offs come from training-set score quintiles, so they stay fixed when scoring new applicants.
    p_train = xgb.predict_proba(X_train)[:, 1]
    cutoffs = np.quantile(p_train, [0.2, 0.4, 0.6, 0.8]).tolist()
    test_tiers = np.array(TIERS)[np.searchsorted(cutoffs, p_xgb, side="right")]
    tier_table = (
        pd.DataFrame({"tier": test_tiers, "default": y_test})
        .groupby("tier")["default"]
        .agg(applicants="size", default_rate="mean")
        .reindex(TIERS)
    )
    tier_table["default_rate"] = tier_table["default_rate"].round(4)

    importance = (
        pd.Series(xgb.get_booster().get_score(importance_type="gain"))
        .sort_values(ascending=False)
        .head(15)
        .round(1)
    )

    results = {
        "rows": {"train": int(len(X_train)), "test": int(len(X_test)), "default_rate": round(float(y.mean()), 4)},
        "logistic_regression": metrics(y_test, p_logreg),
        "xgboost": {**metrics(y_test, p_xgb), "best_iteration": int(xgb.best_iteration)},
        "tier_cutoffs": dict(zip(["A|B", "B|C", "C|D", "D|E"], [round(c, 4) for c in cutoffs])),
        "tiers": tier_table.reset_index().to_dict(orient="records"),
        "top_features_by_gain": importance.to_dict(),
    }

    xgb.save_model(MODELS / "xgb.json")
    joblib.dump(logreg, MODELS / "logreg.joblib")
    (MODELS / "feature_columns.json").write_text(json.dumps(list(X.columns)))
    (MODELS / "tiers.json").write_text(json.dumps({"labels": TIERS, "cutoffs": cutoffs}))
    (MODELS / "drift_reference.json").write_text(json.dumps(drift_reference(X_train, numeric)))
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
