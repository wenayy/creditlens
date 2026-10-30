"""Stage 4: prediction API.

POST /predict with an applicant id (scored from the SQL feature table) and get back
the default probability, risk tier, and the lineage of the data that produced it.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DB = ROOT / "data" / "creditlens.db"

model = XGBClassifier()
model.load_model(MODELS / "xgb.json")
COLUMNS = json.loads((MODELS / "feature_columns.json").read_text())
TIERS = json.loads((MODELS / "tiers.json").read_text())

app = FastAPI(title="CreditLens", version="1.0")


class PredictRequest(BaseModel):
    sk_id_curr: int


class PredictResponse(BaseModel):
    sk_id_curr: int
    default_probability: float
    risk_tier: str
    early_warning: bool
    lineage: list[dict]


def tier_for(p: float) -> str:
    return TIERS["labels"][int(np.searchsorted(TIERS["cutoffs"], p, side="right"))]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "xgboost", "features": len(COLUMNS)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    with sqlite3.connect(DB) as con:
        row = pd.read_sql("SELECT * FROM features WHERE SK_ID_CURR = ?", con, params=(req.sk_id_curr,))
        lineage = pd.read_sql("SELECT table_name, source_file, source_sha256, loaded_at FROM lineage", con)
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No applicant with SK_ID_CURR={req.sk_id_curr}")

    X = pd.get_dummies(row.drop(columns=["SK_ID_CURR", "TARGET"]), dtype=float).reindex(columns=COLUMNS, fill_value=0.0)
    p = float(model.predict_proba(X)[:, 1][0])
    tier = tier_for(p)
    return PredictResponse(
        sk_id_curr=req.sk_id_curr,
        default_probability=round(p, 4),
        risk_tier=tier,
        early_warning=tier == TIERS["labels"][-1],
        lineage=lineage.to_dict(orient="records"),
    )
