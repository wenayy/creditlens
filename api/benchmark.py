"""Latency benchmark for POST /predict: 500 calls in-process, reports p50/p95/p99."""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from main import ROOT, app

N = 500

with sqlite3.connect(ROOT / "data" / "creditlens.db") as con:
    ids = [r[0] for r in con.execute(f"SELECT SK_ID_CURR FROM features ORDER BY RANDOM() LIMIT {N}")]

client = TestClient(app)
client.post("/predict", json={"sk_id_curr": ids[0]})  # warm-up

timings = []
for sk in ids:
    start = time.perf_counter()
    r = client.post("/predict", json={"sk_id_curr": sk})
    timings.append((time.perf_counter() - start) * 1000)
    assert r.status_code == 200, r.text

ms = np.array(timings)
result = {"calls": N, "p50_ms": round(float(np.percentile(ms, 50)), 2),
          "p95_ms": round(float(np.percentile(ms, 95)), 2), "p99_ms": round(float(np.percentile(ms, 99)), 2)}
(Path(ROOT) / "reports" / "latency.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("sample response:", json.dumps(client.post("/predict", json={"sk_id_curr": ids[1]}).json())[:300])
