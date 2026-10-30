"""Stage 1: raw Home Credit CSVs -> cleaned tables in data/creditlens.db.

Every load is recorded in the `lineage` table (file hash, rows in, rows out),
so any downstream number can be traced back to the exact source file.
"""

import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "creditlens.db"

# Value the source uses for "not employed / pensioner" instead of a real day count.
DAYS_EMPLOYED_SENTINEL = 365243

TABLES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "installments": "installments_payments.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL, "DAYS_EMPLOYED"] = np.nan
    df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan
    df.loc[df["NAME_FAMILY_STATUS"] == "Unknown", "NAME_FAMILY_STATUS"] = np.nan
    return df


def clean_previous(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        df.loc[df[col] == DAYS_EMPLOYED_SENTINEL, col] = np.nan
    return df


CLEANERS = {"application": clean_application, "previous_application": clean_previous}


def main() -> None:
    missing = [f for f in TABLES.values() if not (DATA / f).exists()]
    if missing:
        sys.exit(f"Missing source files in {DATA}: {', '.join(missing)}")

    con = sqlite3.connect(DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS lineage (
             table_name TEXT, source_file TEXT, source_sha256 TEXT,
             rows_in INTEGER, rows_out INTEGER, loaded_at TEXT)"""
    )
    for table, filename in TABLES.items():
        path = DATA / filename
        df = pd.read_csv(path)
        rows_in = len(df)
        df = CLEANERS.get(table, lambda d: d)(df).drop_duplicates()
        df.to_sql(table, con, if_exists="replace", index=False, chunksize=100_000)
        con.execute("DELETE FROM lineage WHERE table_name = ?", (table,))
        con.execute(
            "INSERT INTO lineage VALUES (?, ?, ?, ?, ?, ?)",
            (table, filename, sha256(path), rows_in, len(df), datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
        print(f"{table:22s} {rows_in:>12,} rows in  {len(df):>12,} rows out")

    con.execute("CREATE INDEX IF NOT EXISTS ix_bureau_curr ON bureau(SK_ID_CURR)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_prev_curr ON previous_application(SK_ID_CURR)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_inst_curr ON installments(SK_ID_CURR)")
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
