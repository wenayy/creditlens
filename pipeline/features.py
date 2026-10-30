"""Stage 2 runner: executes sql/features.sql against data/creditlens.db."""

import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    con = sqlite3.connect(ROOT / "data" / "creditlens.db")
    start = time.perf_counter()
    con.executescript((ROOT / "sql" / "features.sql").read_text())
    rows, cols = con.execute("SELECT COUNT(*) FROM features").fetchone()[0], len(con.execute("SELECT * FROM features LIMIT 1").description)
    con.close()
    print(f"features: {rows:,} applicants x {cols} columns in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
