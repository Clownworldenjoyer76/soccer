#!/usr/bin/env python3
# docs/win/soccer/scripts/03_edges/build_edges.py

import traceback
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd

BASE = Path(__file__).resolve().parents[2]

INPUT_DIR  = BASE / "02_juice"
OUTPUT_DIR = BASE / "03_edges"
OUTPUT_DIR.mkdir(exist_ok=True)

ERROR_DIR = BASE / "errors" / "03_edges"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "edges_log.txt"


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_found    : {summary['files_found']}",
        f"  files_written  : {summary['files_written']}",
        f"  skipped        : {summary['skipped']}",
        f"  total_rows     : {summary['total_rows']}",
        f"  null_edges     : {summary['null_edges']}",
        f"  errors         : {summary['errors']}",
        "",
        f"  {'file':<50} {'market':<12} {'rows':>5} {'null_edges':>10} {'status':>10}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<50} {pf['market']:<12} {pf['rows']:>5} "
            f"{pf['null_edges']:>10} {pf['status']:>10}"
        )
    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# CORE CALCS
# =========================

def calc_edge(book_odds, fair_odds):
    try:
        if book_odds and fair_odds:
            return (float(book_odds) / float(fair_odds)) - 1
    except Exception:
        pass
    return None


def calc_ev(p, odds):
    try:
        if p and odds:
            p    = float(p)
            odds = float(odds)
            return (p * (odds - 1)) - (1 - p)
    except Exception:
        pass
    return None


def calc_kelly(p, odds):
    try:
        if p and odds:
            p    = float(p)
            odds = float(odds)
            k    = ((p * odds) - 1) / (odds - 1)
            return max(0, k)
    except Exception:
        pass
    return None


def _count_nulls(row: dict, keys: list) -> int:
    return sum(1 for k in keys if row.get(k) is None)


# =========================
# MARKET PROCESSORS
# =========================

def process_match(df) -> tuple[pd.DataFrame, int]:
    rows       = []
    null_edges = 0

    for _, r in df.iterrows():
        row = r.to_dict()

        for side in ["home", "draw", "away"]:
            p    = row.get(f"{side}_prob")
            book = row.get(f"dk_{side}_decimal")
            fair = row.get(f"juiced_{side}_decimal")

            edge  = calc_edge(book, fair)
            ev    = calc_ev(p, book)
            kelly = calc_kelly(p, book)

            row[f"{side}_edge"]  = edge
            row[f"{side}_ev"]    = ev
            row[f"{side}_kelly"] = kelly

            if edge is None:
                null_edges += 1

        rows.append(row)

    return pd.DataFrame(rows), null_edges


def process_totals(df) -> tuple[pd.DataFrame, int]:
    rows       = []
    null_edges = 0

    for _, r in df.iterrows():
        row = r.to_dict()

        p_over    = row.get("engine_over_prob")
        book_over = row.get("dk_over25_decimal") or row.get("dk_over35_decimal")
        fair_over = row.get("fair_over_decimal")

        p_under    = row.get("engine_under_prob")
        book_under = row.get("dk_under25_decimal") or row.get("dk_under35_decimal")
        fair_under = row.get("fair_under_decimal")

        over_edge  = calc_edge(book_over, fair_over)
        under_edge = calc_edge(book_under, fair_under)

        row["over_edge"]   = over_edge
        row["over_ev"]     = calc_ev(p_over, book_over)
        row["over_kelly"]  = calc_kelly(p_over, book_over)
        row["under_edge"]  = under_edge
        row["under_ev"]    = calc_ev(p_under, book_under)
        row["under_kelly"] = calc_kelly(p_under, book_under)

        if over_edge  is None: null_edges += 1
        if under_edge is None: null_edges += 1

        rows.append(row)

    return pd.DataFrame(rows), null_edges


def process_btts(df) -> tuple[pd.DataFrame, int]:
    rows       = []
    null_edges = 0

    for _, r in df.iterrows():
        row = r.to_dict()

        p_yes    = row.get("engine_btts_yes_prob")
        book_yes = row.get("btts_yes")
        fair_yes = row.get("fair_btts_yes_decimal")

        p_no    = row.get("engine_btts_no_prob")
        book_no = row.get("btts_no")
        fair_no = row.get("fair_btts_no_decimal")

        yes_edge = calc_edge(book_yes, fair_yes)
        no_edge  = calc_edge(book_no,  fair_no)

        row["yes_edge"]  = yes_edge
        row["yes_ev"]    = calc_ev(p_yes, book_yes)
        row["yes_kelly"] = calc_kelly(p_yes, book_yes)
        row["no_edge"]   = no_edge
        row["no_ev"]     = calc_ev(p_no, book_no)
        row["no_kelly"]  = calc_kelly(p_no, book_no)

        if yes_edge is None: null_edges += 1
        if no_edge  is None: null_edges += 1

        rows.append(row)

    return pd.DataFrame(rows), null_edges


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== build_edges RUN {_now()} ===\n")

    summary = {
        "files_found":   0,
        "files_written": 0,
        "skipped":       0,
        "total_rows":    0,
        "null_edges":    0,
        "errors":        0,
    }
    per_file = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    _log(f"Files found: {len(input_files)}")

    for file in input_files:
        name   = file.name
        market = None
        pf     = {"name": name, "market": "unknown", "rows": 0, "null_edges": 0, "status": "ok"}

        if "match_odds" in name:
            market = "match_odds"
        elif "total" in name:
            market = "total"
        elif "btts" in name:
            market = "btts"
        else:
            _log(f"SKIP unrecognized file: {name}")
            summary["skipped"] += 1
            pf["status"] = "skipped"
            per_file.append(pf)
            continue

        pf["market"] = market
        summary["files_found"] += 1
        _log(f"--- FILE: {name}  market={market}")

        try:
            df = pd.read_csv(file)

            if df.empty:
                _log(f"{name} empty — skipping")
                pf["status"] = "empty"
                summary["skipped"] += 1
                per_file.append(pf)
                continue

            pf["rows"] = len(df)
            summary["total_rows"] += len(df)

            if market == "match_odds":
                out, null_edges = process_match(df)
            elif market == "total":
                out, null_edges = process_totals(df)
            else:
                out, null_edges = process_btts(df)

            pf["null_edges"]    = null_edges
            summary["null_edges"] += null_edges

            if null_edges > 0:
                _log(f"{name} | {null_edges} null edges (missing book/fair inputs)", "WARN")

            out_path = OUTPUT_DIR / name
            out.to_csv(out_path, index=False)

            summary["files_written"] += 1
            _log(f"WROTE: {out_path} ({len(out)} rows, {null_edges} null edges)")

        except Exception as e:
            _log(f"{name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
            pf["status"] = "error"
            summary["errors"] += 1

        per_file.append(pf)

    _write_summary(summary, per_file)
    print("edges complete.")


if __name__ == "__main__":
    main()
