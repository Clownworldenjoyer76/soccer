#!/usr/bin/env python3
# docs/win/soccer/scripts/03_edges/build_edges.py
#
# PRODUCTION PRICING CONSUMER
# ---------------------------
# Stage 2 is the single authoritative pricing source. Stage 2 writes the
# production probability into engine_* regardless of whether the underlying
# league source is Dixon-Coles or a trained league-specific ML model.
#
# This stage consumes only the Stage-2 engine_* contract, derives EV/Kelly/edge,
# and carries explicit underlying probability provenance forward. It does not
# independently choose or recalculate a competing probability source.

import math
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE / "02_juice"
OUTPUT_DIR = BASE / "03_edges"
OUTPUT_DIR.mkdir(exist_ok=True)
CONFIG_PATH = BASE / "config" / "markets.yaml"
ERROR_DIR = BASE / "errors" / "03_edges"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "edges_log.txt"

MATCH_ODDS_SOURCE_KEYS = (
    "ev",
    "kelly",
    "fair_odds",
    "edge",
    "selection_filter",
)
AUTHORITATIVE_MATCH_ODDS_SOURCE = "engine"
REDUNDANT_SIGNAL_RELATION = "alias_of_ev"
FAIR_DECIMAL_REL_TOLERANCE = 1e-9
FAIR_DECIMAL_ABS_TOLERANCE = 1e-12

EDGE_COLUMNS_BY_MARKET = {
    "match_odds": ["home_edge", "draw_edge", "away_edge"],
    "total": ["over_edge", "under_edge"],
    "btts": ["yes_edge", "no_edge"],
}


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


def validate_engine_only_probability_config() -> None:
    """
    markets.yaml retains the Stage-2 authority contract. The string 'engine'
    means consume the Stage-2 engine_* output; the underlying source can now be
    league-specific and is carried in engine_*_prob_source.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        config = data["probability_sources"]["soccer"]["match_odds"]
    except (TypeError, KeyError) as e:
        raise ValueError(
            "markets.yaml missing probability_sources.soccer.match_odds"
        ) from e
    if not isinstance(config, dict):
        raise ValueError("probability_sources.soccer.match_odds must be a mapping")
    missing = [key for key in MATCH_ODDS_SOURCE_KEYS if key not in config]
    if missing:
        raise ValueError(
            "probability_sources.soccer.match_odds missing keys: "
            f"{missing}"
        )
    non_engine = {
        key: str(config[key]).strip().lower()
        for key in MATCH_ODDS_SOURCE_KEYS
        if str(config[key]).strip().lower() != AUTHORITATIVE_MATCH_ODDS_SOURCE
    }
    if non_engine:
        raise ValueError(
            "Competing 1X2 pricing sources are disabled. Stage-2 engine output "
            f"is authoritative; non-engine settings found: {non_engine}"
        )


def _finite_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _valid_probability(value):
    p = _finite_float(value)
    if p is None or not 0.0 <= p <= 1.0:
        return None
    return p


def _valid_decimal_odds(value):
    odds = _finite_float(value)
    if odds is None or odds <= 1.0:
        return None
    return odds


def _first_valid_decimal(row: dict, *columns: str):
    for column in columns:
        odds = _valid_decimal_odds(row.get(column))
        if odds is not None:
            return odds
    return None


def _is_missing_or_nonfinite(value):
    return _finite_float(value) is None


def _count_null_edges(out: pd.DataFrame, market: str) -> int:
    edge_columns = EDGE_COLUMNS_BY_MARKET[market]
    missing_columns = [col for col in edge_columns if col not in out.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected edge output columns for market={market}: {missing_columns}"
        )
    return sum(
        int(out[column].map(_is_missing_or_nonfinite).sum())
        for column in edge_columns
    )


def _engine_probability_column(side: str) -> str:
    return f"engine_{side}_prob"


def _engine_fair_decimal_column(side: str) -> str:
    return f"engine_{side}_fair_decimal"


def _underlying_source(row: dict, probability_column: str) -> str:
    raw = row.get(f"{probability_column}_source")
    if raw is None or pd.isna(raw) or not str(raw).strip():
        return probability_column
    return str(raw).strip()


def _validated_engine_price_pair(row: dict, side: str):
    probability_column = _engine_probability_column(side)
    fair_decimal_column = _engine_fair_decimal_column(side)
    probability = _valid_probability(row.get(probability_column))
    fair_decimal = _valid_decimal_odds(row.get(fair_decimal_column))
    if probability is None or probability <= 0.0 or fair_decimal is None:
        return probability, None, probability_column, fair_decimal_column
    if not math.isclose(
        fair_decimal,
        1.0 / probability,
        rel_tol=FAIR_DECIMAL_REL_TOLERANCE,
        abs_tol=FAIR_DECIMAL_ABS_TOLERANCE,
    ):
        return probability, None, probability_column, fair_decimal_column
    return probability, fair_decimal, probability_column, fair_decimal_column


def calc_edge(book_odds, fair_odds):
    book = _valid_decimal_odds(book_odds)
    fair = _valid_decimal_odds(fair_odds)
    if book is None or fair is None:
        return None
    edge = (book / fair) - 1.0
    return edge if math.isfinite(edge) else None


def calc_ev(p, odds):
    probability = _valid_probability(p)
    decimal_odds = _valid_decimal_odds(odds)
    if probability is None or decimal_odds is None:
        return None
    ev = probability * (decimal_odds - 1.0) - (1.0 - probability)
    return ev if math.isfinite(ev) else None


def calc_kelly(p, odds):
    probability = _valid_probability(p)
    decimal_odds = _valid_decimal_odds(odds)
    if probability is None or decimal_odds is None:
        return None
    kelly = ((probability * decimal_odds) - 1.0) / (decimal_odds - 1.0)
    if not math.isfinite(kelly):
        return None
    return max(0.0, kelly)


def process_match(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        for side in ("home", "draw", "away"):
            book = _valid_decimal_odds(row.get(f"dk_{side}_decimal"))
            (
                engine_prob,
                engine_fair_decimal,
                probability_column,
                fair_decimal_column,
            ) = _validated_engine_price_pair(row, side)
            underlying = _underlying_source(row, probability_column)
            edge = calc_edge(book, engine_fair_decimal)
            ev = calc_ev(engine_prob, book)
            kelly = calc_kelly(engine_prob, book)

            # Existing Stage-2 authority provenance is retained for compatibility.
            row[f"{side}_ev_prob"] = engine_prob
            row[f"{side}_ev_prob_source"] = probability_column
            row[f"{side}_kelly_prob"] = engine_prob
            row[f"{side}_kelly_prob_source"] = probability_column
            row[f"{side}_fair_odds_prob"] = engine_prob
            row[f"{side}_fair_odds_prob_source"] = probability_column
            row[f"{side}_fair_decimal"] = engine_fair_decimal
            row[f"{side}_edge_prob"] = engine_prob
            row[f"{side}_edge_prob_source"] = probability_column
            row[f"{side}_edge_fair_decimal"] = engine_fair_decimal
            row[f"{side}_selection_prob"] = engine_prob
            row[f"{side}_selection_prob_source"] = probability_column

            # New underlying provenance identifies the actual production model.
            row[f"{side}_ev_prob_underlying_source"] = underlying
            row[f"{side}_kelly_prob_underlying_source"] = underlying
            row[f"{side}_fair_odds_prob_underlying_source"] = underlying
            row[f"{side}_edge_prob_underlying_source"] = underlying
            row[f"{side}_selection_prob_underlying_source"] = underlying

            row[f"{side}_edge"] = edge
            row[f"{side}_ev"] = ev
            row[f"{side}_kelly"] = kelly
        rows.append(row)
    return pd.DataFrame(rows)


def process_totals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        p_over = _valid_probability(row.get("engine_over_prob"))
        p_under = _valid_probability(row.get("engine_under_prob"))
        book_over = _first_valid_decimal(row, "dk_over25_decimal", "dk_over35_decimal")
        book_under = _first_valid_decimal(row, "dk_under25_decimal", "dk_under35_decimal")
        over_ev = calc_ev(p_over, book_over)
        under_ev = calc_ev(p_under, book_under)

        row["over_ev"] = over_ev
        row["over_edge"] = over_ev
        row["over_edge_relation"] = REDUNDANT_SIGNAL_RELATION
        row["over_kelly"] = calc_kelly(p_over, book_over)
        row["over_model_prob_source"] = _underlying_source(row, "engine_over_prob")

        row["under_ev"] = under_ev
        row["under_edge"] = under_ev
        row["under_edge_relation"] = REDUNDANT_SIGNAL_RELATION
        row["under_kelly"] = calc_kelly(p_under, book_under)
        row["under_model_prob_source"] = _underlying_source(row, "engine_under_prob")
        rows.append(row)
    return pd.DataFrame(rows)


def process_btts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        p_yes = _valid_probability(row.get("engine_btts_yes_prob"))
        p_no = _valid_probability(row.get("engine_btts_no_prob"))
        book_yes = _valid_decimal_odds(row.get("btts_yes"))
        book_no = _valid_decimal_odds(row.get("btts_no"))
        yes_ev = calc_ev(p_yes, book_yes)
        no_ev = calc_ev(p_no, book_no)

        row["yes_ev"] = yes_ev
        row["yes_edge"] = yes_ev
        row["yes_edge_relation"] = REDUNDANT_SIGNAL_RELATION
        row["yes_kelly"] = calc_kelly(p_yes, book_yes)
        row["yes_model_prob_source"] = _underlying_source(row, "engine_btts_yes_prob")

        row["no_ev"] = no_ev
        row["no_edge"] = no_ev
        row["no_edge_relation"] = REDUNDANT_SIGNAL_RELATION
        row["no_kelly"] = calc_kelly(p_no, book_no)
        row["no_model_prob_source"] = _underlying_source(row, "engine_btts_no_prob")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== build_edges RUN {_now()} ===\n")

    summary = {
        "files_found": 0,
        "files_written": 0,
        "skipped": 0,
        "total_rows": 0,
        "null_edges": 0,
        "errors": 0,
    }
    per_file = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")
    _log(f"CONFIG    : {CONFIG_PATH}")
    validate_engine_only_probability_config()
    _log(
        "PRODUCTION AUTHORITY | Stage-2 engine_* contract | "
        "underlying source carried via engine_*_prob_source"
    )
    _log(
        "TOTALS/BTTS SIGNAL RELATIONSHIP | edge=EV | edge is an alias for compatibility"
    )

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    _log(f"Files found: {len(input_files)}")

    for file in input_files:
        name = file.name
        market = None
        pf = {"name": name, "market": "unknown", "rows": 0, "null_edges": 0, "status": "ok"}
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
        _log(f"--- FILE: {name} market={market}")

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
                out = process_match(df)
            elif market == "total":
                out = process_totals(df)
            else:
                out = process_btts(df)

            null_edges = _count_null_edges(out, market)
            pf["null_edges"] = null_edges
            summary["null_edges"] += null_edges
            if null_edges > 0:
                _log(
                    f"{name} | {null_edges} null/invalid edge calculations in written edge columns",
                    "WARN",
                )

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
