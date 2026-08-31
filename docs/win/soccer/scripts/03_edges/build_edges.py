#!/usr/bin/env python3
# docs/win/soccer/scripts/03_edges/build_edges.py

import math
import traceback
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[2]

INPUT_DIR  = BASE / "02_juice"
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
SUPPORTED_MATCH_ODDS_SOURCES = {"raw", "adjusted", "engine"}

REDUNDANT_SIGNAL_RELATION = "alias_of_ev"


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
# CONFIG
# =========================

def load_match_odds_probability_sources() -> dict[str, str]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        config = data["probability_sources"]["soccer"]["match_odds"]
    except (TypeError, KeyError) as e:
        raise ValueError(
            "markets.yaml missing probability_sources.soccer.match_odds"
        ) from e

    if not isinstance(config, dict):
        raise ValueError(
            "probability_sources.soccer.match_odds must be a mapping"
        )

    missing = [
        key
        for key in MATCH_ODDS_SOURCE_KEYS
        if key not in config
    ]

    if missing:
        raise ValueError(
            f"probability_sources.soccer.match_odds missing keys: {missing}"
        )

    normalized = {}

    for key in MATCH_ODDS_SOURCE_KEYS:
        source = str(config[key]).strip().lower()

        if source not in SUPPORTED_MATCH_ODDS_SOURCES:
            raise ValueError(
                f"Unsupported 1X2 probability source for {key}: {source!r}. "
                f"Supported: {sorted(SUPPORTED_MATCH_ODDS_SOURCES)}"
            )

        normalized[key] = source

    return normalized


MATCH_ODDS_PROBABILITY_SOURCES = load_match_odds_probability_sources()


# =========================
# CORE CALCS
# =========================

def calc_edge(book_odds, fair_odds):
    try:
        if book_odds is not None and fair_odds is not None:
            book_odds = float(book_odds)
            fair_odds = float(fair_odds)

            if not math.isfinite(book_odds) or not math.isfinite(fair_odds):
                return None

            if book_odds <= 0 or fair_odds <= 0:
                return None

            return (book_odds / fair_odds) - 1

    except Exception:
        pass

    return None


def calc_ev(p, odds):
    try:
        if p is not None and odds is not None:
            p = float(p)
            odds = float(odds)

            if not math.isfinite(p) or not math.isfinite(odds):
                return None

            return (p * (odds - 1)) - (1 - p)

    except Exception:
        pass

    return None


def calc_kelly(p, odds):
    try:
        if p is not None and odds is not None:
            p = float(p)
            odds = float(odds)

            if (
                not math.isfinite(p)
                or not math.isfinite(odds)
                or odds <= 1
            ):
                return None

            k = ((p * odds) - 1) / (odds - 1)
            return max(0, k)

    except Exception:
        pass

    return None


def _probability_column(side: str, source: str) -> str:
    if source == "raw":
        return f"{side}_prob"

    if source == "adjusted":
        return f"juiced_{side}_prob"

    if source == "engine":
        return f"engine_{side}_prob"

    raise ValueError(
        f"Unsupported probability source: {source!r}"
    )


def _probability_value(
    row: dict,
    side: str,
    source: str,
) -> tuple[float | None, str]:
    column = _probability_column(
        side,
        source,
    )
    value = row.get(column)

    try:
        value = float(value)
    except Exception:
        return None, column

    if (
        not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        return None, column

    return value, column


def _fair_decimal_from_prob(
    probability,
) -> float | None:
    try:
        p = float(probability)
    except Exception:
        return None

    if (
        not math.isfinite(p)
        or p <= 0
        or p > 1
    ):
        return None

    return 1.0 / p


# =========================
# MARKET PROCESSORS
# =========================

def process_match(df) -> tuple[pd.DataFrame, int]:
    rows       = []
    null_edges = 0

    for _, r in df.iterrows():
        row = r.to_dict()

        for side in ["home", "draw", "away"]:
            book = row.get(
                f"dk_{side}_decimal"
            )

            ev_prob, ev_source_column = _probability_value(
                row,
                side,
                MATCH_ODDS_PROBABILITY_SOURCES["ev"],
            )

            kelly_prob, kelly_source_column = _probability_value(
                row,
                side,
                MATCH_ODDS_PROBABILITY_SOURCES["kelly"],
            )

            fair_prob, fair_source_column = _probability_value(
                row,
                side,
                MATCH_ODDS_PROBABILITY_SOURCES["fair_odds"],
            )

            edge_prob, edge_source_column = _probability_value(
                row,
                side,
                MATCH_ODDS_PROBABILITY_SOURCES["edge"],
            )

            selection_prob, selection_source_column = _probability_value(
                row,
                side,
                MATCH_ODDS_PROBABILITY_SOURCES["selection_filter"],
            )

            fair_decimal = _fair_decimal_from_prob(
                fair_prob
            )

            edge_fair_decimal = _fair_decimal_from_prob(
                edge_prob
            )

            edge = calc_edge(
                book,
                edge_fair_decimal,
            )

            ev = calc_ev(
                ev_prob,
                book,
            )

            kelly = calc_kelly(
                kelly_prob,
                book,
            )

            row[f"{side}_ev_prob"] = ev_prob
            row[f"{side}_ev_prob_source"] = ev_source_column

            row[f"{side}_kelly_prob"] = kelly_prob
            row[f"{side}_kelly_prob_source"] = kelly_source_column

            row[f"{side}_fair_odds_prob"] = fair_prob
            row[f"{side}_fair_odds_prob_source"] = fair_source_column
            row[f"{side}_fair_decimal"] = fair_decimal

            row[f"{side}_edge_prob"] = edge_prob
            row[f"{side}_edge_prob_source"] = edge_source_column
            row[f"{side}_edge_fair_decimal"] = edge_fair_decimal

            row[f"{side}_selection_prob"] = selection_prob
            row[f"{side}_selection_prob_source"] = selection_source_column

            row[f"{side}_edge"] = edge
            row[f"{side}_ev"] = ev
            row[f"{side}_kelly"] = kelly

            if edge is None:
                null_edges += 1

        rows.append(row)

    return pd.DataFrame(rows), null_edges


def process_totals(df) -> tuple[pd.DataFrame, int]:
    """
    Totals edge and EV are the same mathematical signal because:

        fair_odds = 1 / p
        edge = book_odds / fair_odds - 1
             = p * book_odds - 1
             = EV

    EV is therefore calculated once and edge is written as an explicit
    compatibility alias. edge must not be treated as independent confirmation.
    """
    rows       = []
    null_edges = 0

    for _, r in df.iterrows():
        row = r.to_dict()

        p_over = row.get(
            "engine_over_prob"
        )
        book_over = (
            row.get("dk_over25_decimal")
            or row.get("dk_over35_decimal")
        )

        p_under = row.get(
            "engine_under_prob"
        )
        book_under = (
            row.get("dk_under25_decimal")
            or row.get("dk_under35_decimal")
        )

        over_ev = calc_ev(
            p_over,
            book_over,
        )
        under_ev = calc_ev(
            p_under,
            book_under,
        )

        row["over_ev"] = over_ev
        row["over_edge"] = over_ev
        row["over_edge_relation"] = REDUNDANT_SIGNAL_RELATION
        row["over_kelly"] = calc_kelly(
            p_over,
            book_over,
        )

        row["under_ev"] = under_ev
        row["under_edge"] = under_ev
        row["under_edge_relation"] = REDUNDANT_SIGNAL_RELATION
        row["under_kelly"] = calc_kelly(
            p_under,
            book_under,
        )

        if over_ev is None:
            null_edges += 1

        if under_ev is None:
            null_edges += 1

        rows.append(row)

    return pd.DataFrame(rows), null_edges


def process_btts(df) -> tuple[pd.DataFrame, int]:
    """
    BTTS edge and EV are the same mathematical signal because:

        fair_odds = 1 / p
        edge = book_odds / fair_odds - 1
             = p * book_odds - 1
             = EV

    EV is therefore calculated once and edge is written as an explicit
    compatibility alias. edge must not be treated as independent confirmation.
    """
    rows       = []
    null_edges = 0

    for _, r in df.iterrows():
        row = r.to_dict()

        p_yes = row.get(
            "engine_btts_yes_prob"
        )
        book_yes = row.get(
            "btts_yes"
        )

        p_no = row.get(
            "engine_btts_no_prob"
        )
        book_no = row.get(
            "btts_no"
        )

        yes_ev = calc_ev(
            p_yes,
            book_yes,
        )
        no_ev = calc_ev(
            p_no,
            book_no,
        )

        row["yes_ev"] = yes_ev
        row["yes_edge"] = yes_ev
        row["yes_edge_relation"] = REDUNDANT_SIGNAL_RELATION
        row["yes_kelly"] = calc_kelly(
            p_yes,
            book_yes,
        )

        row["no_ev"] = no_ev
        row["no_edge"] = no_ev
        row["no_edge_relation"] = REDUNDANT_SIGNAL_RELATION
        row["no_kelly"] = calc_kelly(
            p_no,
            book_no,
        )

        if yes_ev is None:
            null_edges += 1

        if no_ev is None:
            null_edges += 1

        rows.append(row)

    return pd.DataFrame(rows), null_edges


# =========================
# MAIN
# =========================

def main():
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== build_edges RUN {_now()} ===\n"
        )

    summary = {
        "files_found":   0,
        "files_written": 0,
        "skipped":       0,
        "total_rows":    0,
        "null_edges":    0,
        "errors":        0,
    }

    per_file = []

    _log(
        f"INPUT_DIR : {INPUT_DIR}"
    )
    _log(
        f"OUTPUT_DIR: {OUTPUT_DIR}"
    )
    _log(
        f"CONFIG    : {CONFIG_PATH}"
    )

    _log(
        "MATCH_ODDS PROBABILITY SOURCES | "
        + " | ".join(
            f"{key}={MATCH_ODDS_PROBABILITY_SOURCES[key]}"
            for key in MATCH_ODDS_SOURCE_KEYS
        )
    )

    _log(
        "TOTALS/BTTS SIGNAL RELATIONSHIP | "
        "edge=EV | edge is an alias for compatibility, "
        "not an independent signal"
    )

    input_files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    _log(
        f"Files found: {len(input_files)}"
    )

    for file in input_files:
        name   = file.name
        market = None

        pf = {
            "name": name,
            "market": "unknown",
            "rows": 0,
            "null_edges": 0,
            "status": "ok",
        }

        if "match_odds" in name:
            market = "match_odds"

        elif "total" in name:
            market = "total"

        elif "btts" in name:
            market = "btts"

        else:
            _log(
                f"SKIP unrecognized file: {name}"
            )

            summary["skipped"] += 1
            pf["status"] = "skipped"
            per_file.append(pf)
            continue

        pf["market"] = market
        summary["files_found"] += 1

        _log(
            f"--- FILE: {name}  market={market}"
        )

        try:
            df = pd.read_csv(file)

            if df.empty:
                _log(
                    f"{name} empty — skipping"
                )

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

            pf["null_edges"] = null_edges
            summary["null_edges"] += null_edges

            if null_edges > 0:
                _log(
                    f"{name} | {null_edges} null edges "
                    f"(missing probability/book inputs)",
                    "WARN",
                )

            out_path = OUTPUT_DIR / name

            out.to_csv(
                out_path,
                index=False,
            )

            summary["files_written"] += 1

            _log(
                f"WROTE: {out_path} "
                f"({len(out)} rows, {null_edges} null edges)"
            )

        except Exception as e:
            _log(
                f"{name} FAILED: {e}\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )

            pf["status"] = "error"
            summary["errors"] += 1

        per_file.append(pf)

    _write_summary(
        summary,
        per_file,
    )

    print("edges complete.")


if __name__ == "__main__":
    main()