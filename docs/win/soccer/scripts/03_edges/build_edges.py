#!/usr/bin/env python3
# docs/win/soccer/scripts/03_edges/build_edges.py

import math
import traceback
from pathlib import Path
from datetime import datetime, UTC

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
SUPPORTED_MATCH_ODDS_SOURCES = {"raw", "adjusted", "engine"}

REDUNDANT_SIGNAL_RELATION = "alias_of_ev"

EDGE_COLUMNS_BY_MARKET = {
    "match_odds": ["home_edge", "draw_edge", "away_edge"],
    "total": ["over_edge", "under_edge"],
    "btts": ["yes_edge", "no_edge"],
}


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
# VALIDATION HELPERS
# =========================

def _finite_float(value) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _valid_probability(value) -> float | None:
    probability = _finite_float(value)

    if probability is None:
        return None

    if not 0.0 <= probability <= 1.0:
        return None

    return probability


def _valid_decimal_odds(value) -> float | None:
    odds = _finite_float(value)

    if odds is None:
        return None

    if odds <= 1.0:
        return None

    return odds


def _first_valid_decimal(row: dict, *columns: str) -> float | None:
    for column in columns:
        odds = _valid_decimal_odds(row.get(column))

        if odds is not None:
            return odds

    return None


def _is_missing_or_nonfinite(value) -> bool:
    return _finite_float(value) is None


def _count_null_edges(out: pd.DataFrame, market: str) -> int:
    edge_columns = EDGE_COLUMNS_BY_MARKET[market]

    missing_columns = [
        col
        for col in edge_columns
        if col not in out.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected edge output columns for market={market}: "
            f"{missing_columns}"
        )

    null_edges = 0

    for column in edge_columns:
        null_edges += int(
            out[column].map(_is_missing_or_nonfinite).sum()
        )

    return null_edges


# =========================
# CORE CALCS
# =========================

def calc_edge(book_odds, fair_odds):
    book = _valid_decimal_odds(book_odds)
    fair = _valid_decimal_odds(fair_odds)

    if book is None or fair is None:
        return None

    edge = (book / fair) - 1.0

    if not math.isfinite(edge):
        return None

    return edge


def calc_ev(p, odds):
    probability = _valid_probability(p)
    decimal_odds = _valid_decimal_odds(odds)

    if probability is None or decimal_odds is None:
        return None

    ev = (
        probability * (decimal_odds - 1.0)
    ) - (1.0 - probability)

    if not math.isfinite(ev):
        return None

    return ev


def calc_kelly(p, odds):
    probability = _valid_probability(p)
    decimal_odds = _valid_decimal_odds(odds)

    if probability is None or decimal_odds is None:
        return None

    kelly = (
        (probability * decimal_odds) - 1.0
    ) / (decimal_odds - 1.0)

    if not math.isfinite(kelly):
        return None

    return max(0.0, kelly)


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

    return _valid_probability(
        row.get(column)
    ), column


def _fair_decimal_from_prob(
    probability,
) -> float | None:
    p = _valid_probability(
        probability
    )

    if p is None or p <= 0.0:
        return None

    fair_odds = 1.0 / p

    return _valid_decimal_odds(
        fair_odds
    )


# =========================
# MARKET PROCESSORS
# =========================

def process_match(df) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        row = r.to_dict()

        for side in [
            "home",
            "draw",
            "away",
        ]:
            book = _valid_decimal_odds(
                row.get(
                    f"dk_{side}_decimal"
                )
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

        rows.append(row)

    return pd.DataFrame(rows)


def process_totals(df) -> pd.DataFrame:
    """
    Totals edge and EV are the same mathematical signal because:

        fair_odds = 1 / p
        edge = book_odds / fair_odds - 1
             = p * book_odds - 1
             = EV

    EV is therefore calculated once and edge is written as an explicit
    compatibility alias. edge must not be treated as independent confirmation.
    """
    rows = []

    for _, r in df.iterrows():
        row = r.to_dict()

        p_over = _valid_probability(
            row.get(
                "engine_over_prob"
            )
        )

        book_over = _first_valid_decimal(
            row,
            "dk_over25_decimal",
            "dk_over35_decimal",
        )

        p_under = _valid_probability(
            row.get(
                "engine_under_prob"
            )
        )

        book_under = _first_valid_decimal(
            row,
            "dk_under25_decimal",
            "dk_under35_decimal",
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

        rows.append(row)

    return pd.DataFrame(rows)


def process_btts(df) -> pd.DataFrame:
    """
    BTTS edge and EV are the same mathematical signal because:

        fair_odds = 1 / p
        edge = book_odds / fair_odds - 1
             = p * book_odds - 1
             = EV

    EV is therefore calculated once and edge is written as an explicit
    compatibility alias. edge must not be treated as independent confirmation.
    """
    rows = []

    for _, r in df.iterrows():
        row = r.to_dict()

        p_yes = _valid_probability(
            row.get(
                "engine_btts_yes_prob"
            )
        )

        book_yes = _valid_decimal_odds(
            row.get(
                "btts_yes"
            )
        )

        p_no = _valid_probability(
            row.get(
                "engine_btts_no_prob"
            )
        )

        book_no = _valid_decimal_odds(
            row.get(
                "btts_no"
            )
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

        rows.append(row)

    return pd.DataFrame(rows)


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
        "files_found": 0,
        "files_written": 0,
        "skipped": 0,
        "total_rows": 0,
        "null_edges": 0,
        "errors": 0,
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
        name = file.name
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
                out = process_match(df)

            elif market == "total":
                out = process_totals(df)

            else:
                out = process_btts(df)

            null_edges = _count_null_edges(
                out,
                market,
            )

            pf["null_edges"] = null_edges
            summary["null_edges"] += null_edges

            if null_edges > 0:
                _log(
                    f"{name} | {null_edges} null/invalid edge calculations "
                    f"in written edge columns",
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