#!/usr/bin/env python3
"""Combine dated soccer market CSVs into one row per game_id.

All paths are relative to the current working directory (repository root
when executed by the provided GitHub Actions workflows).
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

SOURCE_DIR = Path("docs/win/soccer/03_edges")
OUTPUT_DIR = Path("frontend/data/index_home/soccer")
LEAGUES = ("Bundesliga", "epl", "laliga", "ligue1", "seriea", "mls")
MARKETS = {
    "_btts": "btts",
    "__match_odds": "match_odds",
    "__total_25": "total_25",
    "__total_35": "total_35",
}
FILE_PATTERN = re.compile(
    r"^(\d{4}_\d{2}_\d{2})_([Bb]undesliga|epl|laliga|ligue1|seriea|mls)"
    r"(_btts|__match_odds|__total_25|__total_35)\.csv$"
)

# These field names describe the O/U market itself and cannot be shared
# between O/U 2.5 and O/U 3.5 without losing market-specific values.
TOTAL_SPECIFIC = {
    "dc_over_prob", "dc_under_prob", "fair_over_decimal", "fair_under_decimal",
    "engine_over_prob", "engine_under_prob",
    "engine_over_prob_source", "engine_under_prob_source",
    "over_ev", "over_edge", "over_edge_relation", "over_kelly", "over_model_prob_source",
    "under_ev", "under_edge", "under_edge_relation", "under_kelly", "under_model_prob_source",
}


def read_market(path, market):
    """Return column names and rows, with distinct total-market headers."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original = reader.fieldnames
        if not original or "game_id" not in original:
            raise ValueError(f"Missing game_id header: {path}")
        if len(original) != len(set(original)) or any(not name for name in original):
            raise ValueError(f"Invalid or duplicate headers: {path}")

        prefix = {"total_25": "over25_", "total_35": "over35_"}.get(market, "")
        rename = lambda field: prefix + field if field in TOTAL_SPECIFIC and prefix else field
        headers = [rename(field) for field in original]
        rows = []
        seen = set()
        for line_number, record in enumerate(reader, start=2):
            if None in record:
                raise ValueError(f"More values than headers: {path}:{line_number}")
            game_id = (record.get("game_id") or "").strip()
            if not game_id:
                raise ValueError(f"Missing game_id: {path}:{line_number}")
            if game_id in seen:
                raise ValueError(f"Duplicate game_id {game_id}: {path}:{line_number}")
            seen.add(game_id)
            record["game_id"] = game_id
            rows.append({rename(name): value or "" for name, value in record.items()})
        return headers, rows


def combine(date, league, markets):
    """Outer-join all available markets on game_id; do not discard any games."""
    column_order = []
    combined = {}
    for market in ("btts", "match_odds", "total_25", "total_35"):
        path = markets.get(market)
        if path is None:
            continue
        headers, records = read_market(path, market)
        for column in headers:
            if column not in column_order:
                column_order.append(column)
        for record in records:
            game_id = record["game_id"]
            target = combined.setdefault(game_id, {"game_id": game_id})
            for column, value in record.items():
                existing = target.get(column, "")
                if existing and value and existing != value:
                    raise ValueError(
                        f"Conflicting shared value for game_id={game_id}, "
                        f"column={column}, source={path}: {existing!r} != {value!r}"
                    )
                if value and not existing:
                    target[column] = value

    if not combined:
        print(f"SKIP {date} {league}: no game rows")
        return False

    output = OUTPUT_DIR / league / f"{date}_{league}_games_today.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_order, extrasaction="raise")
        writer.writeheader()
        writer.writerows(combined.values())
    print(f"WROTE {output} ({len(combined)} games)")
    return True


def main():
    if not SOURCE_DIR.is_dir():
        print(f"No source directory: {SOURCE_DIR}; nothing to do.")
        return

    found = defaultdict(dict)
    for path in sorted(SOURCE_DIR.glob("*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        date, league, suffix = match.groups()
        if league == "bundesliga":
            league = "Bundesliga"
        found[(date, league)][MARKETS[suffix]] = path

    if not found:
        print("No matching soccer market CSVs; nothing to do.")
        return

    created = 0
    for (date, league), markets in sorted(found.items()):
        if combine(date, league, markets):
            created += 1
    print(f"Generated {created} soccer games-today CSV(s).")


if __name__ == "__main__":
    main()
