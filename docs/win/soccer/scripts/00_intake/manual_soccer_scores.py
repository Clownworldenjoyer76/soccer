#!/usr/bin/env python3
# docs/win/soccer/scripts/00_intake/manual_soccer_scores.py

import argparse
import csv
import re
from pathlib import Path


OUT_DIRS = [
    Path("docs/win/soccer/05_final_scores/results/final_scores_dirty"),
]

CSV_HEADERS = [
    "sport",
    "league",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)$", re.IGNORECASE)
RECORD_RE = re.compile(r"\s*\([^)]*\)\s*$")
INTEGER_RE = re.compile(r"^\d+$")

IGNORE_LINES = {
    "Time\tTeams\tWin\tDraw\tBest",
    "ML\tFinal",
    "Goals\tSportsbook",
    "Log Loss\tDRatings",
    "Log Loss",
}


def clean_market_for_path(value: str) -> str:
    return (
        (value or "")
        .strip()
        .replace("_", "")
        .replace(" ", "")
        .upper()
    )


def clean_market_for_league_value(value: str) -> str:
    return (
        (value or "")
        .strip()
        .replace("_", "")
        .replace(" ", "")
        .lower()
    )


def normalize_match_date(value: str) -> str:
    value = (value or "").strip()

    if not DATE_RE.match(value):
        raise ValueError(f"Invalid match date format: {value}")

    month, day, year = value.split("/")
    return f"{year}_{month.zfill(2)}_{day.zfill(2)}"


def normalize_match_time(value: str) -> str:
    value = (value or "").strip().upper()

    if not TIME_RE.match(value):
        raise ValueError(f"Invalid match time format: {value}")

    time_part, ampm = value.split()
    hour, minute = time_part.split(":")

    return f"{hour.zfill(2)}:{minute} {ampm}"


def clean_team(value: str) -> str:
    value = (value or "").strip()
    value = RECORD_RE.sub("", value)
    return value.strip()


def split_tabs(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split("\t") if part.strip()]


def read_raw_lines(raw_file: Path) -> list[str]:
    text = raw_file.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def clean_raw_lines(raw_lines: list[str]) -> list[str]:
    cleaned = []

    for line in raw_lines:
        line = line.strip()

        if not line:
            continue

        if line in IGNORE_LINES:
            continue

        cleaned.append(line)

    return cleaned


def split_into_match_blocks(lines: list[str]) -> list[list[str]]:
    blocks = []
    current_block = []

    for line in lines:
        if DATE_RE.match(line):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        elif current_block:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    return blocks


def parse_time_and_away_team(block: list[str]) -> tuple[str, str, int]:
    if len(block) < 2:
        raise ValueError(f"Incomplete block, missing time/away line: {block}")

    parts = split_tabs(block[1])

    if len(parts) >= 2:
        match_time = normalize_match_time(parts[0])
        away_team = clean_team(parts[1])
        next_index = 2
        return match_time, away_team, next_index

    match_time = normalize_match_time(block[1])

    if len(block) < 3:
        raise ValueError(f"Incomplete block, missing away team after time: {block}")

    away_team = clean_team(block[2])
    next_index = 3

    return match_time, away_team, next_index


def parse_home_team(block: list[str], index: int) -> tuple[str, int]:
    if len(block) <= index:
        raise ValueError(f"Incomplete block, missing home team line: {block}")

    parts = split_tabs(block[index])

    if not parts:
        raise ValueError(f"Could not parse home team from: {block[index]}")

    home_team = clean_team(parts[0])

    return home_team, index + 1


def parse_scores(block: list[str], start_index: int) -> tuple[str, str]:
    for i in range(start_index, len(block) - 1):
        away_line = block[i].strip()
        home_line = block[i + 1].strip()

        if not INTEGER_RE.match(away_line):
            continue

        home_parts = split_tabs(home_line)

        if not home_parts:
            continue

        home_score_candidate = home_parts[0]

        if INTEGER_RE.match(home_score_candidate):
            away_score = away_line
            home_score = home_score_candidate
            return home_score, away_score

    raise ValueError(f"Could not parse scores from block: {block}")


def parse_match_block(block: list[str], league_value: str) -> dict:
    if len(block) < 4:
        raise ValueError(f"Incomplete match block: {block}")

    match_date = normalize_match_date(block[0])
    match_time, away_team, next_index = parse_time_and_away_team(block)
    home_team, next_index = parse_home_team(block, next_index)
    home_score, away_score = parse_scores(block, next_index)

    return {
        "sport": "soccer",
        "league": league_value,
        "match_date": match_date,
        "match_time": match_time,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
    }


def parse_rows(raw_lines: list[str], league_value: str) -> list[dict]:
    lines = clean_raw_lines(raw_lines)
    blocks = split_into_match_blocks(lines)

    rows = []

    for block in blocks:
        row = parse_match_block(block, league_value)
        rows.append(row)

    return rows


def group_rows_by_match_date(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {}

    for row in rows:
        match_date = row["match_date"]
        grouped.setdefault(match_date, []).append(row)

    return grouped


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (
        row.get("match_date", "").strip(),
        row.get("match_time", "").strip(),
        row.get("home_team", "").strip(),
        row.get("away_team", "").strip(),
    )


def row_has_scores(row: dict) -> bool:
    return bool(
        row.get("home_score", "").strip()
        and row.get("away_score", "").strip()
    )


def read_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def merge_rows(existing_rows: list[dict], incoming_rows: list[dict]) -> list[dict]:
    merged = []
    existing_by_key = {}

    for row in existing_rows:
        key = row_key(row)
        existing_by_key[key] = row

    incoming_by_key = {}

    for row in incoming_rows:
        key = row_key(row)
        incoming_by_key[key] = row

    handled_keys = set()

    for row in existing_rows:
        key = row_key(row)
        incoming_row = incoming_by_key.get(key)

        if incoming_row is None:
            merged.append(row)
            handled_keys.add(key)
            continue

        if row_has_scores(row):
            merged.append(row)
            handled_keys.add(key)
            continue

        merged.append(incoming_row)
        handled_keys.add(key)

    for row in incoming_rows:
        key = row_key(row)

        if key in handled_keys:
            continue

        if key in existing_by_key and row_has_scores(existing_by_key[key]):
            continue

        merged.append(row)
        handled_keys.add(key)

    return merged


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(grouped_rows: dict[str, list[dict]], market_path_value: str) -> None:
    for base_dir in OUT_DIRS:
        out_dir = base_dir / market_path_value

        for match_date, incoming_rows in grouped_rows.items():
            csv_path = out_dir / f"{match_date}_{market_path_value}.csv"

            existing_rows = read_existing_rows(csv_path)
            final_rows = merge_rows(existing_rows, incoming_rows)

            write_csv(csv_path, final_rows)

            print(f"WROTE CSV: {csv_path} ({len(final_rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--market", required=True)
    parser.add_argument("--raw-file", required=True)

    args = parser.parse_args()

    raw_file = Path(args.raw_file)

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")

    market_path_value = clean_market_for_path(args.market)
    league_value = clean_market_for_league_value(args.market)

    if not market_path_value:
        raise ValueError("Market value is empty after cleanup")

    if not league_value:
        raise ValueError("League value is empty after cleanup")

    raw_lines = read_raw_lines(raw_file)
    rows = parse_rows(raw_lines, league_value)

    if not rows:
        raise ValueError("No soccer final score rows were parsed from raw input")

    grouped_rows = group_rows_by_match_date(rows)

    write_outputs(grouped_rows, market_path_value)


if __name__ == "__main__":
    main()
