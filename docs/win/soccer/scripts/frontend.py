#!/usr/bin/env python3
"""Export the two soccer frontend CSV files (MLS, EPL, La Liga, Ligue 1, Serie A, Bundesliga).

Run from any working directory:
    python frontend.py

Writes:
    frontend/soccer_slate.csv  predictions + sportsbook odds + selected bets, all dates, all leagues
    frontend/soccer_final.csv  graded final results

Slate rows: one row per game (predictions and odds joined on game_id); a game with
selected bets gets one row per bet instead, with the bet columns filled. Games
without a bet have blank bet columns. Source files are never modified.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

LEAGUES = ("mls", "epl", "laliga", "ligue1", "seriea", "bundesliga")
SLATE_FIRST_COLUMNS = (
    "sport", "league", "match_date", "game_id", "match_time", "home_team", "away_team",
)
DATE_PATTERN = r"(\d{4}_\d{2}_\d{2})"


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs/win/soccer").is_dir():
            return parent
    raise FileNotFoundError("Could not find docs/win/soccer above " + str(here))


ROOT = find_root()
SOCCER = ROOT / "docs/win/soccer"
PRED_DIR = SOCCER / "00_intake/predictions/normalized"
ODDS_DIR = SOCCER / "00_intake/sportsbook/normalized"
PICKS_DIR = SOCCER / "04_select"
FINAL_SOURCE = SOCCER / "05_final_scores/results/graded/SOCCER_final.csv"
OUT = ROOT / "frontend"
SLATE_DEST = OUT / "soccer_slate.csv"
FINAL_DEST = OUT / "soccer_final.csv"


def clean(value: str | None) -> str:
    if value is None:
        return ""
    result = value.strip()
    return "" if result.lower() in {"nan", "none", "null"} else result


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers:
            return [], []
        if any(not h for h in headers) or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid or duplicate CSV headers in {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Malformed CSV row in {path} at line {line_number}")
            rows.append({k: clean(v) for k, v in row.items()})
    return headers, rows


def index_by_game_id(rows: list[dict[str, str]], source: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        game_id = row.get("game_id", "")
        if not game_id:
            print(f"WARNING: Skipping row with blank game_id in {rel(source)}")
            continue
        if game_id in result:
            print(f"WARNING: Duplicate game_id={game_id} in {rel(source)}; using the last one")
        result[game_id] = row
    return result


def fill_missing(target: dict[str, str], source_row: dict[str, str]) -> None:
    for field, value in source_row.items():
        if value and not target.get(field, ""):
            target[field] = value


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ordered_headers(first: tuple[str, ...], seen: list[str]) -> list[str]:
    headers = [h for h in first if h in seen]
    headers += [h for h in seen if h not in headers]
    return headers


def league_dates(folder: Path, league: str) -> set[str]:
    if not folder.is_dir():
        return set()
    pattern = re.compile(rf"{DATE_PATTERN}_{league}\.csv", re.IGNORECASE)
    return {
        match.group(1)
        for path in folder.iterdir()
        if path.is_file() and (match := pattern.fullmatch(path.name))
    }


def picks_dates() -> set[str]:
    if not PICKS_DIR.is_dir():
        return set()
    pattern = re.compile(rf"{DATE_PATTERN}_soccer_bets\.csv", re.IGNORECASE)
    return {
        match.group(1)
        for path in PICKS_DIR.iterdir()
        if path.is_file() and (match := pattern.fullmatch(path.name))
    }


def find_file(folder: Path, name: str) -> Path:
    """Return folder/name, matching the file name case-insensitively."""
    exact = folder / name
    if exact.is_file() or not folder.is_dir():
        return exact
    for path in folder.iterdir():
        if path.is_file() and path.name.lower() == name.lower():
            return path
    return exact


def build_slate() -> int:
    columns: list[str] = []
    all_rows: list[dict[str, str]] = []
    total_games = total_picks = 0

    def include(fields: list[str]) -> None:
        for field in fields:
            if field not in columns:
                columns.append(field)

    pick_dates = picks_dates()
    league_date_map = {
        league: league_dates(PRED_DIR, league) | league_dates(ODDS_DIR, league)
        for league in LEAGUES
    }
    dates = sorted(pick_dates.union(*league_date_map.values()))
    if not dates:
        raise FileNotFoundError(f"No dated soccer files found under {SOCCER}")

    for date in dates:
        games: dict[str, dict[str, str]] = {}
        order: list[str] = []

        for league in LEAGUES:
            if date not in league_date_map[league]:
                continue
            for folder in (PRED_DIR, ODDS_DIR):
                path = find_file(folder, f"{date}_{league}.csv")
                headers, rows = read_csv(path)
                include(headers)
                for game_id, row in index_by_game_id(rows, path).items():
                    game = games.get(game_id)
                    if game is None:
                        game = {"game_id": game_id, "league": league, "match_date": date}
                        games[game_id] = game
                        order.append(game_id)
                    fill_missing(game, row)

        picks_path = find_file(PICKS_DIR, f"{date}_soccer_bets.csv")
        pick_headers, pick_rows = read_csv(picks_path)
        include(pick_headers)
        picks_by_game: dict[str, list[dict[str, str]]] = {}
        for pick in pick_rows:
            game_id = pick.get("game_id", "")
            if game_id not in games:
                print(f"WARNING: Selected bet has unmatched game_id={game_id} in {rel(picks_path)}")
            picks_by_game.setdefault(game_id, []).append(pick)

        for game_id in order:
            game = games[game_id]
            bets = picks_by_game.get(game_id)
            if bets:
                for bet in bets:
                    row = dict(game)
                    fill_missing(row, bet)
                    all_rows.append(row)
                    total_picks += 1
            else:
                all_rows.append(game)
            total_games += 1
        # Selected bets for games that are not in the predictions/odds files are kept as-is.
        for game_id, bets in picks_by_game.items():
            if game_id in games:
                continue
            for bet in bets:
                row = dict(bet)
                if not row.get("match_date"):
                    row["match_date"] = date
                all_rows.append(row)
                total_picks += 1
        print(f"{date}: {len(order)} games, {len(pick_rows)} selected bets")

    if not all_rows:
        raise FileNotFoundError("No soccer games found; slate not written")

    headers = ordered_headers(SLATE_FIRST_COLUMNS, columns)
    for extra in ("league", "match_date", "game_id"):
        if extra not in headers:
            headers.append(extra)
    for row in all_rows:
        for field in headers:
            row.setdefault(field, "")
    write_csv(SLATE_DEST, headers, all_rows)
    print(f"WROTE {rel(SLATE_DEST)}: {len(all_rows)} rows ({total_games} games, {total_picks} selected bets)")
    return len(all_rows)


def build_final() -> int:
    headers, rows = read_csv(FINAL_SOURCE)
    if not rows:
        raise FileNotFoundError(f"No soccer final rows found in {rel(FINAL_SOURCE)}; final file not written")
    write_csv(FINAL_DEST, headers, rows)
    print(f"WROTE {rel(FINAL_DEST)}: {len(rows)} rows")
    return len(rows)


def main() -> None:
    build_slate()
    build_final()
    print("Export complete: 2 CSV files (slate, final)")


if __name__ == "__main__":
    main()
