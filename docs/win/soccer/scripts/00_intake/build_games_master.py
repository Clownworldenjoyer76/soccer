#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path

SPORTSBOOK_ROOT = Path("docs/win/soccer/00_intake/sportsbook")
OUTPUT_DIR = Path("docs/win/soccer/00_intake/games")

FIELDS = ["game_id","sport","league","match_date","match_time","home_team","away_team"]

LEAGUE_MAP = {
    "la liga":"laliga","la_liga":"laliga","laliga":"laliga",
    "epl":"epl",
    "serie a":"seriea","serie_a":"seriea","seriea":"seriea",
    "bundesliga":"bundesliga",
    "ligue 1":"ligue1","ligue_1":"ligue1","ligue1":"ligue1",
    "mls":"mls",
}

def clean(v): return str(v or "").strip()

def normalize_league(v):
    v = clean(v).lower()
    return LEAGUE_MAP.get(v, v)

def source_files():
    return [
        p for p in sorted(SPORTSBOOK_ROOT.glob("*/*.csv"))
        if p.parent.name.lower() != "normalized"
    ]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_date = {}
    files_read = rows_read = 0

    for path in source_files():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = set(FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} missing required columns: {sorted(missing)}")
            files_read += 1

            for raw in reader:
                rows_read += 1
                row = {
                    "game_id": clean(raw.get("game_id")),
                    "sport": clean(raw.get("sport")).lower(),
                    "league": normalize_league(raw.get("league")),
                    "match_date": clean(raw.get("match_date")),
                    "match_time": clean(raw.get("match_time")),
                    "home_team": clean(raw.get("home_team")),
                    "away_team": clean(raw.get("away_team")),
                }

                if not row["game_id"]:
                    raise ValueError(f"{path} contains a row with no game_id: {row}")
                if not row["match_date"]:
                    raise ValueError(f"{path} contains a row with no match_date: {row}")

                date_games = by_date.setdefault(row["match_date"], {})
                prior = date_games.get(row["game_id"])
                if prior is not None and prior != row:
                    raise ValueError(
                        f"Conflicting identity for game_id={row['game_id']}: {prior} vs {row}"
                    )
                date_games[row["game_id"]] = row

    files_written = games_written = 0

    for match_date, games_by_id in sorted(by_date.items()):
        rows = list(games_by_id.values())
        rows.sort(key=lambda r: (r["league"], r["match_time"], r["home_team"], r["away_team"], r["game_id"]))
        out = OUTPUT_DIR / f"{match_date}_soccer_games.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        files_written += 1
        games_written += len(rows)
        print(f"WROTE {out} ({len(rows)} games)")

    print(
        f"Games master complete. source_files={files_read} source_rows={rows_read} "
        f"files_written={files_written} games_written={games_written}"
    )

if __name__ == "__main__":
    main()
