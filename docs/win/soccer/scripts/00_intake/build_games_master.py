#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path


SPORTSBOOK_ROOT = Path("docs/win/soccer/00_intake/sportsbook")
OUTPUT_DIR = Path("docs/win/soccer/00_intake/games")

FIELDS = [
    "game_id",
    "sport",
    "league",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
]

LEAGUE_MAP = {
    "la liga": "laliga",
    "la_liga": "laliga",
    "laliga": "laliga",
    "epl": "epl",
    "serie a": "seriea",
    "serie_a": "seriea",
    "seriea": "seriea",
    "bundesliga": "bundesliga",
    "ligue 1": "ligue1",
    "ligue_1": "ligue1",
    "ligue1": "ligue1",
    "mls": "mls",
}


def clean(value) -> str:
    return str(value or "").strip()


def normalize_league(value: str) -> str:
    value = clean(value).lower()
    return LEAGUE_MAP.get(value, value)


def source_files() -> list[Path]:
    return [
        path
        for path in sorted(SPORTSBOOK_ROOT.glob("*/*.csv"))
        if path.parent.name.lower() != "normalized"
    ]


def filename_date(path: Path) -> str:
    match = re.match(r"^(\d{4}_\d{2}_\d{2})_", path.name)
    return match.group(1) if match else ""


def same_identity(a: dict[str, str], b: dict[str, str]) -> bool:
    return (
        a["game_id"] == b["game_id"]
        and a["sport"] == b["sport"]
        and a["league"] == b["league"]
        and a["match_date"] == b["match_date"]
        and a["home_team"] == b["home_team"]
        and a["away_team"] == b["away_team"]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_date: dict[str, dict[str, tuple[dict[str, str], Path]]] = {}

    files_read = 0
    rows_read = 0

    for path in source_files():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            missing = set(FIELDS) - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    f"{path} missing required columns: {sorted(missing)}"
                )

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
                    raise ValueError(
                        f"{path} contains a row with no game_id: {row}"
                    )

                if not row["match_date"]:
                    raise ValueError(
                        f"{path} contains a row with no match_date: {row}"
                    )

                date_games = by_date.setdefault(row["match_date"], {})
                prior_entry = date_games.get(row["game_id"])

                if prior_entry is None:
                    date_games[row["game_id"]] = (row, path)
                    continue

                prior_row, prior_path = prior_entry

                if not same_identity(prior_row, row):
                    raise ValueError(
                        f"Conflicting identity for game_id={row['game_id']}: "
                        f"{prior_row} vs {row}"
                    )

                prior_is_own_date = (
                    filename_date(prior_path) == prior_row["match_date"]
                )

                current_is_own_date = (
                    filename_date(path) == row["match_date"]
                )

                if current_is_own_date and not prior_is_own_date:
                    date_games[row["game_id"]] = (row, path)

                elif current_is_own_date == prior_is_own_date:
                    date_games[row["game_id"]] = (row, path)

    files_written = 0
    games_written = 0

    for match_date, games_by_id in sorted(by_date.items()):
        rows = [
            row
            for row, _source_path in games_by_id.values()
        ]

        rows.sort(
            key=lambda row: (
                row["league"],
                row["match_time"],
                row["home_team"],
                row["away_team"],
                row["game_id"],
            )
        )

        output_path = (
            OUTPUT_DIR
            / f"{match_date}_soccer_games.csv"
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDS,
            )

            writer.writeheader()
            writer.writerows(rows)

        files_written += 1
        games_written += len(rows)

        print(
            f"WROTE {output_path} "
            f"({len(rows)} games)"
        )

    print(
        "Games master complete. "
        f"source_files={files_read} "
        f"source_rows={rows_read} "
        f"files_written={files_written} "
        f"games_written={games_written}"
    )


if __name__ == "__main__":
    main()
