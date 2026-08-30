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

DATE_RE = re.compile(r"^(\d{4}_\d{2}_\d{2})_")

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


def source_date(path: Path) -> str:
    match = DATE_RE.match(path.name)
    return match.group(1) if match else ""


def source_files() -> list[Path]:
    files: list[Path] = []

    for path in sorted(SPORTSBOOK_ROOT.glob("*/*.csv")):
        if path.parent.name.lower() == "normalized":
            continue

        if not path.name.endswith("_soccer.csv"):
            continue

        if not source_date(path):
            continue

        files.append(path)

    return files


def fixture_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["sport"].casefold(),
        row["league"].casefold(),
        row["match_date"],
        row["home_team"].casefold(),
        row["away_team"].casefold(),
    )


def validate_row(row: dict[str, str], path: Path) -> None:
    required = [
        "game_id",
        "sport",
        "league",
        "match_date",
        "home_team",
        "away_team",
    ]

    missing = [field for field in required if not row[field]]

    if missing:
        raise ValueError(
            f"{path} contains fixture with missing required values "
            f"{missing}: {row}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_date: dict[
        str,
        dict[
            tuple[str, str, str, str, str],
            dict[str, str],
        ],
    ] = {}

    source_files_read = 0
    source_rows_read = 0
    off_date_rows_skipped = 0
    duplicate_fixture_rows_removed = 0
    duplicate_fixture_id_changes = 0

    for path in source_files():
        file_date = source_date(path)

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)

            headers = set(reader.fieldnames or [])
            missing_headers = set(FIELDS) - headers

            if missing_headers:
                raise ValueError(
                    f"{path} missing required columns: "
                    f"{sorted(missing_headers)}"
                )

            source_files_read += 1

            for raw in reader:
                source_rows_read += 1

                row = {
                    "game_id": clean(raw.get("game_id")),
                    "sport": clean(raw.get("sport")).lower(),
                    "league": normalize_league(raw.get("league")),
                    "match_date": clean(raw.get("match_date")),
                    "match_time": clean(raw.get("match_time")),
                    "home_team": clean(raw.get("home_team")),
                    "away_team": clean(raw.get("away_team")),
                }

                validate_row(row, path)

                # A sportsbook file is authoritative only for the date in
                # its filename. Historical sportsbook files can retain
                # fixtures that later moved to another date. Those stale
                # rows must not be imported into the destination date's
                # master fixture file.
                if row["match_date"] != file_date:
                    off_date_rows_skipped += 1
                    continue

                date_rows = by_date.setdefault(file_date, {})
                key = fixture_key(row)

                existing = date_rows.get(key)

                if existing is None:
                    date_rows[key] = row
                    continue

                duplicate_fixture_rows_removed += 1

                if existing["game_id"] != row["game_id"]:
                    duplicate_fixture_id_changes += 1

                    print(
                        "WARNING duplicate fixture with multiple game_ids; "
                        "keeping first authoritative row from date file | "
                        f"date={file_date} "
                        f"league={row['league']} "
                        f"home={row['home_team']} "
                        f"away={row['away_team']} "
                        f"kept={existing['game_id']} "
                        f"ignored={row['game_id']}"
                    )

    files_written = 0
    games_written = 0

    for match_date, fixture_rows in sorted(by_date.items()):
        rows = list(fixture_rows.values())

        rows.sort(
            key=lambda row: (
                row["league"],
                row["match_time"],
                row["home_team"],
                row["away_team"],
                row["game_id"],
            )
        )

        seen_fixtures: set[
            tuple[str, str, str, str, str]
        ] = set()

        seen_game_ids: dict[
            str,
            tuple[str, str, str, str, str],
        ] = {}

        for row in rows:
            key = fixture_key(row)

            if key in seen_fixtures:
                raise ValueError(
                    "Duplicate fixture survived master build: "
                    f"{row}"
                )

            seen_fixtures.add(key)

            game_id = row["game_id"]

            prior_key = seen_game_ids.get(game_id)

            if prior_key is not None and prior_key != key:
                raise ValueError(
                    f"game_id {game_id} maps to multiple fixtures "
                    f"on {match_date}: {prior_key} vs {key}"
                )

            seen_game_ids[game_id] = key

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
        f"source_files={source_files_read} "
        f"source_rows={source_rows_read} "
        f"off_date_rows_skipped={off_date_rows_skipped} "
        f"duplicate_fixture_rows_removed="
        f"{duplicate_fixture_rows_removed} "
        f"duplicate_fixture_id_changes="
        f"{duplicate_fixture_id_changes} "
        f"files_written={files_written} "
        f"games_written={games_written}"
    )


if __name__ == "__main__":
    main()