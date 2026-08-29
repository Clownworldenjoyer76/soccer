#!/usr/bin/env python3
# docs/win/soccer/scripts/00_intake/name_normalization.py

import csv
import re
import traceback
from pathlib import Path
from datetime import datetime, timezone

INTAKE_DIR = Path("docs/win/soccer/00_intake")
FINAL_SCORES_DIR = Path("docs/win/soccer/05_final_scores/results/final_scores")

MAP_FILE = Path("docs/win/soccer/mappings/team_map_soccer.csv")

NO_MAP_DIR = Path("docs/win/soccer/mappings/no_map/")
NO_MAP_DIR.mkdir(parents=True, exist_ok=True)
NO_MAP_FILE = NO_MAP_DIR / "no_map_soccer.csv"

ERROR_DIR = Path("docs/win/soccer/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "name_normalization.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== name_normalization RUN {datetime.now(timezone.utc).isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")


# =========================
# LEAGUE NORMALIZATION
# =========================

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


def normalize_league(value: str) -> str:
    if not value:
        return ""

    return LEAGUE_MAP.get(value.strip().lower(), value.strip().lower())


# =========================
# LOAD TEAM MAP
# =========================

team_map = {}

if MAP_FILE.exists():
    with open(MAP_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            league = normalize_league(row.get("league", ""))
            alias = (row.get("alias") or "").strip().lower()
            canonical = (row.get("canonical_team") or "").strip()

            if league and alias and canonical:
                team_map[(league, alias)] = canonical

    log(f"Team map loaded: {len(team_map)} entries")
else:
    log("WARNING: team_map_soccer.csv not found")


# =========================
# BUILD FILE LIST
# =========================

DATE_PAT = re.compile(r"\d{4}_\d{2}_\d{2}")

files_to_process = []

sb_dir = INTAKE_DIR / "sportsbook"
if sb_dir.exists():
    for f in sorted(sb_dir.glob("*/*.csv")):
        if DATE_PAT.search(f.stem) and f.stem.endswith("_soccer"):
            files_to_process.append(f)

pred_dir = INTAKE_DIR / "predictions"
if pred_dir.exists():
    for league_dir in sorted(pred_dir.iterdir()):
        if not league_dir.is_dir():
            continue

        league = league_dir.name

        for f in sorted(league_dir.glob("*.csv")):
            if DATE_PAT.search(f.stem) and f.stem.endswith(f"_{league}"):
                files_to_process.append(f)

if FINAL_SCORES_DIR.exists():
    for league_dir in sorted(FINAL_SCORES_DIR.iterdir()):
        if not league_dir.is_dir():
            continue

        league = league_dir.name

        for f in sorted(league_dir.glob("*.csv")):
            if DATE_PAT.search(f.stem) and f.stem.endswith(f"_{league}"):
                files_to_process.append(f)

log(f"Files to process: {len(files_to_process)}")


# =========================
# PROCESS FILES
# =========================

unmapped = set()
files_processed = 0
rows_processed = 0
rows_updated = 0

try:
    for csv_file in files_to_process:
        try:
            files_processed += 1
            updated_rows = []
            modified = False

            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                if "home_team" not in fieldnames or "away_team" not in fieldnames:
                    log(f"SKIP (no team columns): {csv_file}")
                    continue

                for row in reader:
                    rows_processed += 1

                    if "league" in row:
                        orig = row["league"]
                        norm = normalize_league(orig)

                        if orig != norm:
                            row["league"] = norm
                            modified = True

                    league_val = normalize_league(row.get("league", ""))

                    for side in ["home_team", "away_team"]:
                        team_raw = (row.get(side) or "").strip()
                        team_norm = team_raw.lower()

                        if not team_raw:
                            continue

                        key = (league_val, team_norm)

                        if key in team_map:
                            canonical = team_map[key]

                            if row[side] != canonical:
                                row[side] = canonical
                                modified = True
                                rows_updated += 1

                        else:
                            unmapped.add((league_val, team_raw))

                    updated_rows.append(row)

            if modified and fieldnames:
                with open(csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(updated_rows)

                log(f"UPDATED: {csv_file}")

        except Exception as e:
            log(f"ERROR processing {csv_file}: {e}\n{traceback.format_exc()}")

    # =========================
    # WRITE UNMAPPED
    # =========================

    with open(NO_MAP_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["league", "team"])

        for league, team in sorted(unmapped):
            writer.writerow([league, team])

    log(
        f"SUMMARY | files_processed={files_processed} | "
        f"rows_processed={rows_processed} | "
        f"names_normalized={rows_updated} | "
        f"unmapped={len(unmapped)}"
    )
    log("STATUS: SUCCESS")

except Exception as e:
    log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
    log("STATUS: FAILED")

print("Name normalization complete.")
