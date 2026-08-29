#!/usr/bin/env python3
# docs/win/soccer/scripts/01_merge/merge_intake.py

import csv
import traceback
from pathlib import Path
from datetime import datetime, timezone

PRED_DIR = Path("docs/win/soccer/00_intake/predictions/normalized")
BOOK_DIR = Path("docs/win/soccer/00_intake/sportsbook/normalized")
OUT_DIR  = Path("docs/win/soccer/01_merge")
LOG_DIR  = Path("docs/win/soccer/errors/01_merge")

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "merge_intake.txt"

LEAGUES = ["epl", "laliga", "ligue1", "bundesliga", "seriea", "mls"]


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")


def load_csv(path):
    rows = []
    if not path.exists():
        log(f"MISSING FILE: {path}")
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_game_id_index(rows):
    idx = {}
    for r in rows:
        gid = (r.get("game_id") or "").strip()
        if gid:
            idx[gid] = r
    return idx


def process_slate(date, league, summary):
    try:
        pred_path = PRED_DIR / f"{date}_{league}.csv"
        book_path = BOOK_DIR / f"{date}_{league}.csv"

        preds = load_csv(pred_path)
        books = load_csv(book_path)

        if not preds:
            log(f"SKIP {date} {league}: no predictions")
            summary["skipped"] += 1
            return

        if not books:
            log(f"SKIP {date} {league}: no sportsbook")
            summary["skipped"] += 1
            return

        pred_idx  = build_game_id_index(preds)
        matched   = 0
        unmatched = 0

        match_odds_rows = []
        total_25_rows   = []
        total_35_rows   = []
        btts_rows       = []

        for b in books:
            gid = (b.get("game_id") or "").strip()
            p   = pred_idx.get(gid)

            if not p:
                unmatched += 1
                log(f"UNMATCHED game_id={gid}: {date} {league} | {b.get('home_team')} vs {b.get('away_team')}")
                continue

            matched += 1

            base = [
                b["game_id"], b["sport"], b["league"], b["match_date"], b["match_time"],
                b["home_team"], b["away_team"],
                p["home_prob"], p["draw_prob"], p["away_prob"],
                p["home_xg"], p["away_xg"], p["expected_total_goals"],
            ]

            match_odds_rows.append(base + [b["dk_home_decimal"], b["dk_draw_decimal"], b["dk_away_decimal"]])
            total_25_rows.append(base + [b["dk_over25_decimal"], b["dk_under25_decimal"]])
            total_35_rows.append(base + [b["dk_over35_decimal"], b["dk_under35_decimal"]])
            btts_rows.append(base + [b["btts_yes"], b["btts_no"]])

        log(f"{date} {league} | matched={matched} | unmatched={unmatched}")
        summary["total_matched"]   += matched
        summary["total_unmatched"] += unmatched

        base_header = [
            "game_id", "sport", "league", "match_date", "match_time",
            "home_team", "away_team",
            "home_prob", "draw_prob", "away_prob",
            "home_xg", "away_xg", "expected_total_goals",
        ]

        def write(filename, header, rows):
            if not rows:
                log(f"NO ROWS: {filename} — skipping")
                return
            path = OUT_DIR / filename
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            log(f"WROTE {path} ({len(rows)} rows)")
            summary["files_written"] += 1

        write(f"{date}_{league}_match_odds.csv", base_header + ["dk_home_decimal", "dk_draw_decimal", "dk_away_decimal"], match_odds_rows)
        write(f"{date}_{league}_total_25.csv",   base_header + ["dk_over25_decimal", "dk_under25_decimal"],               total_25_rows)
        write(f"{date}_{league}_total_35.csv",   base_header + ["dk_over35_decimal", "dk_under35_decimal"],               total_35_rows)
        write(f"{date}_{league}_btts.csv",        base_header + ["btts_yes", "btts_no"],                                  btts_rows)

    except Exception as e:
        log(f"ERROR {date} {league}: {e}\n{traceback.format_exc()}")
        summary["errors"] += 1


if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== merge_intake RUN {datetime.now(timezone.utc).isoformat()} ===\n")

    summary = {
        "slates_processed": 0,
        "skipped":          0,
        "files_written":    0,
        "total_matched":    0,
        "total_unmatched":  0,
        "errors":           0,
    }

    try:
        for pred_file in sorted(PRED_DIR.glob("*.csv")):
            stem   = pred_file.stem
            league = None
            date   = None

            for lg in LEAGUES:
                if stem.endswith(f"_{lg}"):
                    league = lg
                    date   = stem[: -(len(lg) + 1)]
                    break

            if not league or not date:
                log(f"SKIP unrecognized file: {pred_file.name}")
                continue

            summary["slates_processed"] += 1
            process_slate(date, league, summary)

        log(
            f"SUMMARY: slates_processed={summary['slates_processed']} | "
            f"skipped={summary['skipped']} | "
            f"files_written={summary['files_written']} | "
            f"total_matched={summary['total_matched']} | "
            f"total_unmatched={summary['total_unmatched']} | "
            f"errors={summary['errors']}"
        )
        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise
