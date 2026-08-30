#!/usr/bin/env python3
# docs/win/soccer/scripts/00_intake/soccer_cleaner.py

import csv
import math
import re
import traceback
from pathlib import Path
from datetime import datetime, timezone

SPORTSBOOK_DIR = Path("docs/win/soccer/00_intake/sportsbook")
PREDICTIONS_DIR = Path("docs/win/soccer/00_intake/predictions")

SB_NORM_DIR = SPORTSBOOK_DIR / "normalized"
PRED_NORM_DIR = PREDICTIONS_DIR / "normalized"

SB_NORM_DIR.mkdir(parents=True, exist_ok=True)
PRED_NORM_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/soccer/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "soccer_cleaner.txt"

DATE_PAT = re.compile(r"\d{4}_\d{2}_\d{2}")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(
        f"=== soccer_cleaner RUN "
        f"{datetime.now(timezone.utc).isoformat()} ===\n"
    )


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()} | {msg}\n"
        )


# summary counters
sb_files_written = []
pred_files_written = []
total_missing_ids = 0
total_invalid_probability_rows = 0


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

    return LEAGUE_MAP.get(
        value.strip().lower(),
        value.strip().lower(),
    )


# =========================
# 1X2 PROBABILITY VALIDATION
# =========================

PROB_COLS = [
    "home_prob",
    "draw_prob",
    "away_prob",
]

# Raw prediction probabilities are percentage values.
# Allow a maximum 0.1 percentage-point source rounding difference.
PROB_SUM_TOLERANCE = 0.001


def pct_to_decimal(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    cleaned = text.rstrip("%").strip()

    if not cleaned:
        return None

    try:
        number = float(cleaned)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return round(
        number / 100.0,
        6,
    )


def validate_1x2_probabilities(row):
    converted = {}

    for col in PROB_COLS:
        raw_value = row.get(col)

        if (
            raw_value is None
            or not str(raw_value).strip()
        ):
            return (
                False,
                {},
                None,
                f"missing {col}",
            )

        value = pct_to_decimal(raw_value)

        if value is None:
            return (
                False,
                {},
                None,
                f"non-numeric {col}={raw_value!r}",
            )

        if (
            value < 0.0
            or value > 1.0
        ):
            return (
                False,
                {},
                None,
                (
                    f"out-of-range {col}={value} "
                    f"raw={raw_value!r}"
                ),
            )

        converted[col] = value

    probability_sum = round(
        sum(
            converted[col]
            for col in PROB_COLS
        ),
        6,
    )

    sum_error = abs(
        round(
            probability_sum - 1.0,
            6,
        )
    )

    if sum_error > PROB_SUM_TOLERANCE:
        return (
            False,
            converted,
            probability_sum,
            (
                f"invalid 1X2 sum="
                f"{probability_sum:.6f} "
                f"tolerance="
                f"{PROB_SUM_TOLERANCE:.6f}"
            ),
        )

    return (
        True,
        converted,
        probability_sum,
        "",
    )


# =========================
# SPORTSBOOK GAME_ID INDEX
# =========================

def build_game_id_index() -> dict:
    index = {}

    for sb_file in sorted(
        SB_NORM_DIR.glob("*.csv")
    ):
        if not DATE_PAT.search(
            sb_file.stem
        ):
            continue

        try:
            with open(
                sb_file,
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    league = normalize_league(
                        row.get(
                            "league",
                            "",
                        )
                    )

                    match_date = (
                        row.get(
                            "match_date",
                        )
                        or ""
                    ).strip()

                    home_team = (
                        row.get(
                            "home_team",
                        )
                        or ""
                    ).strip()

                    away_team = (
                        row.get(
                            "away_team",
                        )
                        or ""
                    ).strip()

                    game_id = (
                        row.get(
                            "game_id",
                        )
                        or ""
                    ).strip()

                    if game_id:
                        key = (
                            league,
                            match_date,
                            home_team,
                            away_team,
                        )

                        index[key] = game_id

        except Exception as e:
            log(
                f"ERROR reading {sb_file}: "
                f"{e}\n"
                f"{traceback.format_exc()}"
            )

    return index


# =========================
# 1. CLEAN SPORTSBOOK
# =========================

def clean_sportsbook():
    sb_fields = [
        "sport",
        "league",
        "game_id",
        "match_date",
        "match_time",
        "home_team",
        "away_team",
        "dk_home_decimal",
        "dk_draw_decimal",
        "dk_away_decimal",
        "dk_over25_decimal",
        "dk_under25_decimal",
        "dk_over35_decimal",
        "dk_under35_decimal",
        "btts_yes",
        "btts_no",
    ]

    for sb_file in sorted(
        SPORTSBOOK_DIR.glob("*/*.csv")
    ):
        if (
            sb_file.parent.name
            == "normalized"
        ):
            continue

        if not DATE_PAT.search(
            sb_file.stem
        ):
            continue

        if not sb_file.stem.endswith(
            "_soccer"
        ):
            continue

        try:
            date_str = DATE_PAT.search(
                sb_file.stem
            ).group(0)

            log(
                f"SPORTSBOOK: "
                f"processing {sb_file}"
            )

            by_league = {}

            with open(
                sb_file,
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    league_raw = (
                        row.get(
                            "league",
                        )
                        or ""
                    ).strip()

                    league_norm = (
                        normalize_league(
                            league_raw
                        )
                    )

                    if not league_norm:
                        log(
                            "  SKIP row — "
                            "no league value: "
                            f"{row}"
                        )
                        continue

                    by_league.setdefault(
                        league_norm,
                        [],
                    ).append(row)

            for (
                league_norm,
                rows,
            ) in by_league.items():

                out_path = (
                    SB_NORM_DIR
                    / (
                        f"{date_str}_"
                        f"{league_norm}.csv"
                    )
                )

                with open(
                    out_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=sb_fields,
                        extrasaction="ignore",
                    )

                    writer.writeheader()
                    writer.writerows(rows)

                sb_files_written.append(
                    (
                        str(out_path),
                        len(rows),
                    )
                )

                log(
                    f"  WROTE {out_path} "
                    f"({len(rows)} rows)"
                )

        except Exception as e:
            log(
                "ERROR processing "
                f"sportsbook {sb_file}: "
                f"{e}\n"
                f"{traceback.format_exc()}"
            )


# =========================
# 2. CLEAN PREDICTIONS
# =========================

PRED_FIELDS = [
    "sport",
    "league",
    "game_id",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
    "home_prob",
    "draw_prob",
    "away_prob",
    "home_xg",
    "away_xg",
    "expected_total_goals",
]


def clean_predictions(
    game_id_index: dict,
):
    global total_missing_ids
    global total_invalid_probability_rows

    for league_dir in sorted(
        PREDICTIONS_DIR.iterdir()
    ):
        if not league_dir.is_dir():
            continue

        if (
            league_dir.name
            == "normalized"
        ):
            continue

        league = league_dir.name

        for pred_file in sorted(
            league_dir.glob("*.csv")
        ):
            if not DATE_PAT.search(
                pred_file.stem
            ):
                continue

            if not pred_file.stem.endswith(
                f"_{league}"
            ):
                continue

            try:
                date_str = DATE_PAT.search(
                    pred_file.stem
                ).group(0)

                league_norm = (
                    normalize_league(
                        league
                    )
                )

                out_path = (
                    PRED_NORM_DIR
                    / (
                        f"{date_str}_"
                        f"{league_norm}.csv"
                    )
                )

                log(
                    "PREDICTIONS: "
                    f"processing {pred_file}"
                )

                rows_out = []
                missing_id = 0
                invalid_probability_rows = 0

                with open(
                    pred_file,
                    newline="",
                    encoding="utf-8",
                ) as f:
                    reader = csv.DictReader(f)

                    for (
                        line_number,
                        row,
                    ) in enumerate(
                        reader,
                        start=2,
                    ):
                        match_date = (
                            row.get(
                                "match_date",
                            )
                            or ""
                        ).strip()

                        home_team = (
                            row.get(
                                "home_team",
                            )
                            or ""
                        ).strip()

                        away_team = (
                            row.get(
                                "away_team",
                            )
                            or ""
                        ).strip()

                        row_league = (
                            normalize_league(
                                row.get(
                                    "league",
                                    "",
                                )
                                or league
                            )
                        )

                        (
                            probabilities_valid,
                            converted_probabilities,
                            probability_sum,
                            probability_error,
                        ) = (
                            validate_1x2_probabilities(
                                row
                            )
                        )

                        if not probabilities_valid:
                            (
                                invalid_probability_rows
                            ) += 1

                            (
                                total_invalid_probability_rows
                            ) += 1

                            raw_probabilities = {
                                col: row.get(
                                    col,
                                    "",
                                )
                                for col
                                in PROB_COLS
                            }

                            if (
                                converted_probabilities
                            ):
                                converted_text = (
                                    converted_probabilities
                                )
                            else:
                                converted_text = (
                                    "unavailable"
                                )

                            if (
                                probability_sum
                                is not None
                            ):
                                sum_text = (
                                    f"{probability_sum:.6f}"
                                )
                            else:
                                sum_text = (
                                    "unavailable"
                                )

                            log(
                                "  REJECT INVALID 1X2 | "
                                f"file={pred_file} | "
                                f"line={line_number} | "
                                f"league={row_league} | "
                                f"match_date={match_date} | "
                                f"home_team={home_team} | "
                                f"away_team={away_team} | "
                                f"raw={raw_probabilities} | "
                                f"converted="
                                f"{converted_text} | "
                                f"sum={sum_text} | "
                                f"reason="
                                f"{probability_error}"
                            )

                            continue

                        for col in PROB_COLS:
                            row[col] = (
                                converted_probabilities[
                                    col
                                ]
                            )

                        key = (
                            row_league,
                            match_date,
                            home_team,
                            away_team,
                        )

                        game_id = (
                            game_id_index.get(
                                key,
                                "",
                            )
                        )

                        if not game_id:
                            missing_id += 1

                            log(
                                "  NO GAME_ID "
                                f"match: {key}"
                            )

                        row["game_id"] = game_id
                        row["league"] = row_league

                        rows_out.append(row)

                with open(
                    out_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=PRED_FIELDS,
                        extrasaction="ignore",
                    )

                    writer.writeheader()
                    writer.writerows(
                        rows_out
                    )

                pred_files_written.append(
                    (
                        str(out_path),
                        len(rows_out),
                        missing_id,
                        invalid_probability_rows,
                    )
                )

                total_missing_ids += (
                    missing_id
                )

                log(
                    f"  WROTE {out_path} "
                    f"({len(rows_out)} rows, "
                    f"{missing_id} "
                    f"missing game_id, "
                    f"{invalid_probability_rows} "
                    f"invalid 1X2 rejected)"
                )

            except Exception as e:
                log(
                    "ERROR processing "
                    f"predictions {pred_file}: "
                    f"{e}\n"
                    f"{traceback.format_exc()}"
                )


# =========================
# MAIN
# =========================

def main():
    log("START")

    log(
        "1X2 validation enabled | "
        f"sum_tolerance="
        f"{PROB_SUM_TOLERANCE:.6f} | "
        "renormalization=disabled"
    )

    clean_sportsbook()

    game_id_index = (
        build_game_id_index()
    )

    log(
        "Game ID index built: "
        f"{len(game_id_index)} entries"
    )

    clean_predictions(
        game_id_index
    )

    # =========================
    # SUMMARY
    # =========================

    log("--- SUMMARY ---")

    log(
        "Sportsbook files written: "
        f"{len(sb_files_written)}"
    )

    for (
        path,
        rows,
    ) in sb_files_written:
        log(
            f"  FILE: {path} "
            f"({rows} rows)"
        )

    log(
        "Prediction files written: "
        f"{len(pred_files_written)}"
    )

    for (
        path,
        rows,
        missing,
        rejected,
    ) in pred_files_written:
        log(
            f"  FILE: {path} "
            f"({rows} rows, "
            f"{missing} missing game_id, "
            f"{rejected} "
            f"invalid 1X2 rejected)"
        )

    log(
        "Total missing game_ids "
        "across all prediction files: "
        f"{total_missing_ids}"
    )

    log(
        "Total invalid 1X2 "
        "prediction rows rejected: "
        f"{total_invalid_probability_rows}"
    )

    log("STATUS: SUCCESS")

    print(
        "Soccer cleaner complete."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        log(
            f"FATAL ERROR: {e}\n"
            f"{traceback.format_exc()}"
        )

        log("STATUS: FAILED")

        raise
