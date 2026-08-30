#!/usr/bin/env python3
# docs/win/soccer/scripts/00_intake/soccer_cleaner.py

import csv
import math
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

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

PROB_COLS = ["home_prob", "draw_prob", "away_prob"]
PROB_SUM_TOLERANCE = 0.001

XG_COLS = ["home_xg", "away_xg", "expected_total_goals"]

# Source xG values are normally rounded to two decimals.
# This allows only a small separate-rounding difference.
XG_TOTAL_TOLERANCE = 0.01

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

SB_FIELDS = [
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

sb_files_written = []
pred_files_written = []
total_missing_ids = 0
total_invalid_probability_rows = 0
total_invalid_xg_rows = 0


def reset_log() -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"=== soccer_cleaner RUN "
            f"{datetime.now(timezone.utc).isoformat()} ===\n"
        )


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{msg}\n"
        )


def normalize_league(value: str) -> str:
    if not value:
        return ""

    cleaned = value.strip().lower()
    return LEAGUE_MAP.get(cleaned, cleaned)


def finite_float(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def pct_to_decimal(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    cleaned = text.rstrip("%").strip()
    number = finite_float(cleaned)

    if number is None:
        return None

    return round(number / 100.0, 6)


def validate_1x2_probabilities(row):
    converted = {}

    for col in PROB_COLS:
        raw_value = row.get(col)

        if raw_value is None or not str(raw_value).strip():
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

        if value < 0.0 or value > 1.0:
            return (
                False,
                {},
                None,
                f"out-of-range {col}={value} "
                f"raw={raw_value!r}",
            )

        converted[col] = value

    probability_sum = round(
        sum(converted.values()),
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
            f"invalid 1X2 sum="
            f"{probability_sum:.6f} "
            f"tolerance="
            f"{PROB_SUM_TOLERANCE:.6f}",
        )

    return (
        True,
        converted,
        probability_sum,
        "",
    )


def validate_xg_fields(row):
    parsed = {}

    for col in XG_COLS:
        raw_value = row.get(col)

        if raw_value is None or not str(raw_value).strip():
            return (
                False,
                {},
                None,
                None,
                f"missing {col}",
            )

        value = finite_float(raw_value)

        if value is None:
            return (
                False,
                {},
                None,
                None,
                f"non-numeric {col}={raw_value!r}",
            )

        if value < 0.0:
            return (
                False,
                {},
                None,
                None,
                f"negative {col}={value}",
            )

        parsed[col] = value

    component_total = round(
        parsed["home_xg"]
        + parsed["away_xg"],
        6,
    )

    total_difference = abs(
        round(
            parsed["expected_total_goals"]
            - component_total,
            6,
        )
    )

    if total_difference > XG_TOTAL_TOLERANCE:
        return (
            False,
            parsed,
            component_total,
            total_difference,
            f"inconsistent expected_total_goals="
            f"{parsed['expected_total_goals']:.6f} "
            f"home_plus_away="
            f"{component_total:.6f} "
            f"difference="
            f"{total_difference:.6f} "
            f"tolerance="
            f"{XG_TOTAL_TOLERANCE:.6f}",
        )

    return (
        True,
        parsed,
        component_total,
        total_difference,
        "",
    )


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


def clean_sportsbook():
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
                    league_norm = (
                        normalize_league(
                            row.get(
                                "league",
                                "",
                            )
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
                        fieldnames=SB_FIELDS,
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


def clean_predictions(
    game_id_index: dict,
):
    global total_missing_ids
    global total_invalid_probability_rows
    global total_invalid_xg_rows

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
                invalid_xg_rows = 0

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

                            sum_text = (
                                f"{probability_sum:.6f}"
                                if probability_sum
                                is not None
                                else "unavailable"
                            )

                            log(
                                "  REJECT INVALID 1X2 | "
                                f"file={pred_file} | "
                                f"line={line_number} | "
                                f"league={row_league} | "
                                f"match_date="
                                f"{match_date} | "
                                f"home_team="
                                f"{home_team} | "
                                f"away_team="
                                f"{away_team} | "
                                f"raw="
                                f"{raw_probabilities} | "
                                f"converted="
                                f"{converted_probabilities or 'unavailable'} | "
                                f"sum={sum_text} | "
                                f"reason="
                                f"{probability_error}"
                            )

                            continue

                        (
                            xg_valid,
                            parsed_xg,
                            component_total,
                            total_difference,
                            xg_error,
                        ) = validate_xg_fields(
                            row
                        )

                        if not xg_valid:
                            invalid_xg_rows += 1
                            total_invalid_xg_rows += 1

                            raw_xg = {
                                col: row.get(
                                    col,
                                    "",
                                )
                                for col
                                in XG_COLS
                            }

                            component_text = (
                                f"{component_total:.6f}"
                                if component_total
                                is not None
                                else "unavailable"
                            )

                            difference_text = (
                                f"{total_difference:.6f}"
                                if total_difference
                                is not None
                                else "unavailable"
                            )

                            log(
                                "  REJECT INVALID XG | "
                                f"file={pred_file} | "
                                f"line={line_number} | "
                                f"league={row_league} | "
                                f"match_date="
                                f"{match_date} | "
                                f"home_team="
                                f"{home_team} | "
                                f"away_team="
                                f"{away_team} | "
                                f"raw={raw_xg} | "
                                f"parsed="
                                f"{parsed_xg or 'unavailable'} | "
                                f"home_plus_away="
                                f"{component_text} | "
                                f"difference="
                                f"{difference_text} | "
                                f"tolerance="
                                f"{XG_TOTAL_TOLERANCE:.6f} | "
                                f"reason={xg_error}"
                            )

                            continue

                        for col in PROB_COLS:
                            row[col] = (
                                converted_probabilities[
                                    col
                                ]
                            )

                        for col in XG_COLS:
                            row[col] = (
                                parsed_xg[col]
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
                        invalid_xg_rows,
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
                    f"invalid 1X2 rejected, "
                    f"{invalid_xg_rows} "
                    f"invalid xG rejected)"
                )

            except Exception as e:
                log(
                    "ERROR processing "
                    f"predictions {pred_file}: "
                    f"{e}\n"
                    f"{traceback.format_exc()}"
                )


def main():
    reset_log()

    log("START")

    log(
        "1X2 validation enabled | "
        f"sum_tolerance="
        f"{PROB_SUM_TOLERANCE:.6f} | "
        "renormalization=disabled"
    )

    log(
        "xG validation enabled | "
        f"total_tolerance="
        f"{XG_TOTAL_TOLERANCE:.6f} | "
        "negative_values=reject"
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
        prob_rejected,
        xg_rejected,
    ) in pred_files_written:

        log(
            f"  FILE: {path} "
            f"({rows} rows, "
            f"{missing} missing game_id, "
            f"{prob_rejected} "
            f"invalid 1X2 rejected, "
            f"{xg_rejected} "
            f"invalid xG rejected)"
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

    log(
        "Total invalid xG "
        "prediction rows rejected: "
        f"{total_invalid_xg_rows}"
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