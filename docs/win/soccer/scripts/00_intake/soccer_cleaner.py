#!/usr/bin/env python3
# docs/win/soccer/scripts/00_intake/soccer_cleaner.py

import csv
import math
import re
import traceback
from collections import Counter
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
XG_TOTAL_TOLERANCE = 0.01

IDENTITY_REASONS = (
    "missing_sportsbook_game_id",
    "date_mismatch",
    "team_name_mismatch",
    "no_sportsbook_fixture",
)

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
    "identity_status",
    "identity_reason",
    "identity_detail",
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
total_prediction_input_rows = 0
total_prediction_rows_written = 0
total_identity_reasons = Counter()


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

    return number if math.isfinite(number) else None


def pct_to_decimal(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    number = finite_float(text.rstrip("%").strip())

    if number is None:
        return None

    return round(number / 100.0, 6)


def validate_1x2_probabilities(row):
    converted = {}

    for col in PROB_COLS:
        raw_value = row.get(col)

        if raw_value is None or not str(raw_value).strip():
            return False, {}, None, f"missing {col}"

        value = pct_to_decimal(raw_value)

        if value is None:
            return False, {}, None, f"non-numeric {col}={raw_value!r}"

        if value < 0.0 or value > 1.0:
            return (
                False,
                {},
                None,
                f"out-of-range {col}={value} raw={raw_value!r}",
            )

        converted[col] = value

    probability_sum = round(sum(converted.values()), 6)
    sum_error = abs(round(probability_sum - 1.0, 6))

    if sum_error > PROB_SUM_TOLERANCE:
        return (
            False,
            converted,
            probability_sum,
            f"invalid 1X2 sum={probability_sum:.6f} "
            f"tolerance={PROB_SUM_TOLERANCE:.6f}",
        )

    return True, converted, probability_sum, ""


def validate_xg_fields(row):
    parsed = {}

    for col in XG_COLS:
        raw_value = row.get(col)

        if raw_value is None or not str(raw_value).strip():
            return False, {}, None, None, f"missing {col}"

        value = finite_float(raw_value)

        if value is None:
            return False, {}, None, None, f"non-numeric {col}={raw_value!r}"

        if value < 0.0:
            return False, {}, None, None, f"negative {col}={value}"

        parsed[col] = value

    component_total = round(
        parsed["home_xg"] + parsed["away_xg"],
        6,
    )

    total_difference = abs(
        round(
            parsed["expected_total_goals"] - component_total,
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
            f"home_plus_away={component_total:.6f} "
            f"difference={total_difference:.6f} "
            f"tolerance={XG_TOTAL_TOLERANCE:.6f}",
        )

    return (
        True,
        parsed,
        component_total,
        total_difference,
        "",
    )


def clean_sportsbook():
    for sb_file in sorted(
        SPORTSBOOK_DIR.glob("*/*.csv")
    ):
        if sb_file.parent.name == "normalized":
            continue

        if not DATE_PAT.search(sb_file.stem):
            continue

        if not sb_file.stem.endswith("_soccer"):
            continue

        try:
            date_str = DATE_PAT.search(
                sb_file.stem
            ).group(0)

            log(
                f"SPORTSBOOK: processing {sb_file}"
            )

            by_league = {}

            with open(
                sb_file,
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    league_norm = normalize_league(
                        row.get("league", "")
                    )

                    if not league_norm:
                        log(
                            f"  SKIP row — "
                            f"no league value: {row}"
                        )
                        continue

                    by_league.setdefault(
                        league_norm,
                        [],
                    ).append(row)

            for league_norm, rows in by_league.items():
                out_path = (
                    SB_NORM_DIR
                    / f"{date_str}_{league_norm}.csv"
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
                f"ERROR processing sportsbook "
                f"{sb_file}: {e}\n"
                f"{traceback.format_exc()}"
            )


def build_sportsbook_fixture_catalog() -> dict:
    exact = {}
    by_league_date = {}
    exact_pair_dates = {}

    for sb_file in sorted(
        SB_NORM_DIR.glob("*.csv")
    ):
        if not DATE_PAT.search(sb_file.stem):
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
                        row.get("league", "")
                    )

                    match_date = (
                        row.get("match_date")
                        or ""
                    ).strip()

                    home_team = (
                        row.get("home_team")
                        or ""
                    ).strip()

                    away_team = (
                        row.get("away_team")
                        or ""
                    ).strip()

                    game_id = (
                        row.get("game_id")
                        or ""
                    ).strip()

                    if (
                        not league
                        or not match_date
                        or not home_team
                        or not away_team
                    ):
                        continue

                    fixture = {
                        "league": league,
                        "match_date": match_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "game_id": game_id,
                        "source_file": str(sb_file),
                    }

                    exact_key = (
                        league,
                        match_date,
                        home_team,
                        away_team,
                    )

                    existing = exact.get(
                        exact_key
                    )

                    if (
                        existing is None
                        or (
                            not existing.get("game_id")
                            and game_id
                        )
                    ):
                        exact[exact_key] = fixture

                    elif (
                        game_id
                        and existing.get("game_id")
                        and game_id
                        != existing.get("game_id")
                    ):
                        log(
                            "SPORTSBOOK IDENTITY CONFLICT | "
                            f"fixture={exact_key} | "
                            f"game_ids="
                            f"{existing.get('game_id')},"
                            f"{game_id}"
                        )

                    by_league_date.setdefault(
                        (
                            league,
                            match_date,
                        ),
                        [],
                    ).append(
                        fixture
                    )

                    exact_pair_dates.setdefault(
                        (
                            league,
                            home_team,
                            away_team,
                        ),
                        set(),
                    ).add(
                        match_date
                    )

        except Exception as e:
            log(
                f"ERROR reading sportsbook "
                f"identity catalog {sb_file}: "
                f"{e}\n"
                f"{traceback.format_exc()}"
            )

    return {
        "exact": exact,
        "by_league_date": by_league_date,
        "exact_pair_dates": exact_pair_dates,
    }


def nearby_exact_pair_dates(
    dates,
    match_date: str,
    max_days: int = 1,
):
    try:
        target = datetime.strptime(
            match_date,
            "%Y_%m_%d",
        ).date()

    except ValueError:
        return []

    nearby = []

    for value in dates:
        try:
            candidate = datetime.strptime(
                value,
                "%Y_%m_%d",
            ).date()

        except ValueError:
            continue

        if (
            abs(
                (
                    candidate
                    - target
                ).days
            )
            <= max_days
            and candidate != target
        ):
            nearby.append(
                value
            )

    return sorted(
        nearby
    )


def classify_prediction_identity(
    catalog: dict,
    league: str,
    match_date: str,
    home_team: str,
    away_team: str,
):
    exact_key = (
        league,
        match_date,
        home_team,
        away_team,
    )

    exact_fixture = (
        catalog["exact"].get(
            exact_key
        )
    )

    if exact_fixture is not None:
        game_id = (
            exact_fixture.get("game_id")
            or ""
        ).strip()

        if game_id:
            return (
                game_id,
                "matched",
                "matched_game_id",
                "",
            )

        return (
            "",
            "unmatched",
            "missing_sportsbook_game_id",
            "exact sportsbook fixture exists "
            "but its game_id is blank",
        )

    other_dates = nearby_exact_pair_dates(
        catalog["exact_pair_dates"].get(
            (
                league,
                home_team,
                away_team,
            ),
            set(),
        ),
        match_date,
    )

    if other_dates:
        return (
            "",
            "unmatched",
            "date_mismatch",
            "exact teams found in sportsbook "
            "on date(s): "
            + ",".join(other_dates),
        )

    slate_fixtures = (
        catalog["by_league_date"].get(
            (
                league,
                match_date,
            ),
            [],
        )
    )

    prediction_teams = {
        team
        for team in (
            home_team,
            away_team,
        )
        if team
    }

    if slate_fixtures:
        related = []

        for fixture in slate_fixtures:
            sportsbook_teams = {
                fixture["home_team"],
                fixture["away_team"],
            }

            if (
                prediction_teams
                & sportsbook_teams
            ):
                related.append(
                    f"{fixture['home_team']} "
                    f"vs "
                    f"{fixture['away_team']}"
                )

        if related:
            return (
                "",
                "unmatched",
                "team_name_mismatch",
                "same-date sportsbook fixture "
                "shares a team name: "
                + " ; ".join(related),
            )

    return (
        "",
        "unmatched",
        "no_sportsbook_fixture",
        "no exact sportsbook fixture found "
        "for league/date/teams",
    )


def clean_predictions(
    catalog: dict,
):
    global total_missing_ids
    global total_invalid_probability_rows
    global total_invalid_xg_rows
    global total_prediction_input_rows
    global total_prediction_rows_written

    for league_dir in sorted(
        PREDICTIONS_DIR.iterdir()
    ):
        if (
            not league_dir.is_dir()
            or league_dir.name
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

                league_norm = normalize_league(
                    league
                )

                out_path = (
                    PRED_NORM_DIR
                    / f"{date_str}_{league_norm}.csv"
                )

                log(
                    f"PREDICTIONS: "
                    f"processing {pred_file}"
                )

                rows_out = []
                input_rows = 0
                invalid_probability_rows = 0
                invalid_xg_rows = 0
                identity_counts = Counter()

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
                        input_rows += 1
                        total_prediction_input_rows += 1

                        match_date = (
                            row.get("match_date")
                            or ""
                        ).strip()

                        home_team = (
                            row.get("home_team")
                            or ""
                        ).strip()

                        away_team = (
                            row.get("away_team")
                            or ""
                        ).strip()

                        row_league = normalize_league(
                            row.get(
                                "league",
                                "",
                            )
                            or league
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
                            invalid_probability_rows += 1
                            total_invalid_probability_rows += 1

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
                                f"match_date={match_date} | "
                                f"home_team={home_team} | "
                                f"away_team={away_team} | "
                                f"raw={raw_probabilities} | "
                                f"converted="
                                f"{converted_probabilities or 'unavailable'} | "
                                f"sum={sum_text} | "
                                f"reason={probability_error}"
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
                                f"match_date={match_date} | "
                                f"home_team={home_team} | "
                                f"away_team={away_team} | "
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
                                parsed_xg[
                                    col
                                ]
                            )

                        (
                            game_id,
                            identity_status,
                            identity_reason,
                            identity_detail,
                        ) = (
                            classify_prediction_identity(
                                catalog,
                                row_league,
                                match_date,
                                home_team,
                                away_team,
                            )
                        )

                        row["game_id"] = game_id
                        row["league"] = row_league
                        row["identity_status"] = (
                            identity_status
                        )
                        row["identity_reason"] = (
                            identity_reason
                        )
                        row["identity_detail"] = (
                            identity_detail
                        )

                        rows_out.append(
                            row
                        )

                        identity_counts[
                            identity_reason
                        ] += 1

                        total_identity_reasons[
                            identity_reason
                        ] += 1

                        if not game_id:
                            total_missing_ids += 1

                            log(
                                "  MISSING GAME_ID | "
                                f"file={pred_file} | "
                                f"line={line_number} | "
                                f"date={match_date} | "
                                f"league={row_league} | "
                                f"home_team={home_team} | "
                                f"away_team={away_team} | "
                                f"reason={identity_reason} | "
                                f"detail={identity_detail}"
                            )

                accounted = (
                    len(rows_out)
                    + invalid_probability_rows
                    + invalid_xg_rows
                )

                if accounted != input_rows:
                    raise RuntimeError(
                        f"prediction accounting "
                        f"failure for {pred_file}: "
                        f"input_rows={input_rows} "
                        f"accounted={accounted}"
                    )

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

                total_prediction_rows_written += (
                    len(rows_out)
                )

                missing_id = sum(
                    identity_counts[
                        reason
                    ]
                    for reason
                    in IDENTITY_REASONS
                )

                pred_files_written.append(
                    (
                        str(out_path),
                        input_rows,
                        len(rows_out),
                        missing_id,
                        invalid_probability_rows,
                        invalid_xg_rows,
                        dict(
                            identity_counts
                        ),
                    )
                )

                log(
                    "  IDENTITY SUMMARY | "
                    f"date={date_str} | "
                    f"league={league_norm} | "
                    f"input_rows={input_rows} | "
                    f"valid_predictions="
                    f"{len(rows_out)} | "
                    f"matched="
                    f"{identity_counts['matched_game_id']} | "
                    f"missing_identity="
                    f"{missing_id} | "
                    f"no_sportsbook_fixture="
                    f"{identity_counts['no_sportsbook_fixture']} | "
                    f"team_name_mismatch="
                    f"{identity_counts['team_name_mismatch']} | "
                    f"date_mismatch="
                    f"{identity_counts['date_mismatch']} | "
                    f"missing_sportsbook_game_id="
                    f"{identity_counts['missing_sportsbook_game_id']} | "
                    f"invalid_1x2_rejected="
                    f"{invalid_probability_rows} | "
                    f"invalid_xg_rejected="
                    f"{invalid_xg_rows}"
                )

                log(
                    f"  WROTE {out_path} "
                    f"({len(rows_out)} rows retained "
                    f"for model evaluation; "
                    f"{missing_id} cannot enter "
                    f"sportsbook merge)"
                )

            except Exception as e:
                log(
                    f"ERROR processing predictions "
                    f"{pred_file}: {e}\n"
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

    catalog = (
        build_sportsbook_fixture_catalog()
    )

    log(
        "Sportsbook identity catalog built | "
        f"exact_fixtures="
        f"{len(catalog['exact'])} | "
        f"league_date_slates="
        f"{len(catalog['by_league_date'])}"
    )

    clean_predictions(
        catalog
    )

    log("--- SUMMARY ---")

    log(
        f"Sportsbook files written: "
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
        f"Prediction files written: "
        f"{len(pred_files_written)}"
    )

    for (
        path,
        input_rows,
        rows_written,
        missing,
        prob_rejected,
        xg_rejected,
        identity_counts,
    ) in pred_files_written:

        log(
            f"  FILE: {path} | "
            f"input_rows={input_rows} | "
            f"retained={rows_written} | "
            f"missing_game_id={missing} | "
            f"invalid_1X2_rejected="
            f"{prob_rejected} | "
            f"invalid_xG_rejected="
            f"{xg_rejected} | "
            f"identity_reasons="
            f"{identity_counts}"
        )

    globally_accounted = (
        total_prediction_rows_written
        + total_invalid_probability_rows
        + total_invalid_xg_rows
    )

    if (
        globally_accounted
        != total_prediction_input_rows
    ):
        raise RuntimeError(
            "global prediction accounting "
            "failure | "
            f"input_rows="
            f"{total_prediction_input_rows} | "
            f"accounted="
            f"{globally_accounted}"
        )

    log(
        "PREDICTION ACCOUNTING | "
        f"input_rows="
        f"{total_prediction_input_rows} | "
        f"retained_for_evaluation="
        f"{total_prediction_rows_written} | "
        f"missing_game_id="
        f"{total_missing_ids} | "
        f"invalid_1X2_rejected="
        f"{total_invalid_probability_rows} | "
        f"invalid_xG_rejected="
        f"{total_invalid_xg_rows}"
    )

    log(
        "MISSING IDENTITY REASONS | "
        + " | ".join(
            f"{reason}="
            f"{total_identity_reasons[reason]}"
            for reason
            in IDENTITY_REASONS
        )
    )

    status = (
        "SUCCESS WITH MISSING IDENTITY"
        if total_missing_ids
        else "SUCCESS"
    )

    log(
        f"STATUS: {status}"
    )

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

        log(
            "STATUS: FAILED"
        )

        raise