#!/usr/bin/env python3
# docs/win/soccer/scripts/01_merge/ligue1_ml_infer.py
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

import _ml_infer_common as ml_common


MODEL_REGISTRY = {
    "1x2": ("1x2", "logistic", "wrapper_calibrated.joblib"),
    "over25": ("over25", "xgboost", "wrapper_calibrated.joblib"),
    "over35": ("over35", "extra_trees", "wrapper_raw.joblib"),
    "btts": ("btts", "lightgbm", "wrapper_calibrated.joblib"),
    "goals_catboost": ("goals", "catboost", "goal_bundle.joblib"),
    "goals_extra_trees": ("goals", "extra_trees", "goal_bundle.joblib"),
    "1x2_predictability": ("1x2_predictability", "logistic", "wrapper_raw.joblib"),
    "1x2_skip": ("1x2_skip", "logistic", "wrapper_raw.joblib"),
    "over25_predictability": ("over25_predictability", "xgboost", "wrapper_raw.joblib"),
    "over25_skip": ("over25_skip", "xgboost", "wrapper_raw.joblib"),
    "over35_predictability": ("over35_predictability", "extra_trees", "wrapper_raw.joblib"),
    "over35_skip": ("over35_skip", "extra_trees", "wrapper_raw.joblib"),
    "btts_predictability": ("btts_predictability", "lightgbm", "wrapper_raw.joblib"),
    "btts_skip": ("btts_skip", "lightgbm", "wrapper_raw.joblib"),
}

ROLE_SOURCE_COLUMNS = {
    "role_1x2_home_odds": "dk_home_decimal",
    "role_1x2_draw_odds": "dk_draw_decimal",
    "role_1x2_away_odds": "dk_away_decimal",
    "role_over25_odds": "dk_over25_decimal",
    "role_under25_odds": "dk_under25_decimal",
}
MODEL_ODDS_COLUMNS = tuple(ROLE_SOURCE_COLUMNS.values())

BASE_MODEL_KEYS = ("1x2", "over25", "over35", "btts")
GOAL_MODEL_KEYS = ("goals_catboost", "goals_extra_trees")
SECOND_STAGE_KEYS = (
    "1x2_predictability", "1x2_skip",
    "over25_predictability", "over25_skip",
    "over35_predictability", "over35_skip",
    "btts_predictability", "btts_skip",
)

MERGE_SUFFIXES = ("match_odds", "total_25", "total_35", "btts")
REQUIRED_CURRENT_COLUMNS = ("game_id", "league", "match_date", "home_team", "away_team")
IDENTITY_COLUMNS = ("league", "match_date", "home_team", "away_team")

MERGE_FILE_RE = re.compile(
    r"^(\d{4}_\d{2}_\d{2})_ligue1_(match_odds|total_25|total_35|btts)\.csv$",
    re.IGNORECASE,
)

PROBABILITY_COLUMNS = (
    "ml_home_prob", "ml_draw_prob", "ml_away_prob",
    "ml_over25_prob", "ml_under25_prob",
    "ml_over35_prob", "ml_under35_prob",
    "ml_btts_yes_prob", "ml_btts_no_prob",
    "ml_1x2_predictability", "ml_1x2_skip_prob",
    "ml_over25_predictability", "ml_over25_skip_prob",
    "ml_over35_predictability", "ml_over35_skip_prob",
    "ml_btts_predictability", "ml_btts_skip_prob",
)
GOAL_COLUMNS = ("ml_home_goals", "ml_away_goals")

































SUPPORT = ml_common.InferenceSupport.from_module(
    globals(),
    'ligue1',
    'Ligue 1',
    False,
)

def predict_frame(bundles, current: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CURRENT_COLUMNS if c not in current.columns]
    if missing:
        raise RuntimeError(
            "Ligue 1 inference stopped: required sportsbook columns absent: "
            f"{missing}"
        )

    current = SUPPORT.consolidate_duplicate_games(current)
    league = current["league"].astype("string").str.strip().str.casefold()
    if not current.loc[~league.eq("ligue1")].empty:
        raise RuntimeError(
            "Ligue 1 pipeline inference received non-Ligue 1 rows "
            "in a Ligue 1 sportsbook file."
        )

    features = SUPPORT.make_feature_frame(current)
    predicted = current[["game_id"]].copy()

    # Validation winner is the calibrated 1X2 Logistic model. Its Draw
    # component beat the standalone Draw model on binary validation log loss,
    # so Home/Draw/Away are taken directly from this single calibrated model.
    one_x_two = bundles["1x2"].predict(features)
    if len(one_x_two) != len(current):
        raise RuntimeError(
            "Ligue 1 inference stopped: 1x2 returned wrong row count."
        )
    for column in ("ml_home_prob", "ml_draw_prob", "ml_away_prob"):
        if column not in one_x_two.columns:
            raise RuntimeError(
                f"Ligue 1 inference stopped: 1x2 output missing {column}."
            )
        predicted[column] = one_x_two[column].to_numpy()

    for key in ("over25", "over35", "btts"):
        pred = bundles[key].predict(features)
        if len(pred) != len(current):
            raise RuntimeError(
                f"Ligue 1 inference stopped: {key} returned wrong row count."
            )
        for column in pred.columns:
            predicted[column] = pred[column].to_numpy()

    # Mixed validation goal winners.
    home_goals = bundles["goals_catboost"].home.predict(features)
    away_goals = bundles["goals_extra_trees"].away.predict(features)
    predicted["ml_home_goals"] = home_goals["ml_home_goals"].to_numpy()
    predicted["ml_away_goals"] = away_goals["ml_away_goals"].to_numpy()

    for key in SECOND_STAGE_KEYS:
        pred = bundles[key].predict(features)
        if len(pred) != len(current):
            raise RuntimeError(
                f"Ligue 1 inference stopped: {key} returned wrong row count."
            )
        for column in pred.columns:
            predicted[column] = pred[column].to_numpy()

    SUPPORT.validate_predictions(predicted)
    return predicted







SUPPORT.predict_func = predict_frame

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run Ligue 1-only ML inference across all historical Ligue 1 "
            "merge dates through the requested cutoff date and attach outputs "
            "to Ligue 1 merge files."
        )
    )
    ap.add_argument("--soccer-root", default="docs/win/soccer")
    ap.add_argument("--ligue1-root", default="docs/win/soccer/ml/ligue1")
    ap.add_argument(
        "--date",
        default=os.environ.get("RUN_DATE", ""),
        help=(
            "Inclusive cutoff date (YYYY_MM_DD or YYYY-MM-DD). "
            "All Ligue 1 merge dates on or before this date are processed."
        ),
    )
    args = ap.parse_args()

    soccer_root = Path(args.soccer_root)
    ligue1_root = Path(args.ligue1_root)
    cutoff_date = SUPPORT.resolve_date(args.date)
    merge_dir = soccer_root / "01_merge"

    wrapper_module = ligue1_root / "soccer_model_wrapper.py"
    if not wrapper_module.exists():
        raise FileNotFoundError(
            f"Required Ligue 1 wrapper module missing: {wrapper_module}"
        )

    sys.path.insert(0, str(ligue1_root))
    import joblib
    import soccer_model_wrapper  # noqa: F401

    dates = SUPPORT.discover_merge_dates(merge_dir, cutoff_date)
    if not dates:
        print(
            "Ligue 1 ML inference: no Ligue 1 merge dates found "
            f"through {cutoff_date}; nothing to do."
        )
        return

    print(
        "Ligue 1 ML inference: processing "
        f"{len(dates)} Ligue 1 merge date(s) through {cutoff_date}."
    )
    bundles = SUPPORT.load_all_bundles(joblib, ligue1_root)

    processed_dates = 0
    updated_files = 0
    updated_rows = 0

    for date_text in dates:
        updated = SUPPORT.process_date(date_text, soccer_root, bundles)
        if updated:
            processed_dates += 1
            updated_files += len(updated)
            updated_rows += sum(rows for _, rows in updated)

    print(
        "Ligue 1 ML historical inference complete: "
        f"{processed_dates} date(s), {updated_files} merge file(s), "
        f"{updated_rows} total merged row(s) enriched."
    )


if __name__ == "__main__":
    main()
