# docs/win/soccer/scripts/01_merge/epl_ml_infer.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _ml_infer_common as ml_common


MODEL_REGISTRY = {
    "1x2": ("1x2", "extra_trees", "wrapper_raw.joblib"),
    "over25": ("over25", "lightgbm", "wrapper_raw.joblib"),
    "over35": ("over35", "xgboost", "wrapper_calibrated.joblib"),
    "btts": ("btts", "random_forest", "wrapper_raw.joblib"),
    "goals": ("goals", "lightgbm", "goal_bundle.joblib"),
    "1x2_predictability": (
        "1x2_predictability",
        "extra_trees",
        "wrapper_raw.joblib",
    ),
    "1x2_skip": (
        "1x2_skip",
        "extra_trees",
        "wrapper_raw.joblib",
    ),
    "over25_predictability": (
        "over25_predictability",
        "lightgbm",
        "wrapper_raw.joblib",
    ),
    "over25_skip": (
        "over25_skip",
        "lightgbm",
        "wrapper_raw.joblib",
    ),
    "over35_predictability": (
        "over35_predictability",
        "xgboost",
        "wrapper_raw.joblib",
    ),
    "over35_skip": (
        "over35_skip",
        "xgboost",
        "wrapper_raw.joblib",
    ),
    "btts_predictability": (
        "btts_predictability",
        "random_forest",
        "wrapper_raw.joblib",
    ),
    "btts_skip": (
        "btts_skip",
        "random_forest",
        "wrapper_raw.joblib",
    ),
}


ROLE_SOURCE_COLUMNS = {
    "role_1x2_home_odds": "dk_home_decimal",
    "role_1x2_draw_odds": "dk_draw_decimal",
    "role_1x2_away_odds": "dk_away_decimal",
    "role_over25_odds": "dk_over25_decimal",
    "role_under25_odds": "dk_under25_decimal",
}


MODEL_ODDS_COLUMNS = tuple(
    ROLE_SOURCE_COLUMNS.values()
)


BASE_MODEL_KEYS = (
    "1x2",
    "over25",
    "over35",
    "btts",
    "goals",
)


SECOND_STAGE_KEYS = (
    "1x2_predictability",
    "1x2_skip",
    "over25_predictability",
    "over25_skip",
    "over35_predictability",
    "over35_skip",
    "btts_predictability",
    "btts_skip",
)


MERGE_SUFFIXES = (
    "match_odds",
    "total_25",
    "total_35",
    "btts",
)


REQUIRED_CURRENT_COLUMNS = (
    "game_id",
    "league",
    "match_date",
    "home_team",
    "away_team",
)


IDENTITY_COLUMNS = (
    "league",
    "match_date",
    "home_team",
    "away_team",
)


MERGE_FILE_RE = re.compile(
    r"^(\d{4}_\d{2}_\d{2})_epl_"
    r"(match_odds|total_25|total_35|btts)\.csv$",
    re.IGNORECASE,
)































SUPPORT = ml_common.InferenceSupport.from_module(
    globals(),
    'epl',
    'EPL',
    True,
)

def predict_frame(
    bundles: dict[str, object],
    current: pd.DataFrame,
) -> pd.DataFrame:

    missing = [
        column
        for column
        in REQUIRED_CURRENT_COLUMNS
        if column not in current.columns
    ]

    if missing:
        raise RuntimeError(
            "EPL inference stopped: "
            "required sportsbook columns "
            f"absent: {missing}"
        )

    current = (
        SUPPORT.consolidate_duplicate_games(
            current
        )
    )

    league = (
        current["league"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    non_epl = current.loc[
        ~league.eq("epl")
    ]

    if not non_epl.empty:
        raise RuntimeError(
            "EPL pipeline inference received "
            "non-EPL rows in an EPL "
            "sportsbook file."
        )

    features = SUPPORT.make_feature_frame(
        current
    )

    predicted = current[
        ["game_id"]
    ].copy()

    for key in (
        BASE_MODEL_KEYS
        + SECOND_STAGE_KEYS
    ):

        pred = bundles[
            key
        ].predict(features)

        if len(pred) != len(current):
            raise RuntimeError(
                "EPL inference stopped: "
                f"{key} returned wrong "
                "row count."
            )

        for column in pred.columns:
            predicted[column] = (
                pred[column]
                .to_numpy()
            )

    required = [
        "ml_home_prob",
        "ml_draw_prob",
        "ml_away_prob",
        "ml_over25_prob",
        "ml_under25_prob",
        "ml_over35_prob",
        "ml_under35_prob",
        "ml_btts_yes_prob",
        "ml_btts_no_prob",
        "ml_home_goals",
        "ml_away_goals",
    ]

    missing_outputs = [
        column
        for column in required
        if column
        not in predicted.columns
    ]

    if missing_outputs:
        raise RuntimeError(
            "EPL inference stopped: "
            "required model outputs "
            f"absent: {missing_outputs}"
        )

    one_x_two_sum = predicted[
        [
            "ml_home_prob",
            "ml_draw_prob",
            "ml_away_prob",
        ]
    ].sum(
        axis=1
    ).to_numpy(
        float
    )

    if not np.allclose(
        one_x_two_sum,
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "EPL inference stopped: "
            "1X2 probabilities do "
            "not sum to 1."
        )

    for a, b in (
        (
            "ml_over25_prob",
            "ml_under25_prob",
        ),
        (
            "ml_over35_prob",
            "ml_under35_prob",
        ),
        (
            "ml_btts_yes_prob",
            "ml_btts_no_prob",
        ),
    ):

        pair_sum = predicted[
            [a, b]
        ].sum(
            axis=1
        ).to_numpy(
            float
        )

        if not np.allclose(
            pair_sum,
            1.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "EPL inference stopped: "
                f"{a} + {b} does not "
                "equal 1."
            )

    return predicted







SUPPORT.predict_func = predict_frame

def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Run EPL-only ML inference "
            "across all historical EPL "
            "merge dates through the "
            "requested cutoff date and "
            "attach outputs to EPL "
            "merge files."
        )
    )

    ap.add_argument(
        "--soccer-root",
        default="docs/win/soccer",
    )

    ap.add_argument(
        "--epl-root",
        default="docs/win/soccer/ml/epl",
    )

    ap.add_argument(
        "--date",
        default=os.environ.get(
            "RUN_DATE",
            "",
        ),
        help=(
            "Inclusive cutoff date "
            "(YYYY_MM_DD or YYYY-MM-DD). "
            "All EPL merge dates on or "
            "before this date are processed."
        ),
    )

    args = ap.parse_args()

    soccer_root = Path(
        args.soccer_root
    )

    epl_root = Path(
        args.epl_root
    )

    cutoff_date = SUPPORT.resolve_date(
        args.date
    )

    merge_dir = (
        soccer_root
        / "01_merge"
    )

    wrapper_module = (
        epl_root
        / "soccer_model_wrapper.py"
    )

    if not wrapper_module.exists():
        raise FileNotFoundError(
            "Required EPL wrapper module "
            f"missing: {wrapper_module}"
        )

    sys.path.insert(
        0,
        str(epl_root),
    )

    import joblib
    import soccer_model_wrapper  # noqa: F401

    dates = SUPPORT.discover_merge_dates(
        merge_dir,
        cutoff_date,
    )

    if not dates:
        print(
            "EPL ML inference: "
            "no EPL merge dates found "
            f"through {cutoff_date}; "
            "nothing to do."
        )
        return

    print(
        "EPL ML inference: processing "
        f"{len(dates)} EPL merge "
        f"date(s) through {cutoff_date}."
    )

    bundles = SUPPORT.load_all_bundles(
        joblib,
        epl_root,
    )

    processed_dates = 0
    updated_files = 0
    updated_rows = 0

    for date_text in dates:

        updated = SUPPORT.process_date(
            date_text,
            soccer_root,
            bundles,
        )

        if updated:

            processed_dates += 1

            updated_files += len(
                updated
            )

            updated_rows += sum(
                rows
                for _, rows
                in updated
            )

    print(
        "EPL ML historical inference "
        "complete: "
        f"{processed_dates} date(s), "
        f"{updated_files} merge file(s), "
        f"{updated_rows} total merged "
        "row(s) enriched."
    )


if __name__ == "__main__":
    main()
