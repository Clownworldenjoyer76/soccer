# docs/win/soccer/scripts/01_merge/epl_ml_infer.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MODEL_REGISTRY = {
    "1x2": ("1x2", "extra_trees", "wrapper_raw.joblib"),
    "over25": ("over25", "lightgbm", "wrapper_raw.joblib"),
    "over35": ("over35", "xgboost", "wrapper_calibrated.joblib"),
    "btts": ("btts", "random_forest", "wrapper_raw.joblib"),
    "goals": ("goals", "lightgbm", "goal_bundle.joblib"),
    "1x2_predictability": ("1x2_predictability", "extra_trees", "wrapper_raw.joblib"),
    "1x2_skip": ("1x2_skip", "extra_trees", "wrapper_raw.joblib"),
    "over25_predictability": ("over25_predictability", "lightgbm", "wrapper_raw.joblib"),
    "over25_skip": ("over25_skip", "lightgbm", "wrapper_raw.joblib"),
    "over35_predictability": ("over35_predictability", "xgboost", "wrapper_raw.joblib"),
    "over35_skip": ("over35_skip", "xgboost", "wrapper_raw.joblib"),
    "btts_predictability": ("btts_predictability", "random_forest", "wrapper_raw.joblib"),
    "btts_skip": ("btts_skip", "random_forest", "wrapper_raw.joblib"),
}

ROLE_SOURCE_COLUMNS = {
    "role_1x2_home_odds": "dk_home_decimal",
    "role_1x2_draw_odds": "dk_draw_decimal",
    "role_1x2_away_odds": "dk_away_decimal",
    "role_over25_odds": "dk_over25_decimal",
    "role_under25_odds": "dk_under25_decimal",
}

BASE_MODEL_KEYS = ("1x2", "over25", "over35", "btts", "goals")

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

MERGE_FILE_RE = re.compile(
    r"^(\d{4}_\d{2}_\d{2})_epl_(match_odds|total_25|total_35|btts)\.csv$",
    re.IGNORECASE,
)


def clean_team(value):
    if pd.isna(value):
        return pd.NA

    return unicodedata.normalize(
        "NFKC",
        str(value).strip(),
    ).casefold()


def resolve_date(raw: str | None) -> str:
    value = (raw or "").strip()

    if not value:
        value = datetime.now(
            ZoneInfo("America/New_York")
        ).strftime("%Y_%m_%d")

    value = value.replace("-", "_")

    datetime.strptime(value, "%Y_%m_%d")

    return value


def discover_merge_dates(
    merge_dir: Path,
    cutoff_date: str,
) -> list[str]:

    cutoff = datetime.strptime(
        cutoff_date,
        "%Y_%m_%d",
    ).date()

    dates: set[str] = set()

    if not merge_dir.exists():
        return []

    for path in merge_dir.glob("*_epl_*.csv"):

        match = MERGE_FILE_RE.match(path.name)

        if not match:
            continue

        date_text = match.group(1)

        file_date = datetime.strptime(
            date_text,
            "%Y_%m_%d",
        ).date()

        if file_date <= cutoff:
            dates.add(date_text)

    return sorted(dates)


def model_path(
    root: Path,
    task: str,
    algorithm: str,
    filename: str,
) -> Path:

    return (
        root
        / "models"
        / task
        / "production-compatible"
        / algorithm
        / filename
    )


def load_bundle(
    joblib,
    root: Path,
    item,
):

    task, algorithm, filename = item

    path = model_path(
        root,
        task,
        algorithm,
        filename,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Required EPL model missing: {path}"
        )

    return joblib.load(path)


def load_all_bundles(
    joblib,
    root: Path,
) -> dict[str, object]:

    bundles = {}

    for key in BASE_MODEL_KEYS + SECOND_STAGE_KEYS:

        bundles[key] = load_bundle(
            joblib,
            root,
            MODEL_REGISTRY[key],
        )

    return bundles


def make_feature_frame(
    epl: pd.DataFrame,
) -> pd.DataFrame:

    dates = pd.to_datetime(
        epl["match_date"]
        .astype("string")
        .str.strip()
        .str.replace("_", "-", regex=False),
        errors="coerce",
    )

    if dates.isna().any():

        bad = epl.loc[
            dates.isna(),
            [
                "game_id",
                "match_date",
                "home_team",
                "away_team",
            ],
        ]

        raise RuntimeError(
            "EPL inference stopped: "
            "unparseable match_date rows:\n"
            + bad.to_string(index=False)
        )

    X = pd.DataFrame(index=epl.index)

    X["_date_ordinal"] = dates.map(
        lambda d: int(d.toordinal())
    )

    X["_home_team_clean"] = epl[
        "home_team"
    ].map(clean_team)

    X["_away_team_clean"] = epl[
        "away_team"
    ].map(clean_team)

    for role, source_col in ROLE_SOURCE_COLUMNS.items():

        if source_col not in epl.columns:
            X[role] = np.nan
            continue

        values = pd.to_numeric(
            epl[source_col],
            errors="coerce",
        )

        values = values.mask(
            values <= 1.0
        )

        X[role] = values

    return X


def deduplicate_current(
    current: pd.DataFrame,
) -> pd.DataFrame:

    current = current.copy()

    current["game_id"] = (
        current["game_id"]
        .astype("string")
        .str.strip()
    )

    duplicate_ids = current.loc[
        current["game_id"].duplicated(keep=False),
        "game_id",
    ].dropna().unique().tolist()

    if not duplicate_ids:
        return current

    compare_columns = [
        c
        for c in (
            "league",
            "match_date",
            "home_team",
            "away_team",
            "dk_home_decimal",
            "dk_draw_decimal",
            "dk_away_decimal",
            "dk_over25_decimal",
            "dk_under25_decimal",
        )
        if c in current.columns
    ]

    conflicting = []

    for game_id in duplicate_ids:

        group = current.loc[
            current["game_id"].eq(game_id),
            compare_columns,
        ].copy()

        normalized = pd.DataFrame(
            index=group.index
        )

        for column in compare_columns:

            if column in (
                "dk_home_decimal",
                "dk_draw_decimal",
                "dk_away_decimal",
                "dk_over25_decimal",
                "dk_under25_decimal",
            ):
                normalized[column] = pd.to_numeric(
                    group[column],
                    errors="coerce",
                )
            else:
                normalized[column] = (
                    group[column]
                    .astype("string")
                    .str.strip()
                    .fillna("<MISSING>")
                )

        normalized = normalized.fillna(
            "<MISSING>"
        )

        if len(normalized.drop_duplicates()) > 1:
            conflicting.append(game_id)

    if conflicting:
        raise RuntimeError(
            "EPL inference stopped: "
            "duplicate game_id rows contain conflicting "
            "model inputs: "
            f"{conflicting}"
        )

    print(
        "EPL ML inference: removing duplicate sportsbook "
        "row(s) for game_id(s): "
        f"{duplicate_ids}"
    )

    current = (
        current
        .drop_duplicates(
            subset=["game_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return current


def predict_frame(
    bundles: dict[str, object],
    current: pd.DataFrame,
) -> pd.DataFrame:

    missing = [
        c
        for c in REQUIRED_CURRENT_COLUMNS
        if c not in current.columns
    ]

    if missing:
        raise RuntimeError(
            "EPL inference stopped: "
            f"required sportsbook columns absent: {missing}"
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
            "non-EPL rows in an EPL sportsbook file."
        )

    current = deduplicate_current(
        current
    )

    X = make_feature_frame(current)

    predicted = current[
        ["game_id"]
    ].copy()

    for key in BASE_MODEL_KEYS + SECOND_STAGE_KEYS:

        pred = bundles[key].predict(X)

        if len(pred) != len(current):
            raise RuntimeError(
                "EPL inference stopped: "
                f"{key} returned wrong row count."
            )

        for col in pred.columns:
            predicted[col] = pred[
                col
            ].to_numpy()

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
        c
        for c in required
        if c not in predicted.columns
    ]

    if missing_outputs:
        raise RuntimeError(
            "EPL inference stopped: "
            f"required model outputs absent: {missing_outputs}"
        )

    one_x_two_sum = predicted[
        [
            "ml_home_prob",
            "ml_draw_prob",
            "ml_away_prob",
        ]
    ].sum(axis=1).to_numpy(float)

    if not np.allclose(
        one_x_two_sum,
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "EPL inference stopped: "
            "1X2 probabilities do not sum to 1."
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
        ].sum(axis=1).to_numpy(float)

        if not np.allclose(
            pair_sum,
            1.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "EPL inference stopped: "
                f"{a} + {b} does not equal 1."
            )

    return predicted


def enrich_merge_file(
    path: Path,
    predictions: pd.DataFrame,
) -> int:

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if df.empty:
        return 0

    if "game_id" not in df.columns:
        raise RuntimeError(
            "EPL inference stopped: "
            f"game_id absent from merged file {path}"
        )

    merge_game_ids = df[
        "game_id"
    ].astype("string")

    if merge_game_ids.duplicated().any():
        raise RuntimeError(
            "EPL inference stopped: "
            f"duplicate game_id in merged file {path}"
        )

    pred = predictions.copy()

    pred["game_id"] = pred[
        "game_id"
    ].astype("string")

    df["game_id"] = df[
        "game_id"
    ].astype("string")

    ml_cols = [
        c
        for c in pred.columns
        if c != "game_id"
    ]

    existing = [
        c
        for c in ml_cols
        if c in df.columns
    ]

    if existing:
        df = df.drop(
            columns=existing
        )

    out = df.merge(
        pred,
        how="left",
        on="game_id",
        validate="one_to_one",
    )

    missing_predictions = (
        out[ml_cols]
        .isna()
        .all(axis=1)
    )

    if missing_predictions.any():

        bad_columns = [
            c
            for c in (
                "game_id",
                "home_team",
                "away_team",
            )
            if c in out.columns
        ]

        bad = out.loc[
            missing_predictions,
            bad_columns,
        ]

        raise RuntimeError(
            "EPL inference stopped: "
            f"merged rows have no model prediction in {path}:\n"
            + bad.to_string(index=False)
        )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    out.to_csv(
        temp,
        index=False,
    )

    temp.replace(path)

    return len(out)


def process_date(
    date_text: str,
    soccer_root: Path,
    bundles: dict[str, object],
) -> list[tuple[str, int]]:

    merge_dir = (
        soccer_root
        / "01_merge"
    )

    merge_paths = [
        merge_dir
        / f"{date_text}_epl_{suffix}.csv"
        for suffix in MERGE_SUFFIXES
    ]

    existing_merge_paths = [
        path
        for path in merge_paths
        if path.exists()
    ]

    if not existing_merge_paths:
        return []

    sportsbook_path = (
        soccer_root
        / "00_intake"
        / "sportsbook"
        / "normalized"
        / f"{date_text}_epl.csv"
    )

    if not sportsbook_path.exists():
        raise FileNotFoundError(
            "EPL inference stopped: "
            f"merge files exist for {date_text}, "
            "but normalized sportsbook input is missing: "
            f"{sportsbook_path}"
        )

    current = pd.read_csv(
        sportsbook_path,
        low_memory=False,
    )

    if current.empty:
        raise RuntimeError(
            "EPL inference stopped: "
            "normalized sportsbook file is empty "
            f"for {date_text}: {sportsbook_path}"
        )

    predictions = predict_frame(
        bundles,
        current,
    )

    updated = []

    for path in existing_merge_paths:

        rows = enrich_merge_file(
            path,
            predictions,
        )

        updated.append(
            (
                path.name,
                rows,
            )
        )

    print(
        f"EPL ML inference complete for {date_text}: "
        f"{len(predictions)} match prediction(s), "
        f"{len(updated)} merge file(s) enriched."
    )

    for name, rows in updated:
        print(
            f"  enriched {name}: "
            f"{rows} row(s)"
        )

    return updated


def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Run EPL-only ML inference across all historical "
            "EPL merge dates through the requested cutoff date "
            "and attach outputs to EPL merge files."
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
            "All EPL merge dates on or before this date "
            "are processed. Defaults to RUN_DATE or today's "
            "America/New_York date."
        ),
    )

    args = ap.parse_args()

    soccer_root = Path(
        args.soccer_root
    )

    epl_root = Path(
        args.epl_root
    )

    cutoff_date = resolve_date(
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
            "Required EPL wrapper module missing: "
            f"{wrapper_module}"
        )

    sys.path.insert(
        0,
        str(epl_root),
    )

    import joblib
    import soccer_model_wrapper  # noqa: F401

    dates = discover_merge_dates(
        merge_dir,
        cutoff_date,
    )

    if not dates:
        print(
            "EPL ML inference: "
            "no EPL merge dates found through "
            f"{cutoff_date}; nothing to do."
        )
        return

    print(
        "EPL ML inference: "
        f"processing {len(dates)} EPL merge date(s) "
        f"through {cutoff_date}."
    )

    bundles = load_all_bundles(
        joblib,
        epl_root,
    )

    processed_dates = 0
    updated_files = 0
    updated_rows = 0

    for date_text in dates:

        updated = process_date(
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
                for _, rows in updated
            )

    print(
        "EPL ML historical inference complete: "
        f"{processed_dates} date(s), "
        f"{updated_files} merge file(s), "
        f"{updated_rows} total merged row(s) enriched."
    )


if __name__ == "__main__":
    main()
