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


def clean_team(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    return unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()


def normalize_game_id(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]

    return text


def normalize_identity_value(
    column: str,
    value,
):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if column == "league":
        return text.casefold()

    if column in ("home_team", "away_team"):
        cleaned = clean_team(text)

        if pd.isna(cleaned):
            return None

        return cleaned

    if column == "match_date":
        parsed = pd.to_datetime(
            text.replace("_", "-"),
            errors="coerce",
        )

        if pd.isna(parsed):
            return text

        return parsed.strftime("%Y-%m-%d")

    return text


def resolve_date(
    raw: str | None,
) -> str:
    value = (raw or "").strip()

    if not value:
        value = datetime.now(
            ZoneInfo("America/New_York")
        ).strftime("%Y_%m_%d")

    value = value.replace("-", "_")

    datetime.strptime(
        value,
        "%Y_%m_%d",
    )

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

    for path in merge_dir.glob(
        "*_epl_*.csv"
    ):
        match = MERGE_FILE_RE.match(
            path.name
        )

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

    for key in (
        BASE_MODEL_KEYS
        + SECOND_STAGE_KEYS
    ):
        bundles[key] = load_bundle(
            joblib,
            root,
            MODEL_REGISTRY[key],
        )

    return bundles


def valid_numeric_odds(
    series: pd.Series,
) -> pd.Series:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric[
        numeric > 1.0
    ]


def row_model_completeness(
    row: pd.Series,
) -> tuple[int, int]:

    model_count = 0

    for column in MODEL_ODDS_COLUMNS:
        if column not in row.index:
            continue

        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]

        if (
            pd.notna(value)
            and float(value) > 1.0
        ):
            model_count += 1

    total_count = int(
        row.notna().sum()
    )

    return (
        model_count,
        total_count,
    )


def validate_duplicate_identity(
    game_id: str,
    group: pd.DataFrame,
) -> None:

    conflicts = []

    for column in IDENTITY_COLUMNS:
        values = []

        for value in group[column]:
            normalized = (
                normalize_identity_value(
                    column,
                    value,
                )
            )

            if normalized is not None:
                values.append(normalized)

        unique_values = list(
            dict.fromkeys(values)
        )

        if len(unique_values) > 1:
            conflicts.append(
                (
                    column,
                    unique_values,
                )
            )

    if conflicts:
        detail = "; ".join(
            f"{column}={values}"
            for column, values
            in conflicts
        )

        raise RuntimeError(
            "EPL inference stopped: "
            f"game_id {game_id} maps to "
            "conflicting match identities: "
            f"{detail}"
        )


def fill_identity_from_group(
    row: pd.Series,
    group: pd.DataFrame,
) -> pd.Series:

    for column in IDENTITY_COLUMNS:

        current = row[column]

        normalized_current = (
            normalize_identity_value(
                column,
                current,
            )
        )

        if normalized_current is not None:
            continue

        for candidate in group[column]:

            normalized_candidate = (
                normalize_identity_value(
                    column,
                    candidate,
                )
            )

            if normalized_candidate is None:
                continue

            row[column] = candidate
            break

    return row


def consolidate_duplicate_games(
    current: pd.DataFrame,
) -> pd.DataFrame:

    current = current.copy()

    current["game_id"] = current[
        "game_id"
    ].map(normalize_game_id)

    bad_game_ids = current[
        "game_id"
    ].isna()

    if bad_game_ids.any():
        bad = current.loc[
            bad_game_ids,
            [
                c
                for c in (
                    "game_id",
                    "match_date",
                    "home_team",
                    "away_team",
                )
                if c in current.columns
            ],
        ]

        raise RuntimeError(
            "EPL inference stopped: "
            "blank or invalid game_id rows:\n"
            + bad.to_string(index=False)
        )

    if not current[
        "game_id"
    ].duplicated().any():

        return current.reset_index(
            drop=True
        )

    consolidated_rows = []

    duplicate_game_count = 0
    removed_row_count = 0

    for game_id, group in current.groupby(
        "game_id",
        sort=False,
        dropna=False,
    ):

        group = group.copy()

        if len(group) == 1:
            consolidated_rows.append(
                group.iloc[0].copy()
            )
            continue

        duplicate_game_count += 1
        removed_row_count += (
            len(group) - 1
        )

        validate_duplicate_identity(
            str(game_id),
            group,
        )

        ranked_positions = sorted(
            range(len(group)),
            key=lambda position: (
                row_model_completeness(
                    group.iloc[position]
                )[0],
                row_model_completeness(
                    group.iloc[position]
                )[1],
                -position,
            ),
            reverse=True,
        )

        base_position = (
            ranked_positions[0]
        )

        row = group.iloc[
            base_position
        ].copy()

        row = fill_identity_from_group(
            row,
            group,
        )

        conflicting_odds = []

        for column in MODEL_ODDS_COLUMNS:

            if column not in group.columns:
                continue

            valid = valid_numeric_odds(
                group[column]
            )

            unique_valid = pd.unique(
                valid.astype(float)
            )

            if len(unique_valid) > 1:
                conflicting_odds.append(
                    column
                )

            base_value = pd.to_numeric(
                pd.Series(
                    [row[column]]
                ),
                errors="coerce",
            ).iloc[0]

            base_is_valid = (
                pd.notna(base_value)
                and float(base_value) > 1.0
            )

            if (
                not base_is_valid
                and not valid.empty
            ):
                row[column] = float(
                    valid.iloc[0]
                )

        consolidated_rows.append(row)

        message = (
            "EPL ML inference: consolidated "
            f"{len(group)} sportsbook rows "
            f"for game_id {game_id} into 1 row"
        )

        if conflicting_odds:
            message += (
                "; differing available odds in "
                + ", ".join(
                    conflicting_odds
                )
                + " — kept values from the "
                "most-complete row and used "
                "other rows only to fill missing "
                "model inputs"
            )

        print(message + ".")

    consolidated = pd.DataFrame(
        consolidated_rows,
        columns=current.columns,
    ).reset_index(drop=True)

    if consolidated[
        "game_id"
    ].duplicated().any():

        dupes = consolidated.loc[
            consolidated[
                "game_id"
            ].duplicated(
                keep=False
            ),
            "game_id",
        ].tolist()

        raise RuntimeError(
            "EPL inference stopped: "
            "duplicate game_id remained after "
            f"consolidation: {dupes}"
        )

    print(
        "EPL ML inference: sportsbook "
        "duplicate consolidation complete: "
        f"{duplicate_game_count} game(s), "
        f"{removed_row_count} duplicate "
        "row(s) removed."
    )

    return consolidated


def make_feature_frame(
    epl: pd.DataFrame,
) -> pd.DataFrame:

    dates = pd.to_datetime(
        epl["match_date"]
        .astype("string")
        .str.strip()
        .str.replace(
            "_",
            "-",
            regex=False,
        ),
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

    X = pd.DataFrame(
        index=epl.index
    )

    X["_date_ordinal"] = dates.map(
        lambda d: int(
            d.toordinal()
        )
    )

    X["_home_team_clean"] = epl[
        "home_team"
    ].map(clean_team)

    X["_away_team_clean"] = epl[
        "away_team"
    ].map(clean_team)

    for (
        role,
        source_col,
    ) in ROLE_SOURCE_COLUMNS.items():

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
        consolidate_duplicate_games(
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

    X = make_feature_frame(
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
        ].predict(X)

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
            "game_id absent from "
            f"merged file {path}"
        )

    df["game_id"] = df[
        "game_id"
    ].map(normalize_game_id)

    if df[
        "game_id"
    ].isna().any():
        raise RuntimeError(
            "EPL inference stopped: "
            f"blank game_id in merged file {path}"
        )

    merge_game_ids = df[
        "game_id"
    ]

    if merge_game_ids.duplicated().any():

        dupes = df.loc[
            merge_game_ids.duplicated(
                keep=False
            ),
            "game_id",
        ].tolist()

        raise RuntimeError(
            "EPL inference stopped: "
            "duplicate game_id in merged "
            f"file {path}: {dupes}"
        )

    pred = predictions.copy()

    pred["game_id"] = pred[
        "game_id"
    ].map(normalize_game_id)

    ml_cols = [
        column
        for column
        in pred.columns
        if column != "game_id"
    ]

    existing = [
        column
        for column in ml_cols
        if column in df.columns
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
            column
            for column in (
                "game_id",
                "home_team",
                "away_team",
            )
            if column
            in out.columns
        ]

        bad = out.loc[
            missing_predictions,
            bad_columns,
        ]

        raise RuntimeError(
            "EPL inference stopped: "
            "merged rows have no model "
            f"prediction in {path}:\n"
            + bad.to_string(
                index=False
            )
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
        for suffix
        in MERGE_SUFFIXES
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
            "but normalized sportsbook input "
            f"is missing: {sportsbook_path}"
        )

    current = pd.read_csv(
        sportsbook_path,
        low_memory=False,
    )

    if current.empty:
        raise RuntimeError(
            "EPL inference stopped: "
            "normalized sportsbook file "
            f"is empty for {date_text}: "
            f"{sportsbook_path}"
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
        f"EPL ML inference complete for "
        f"{date_text}: "
        f"{len(predictions)} match "
        "prediction(s), "
        f"{len(updated)} merge "
        "file(s) enriched."
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
            "Required EPL wrapper module "
            f"missing: {wrapper_module}"
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
