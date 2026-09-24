#!/usr/bin/env python3
"""Shared infrastructure for league-specific soccer ML inference."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def clean_team(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    return unicodedata.normalize("NFKC", text).casefold()


def normalize_game_id(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def normalize_identity_value(column: str, value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if column == "league":
        return text.casefold()
    if column in ("home_team", "away_team"):
        cleaned = clean_team(text)
        return None if pd.isna(cleaned) else cleaned
    if column == "match_date":
        parsed = pd.to_datetime(
            text.replace("_", "-"),
            errors="coerce",
        )
        return text if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")
    return text


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
    *,
    merge_file_re,
) -> list[str]:
    cutoff = datetime.strptime(
        cutoff_date,
        "%Y_%m_%d",
    ).date()
    dates: set[str] = set()

    if not merge_dir.exists():
        return []

    for path in merge_dir.glob("*.csv"):
        match = merge_file_re.match(path.name)
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
    *,
    league_label: str,
):
    task, algorithm, filename = item
    path = model_path(root, task, algorithm, filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Required {league_label} model missing: {path}"
        )

    return joblib.load(path)


def load_all_bundles(
    joblib,
    root: Path,
    *,
    model_registry: dict,
    model_keys: tuple,
    league_label: str,
) -> dict[str, object]:
    bundles = {}

    for key in model_keys:
        bundles[key] = load_bundle(
            joblib,
            root,
            model_registry[key],
            league_label=league_label,
        )

    return bundles


def valid_numeric_odds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )
    return numeric[numeric > 1.0]


def row_model_completeness(
    row: pd.Series,
    *,
    model_odds_columns: tuple[str, ...],
) -> tuple[int, int]:
    model_count = 0

    for column in model_odds_columns:
        if column not in row.index:
            continue
        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(value) and float(value) > 1.0:
            model_count += 1

    return model_count, int(row.notna().sum())


def validate_duplicate_identity(
    game_id: str,
    group: pd.DataFrame,
    *,
    identity_columns: tuple[str, ...],
    league_label: str,
) -> None:
    conflicts = []

    for column in identity_columns:
        values = []
        for value in group[column]:
            normalized = normalize_identity_value(
                column,
                value,
            )
            if normalized is not None:
                values.append(normalized)
        unique_values = list(dict.fromkeys(values))
        if len(unique_values) > 1:
            conflicts.append((column, unique_values))

    if conflicts:
        detail = "; ".join(
            f"{conflict_column}={conflict_values}"
            for conflict_column, conflict_values in conflicts
        )
        raise RuntimeError(
            f"{league_label} inference stopped: "
            f"game_id {game_id} maps to conflicting "
            f"match identities: {detail}"
        )


def fill_identity_from_group(
    row: pd.Series,
    group: pd.DataFrame,
    *,
    identity_columns: tuple[str, ...],
) -> pd.Series:
    for column in identity_columns:
        if normalize_identity_value(column, row[column]) is not None:
            continue
        for candidate in group[column]:
            if normalize_identity_value(column, candidate) is not None:
                row[column] = candidate
                break

    return row


def consolidate_duplicate_games(
    current: pd.DataFrame,
    *,
    model_odds_columns: tuple[str, ...],
    identity_columns: tuple[str, ...],
    league_label: str,
    detailed_bad_ids: bool,
) -> pd.DataFrame:
    current = current.copy()
    current["game_id"] = current["game_id"].map(normalize_game_id)

    bad_game_ids = current["game_id"].isna()

    if bad_game_ids.any():
        if detailed_bad_ids:
            bad = current.loc[
                bad_game_ids,
                [
                    identity_column
                    for identity_column in (
                        "game_id",
                        "match_date",
                        "home_team",
                        "away_team",
                    )
                    if identity_column in current.columns
                ],
            ]
            raise RuntimeError(
                f"{league_label} inference stopped: "
                "blank or invalid game_id rows:\n"
                + bad.to_string(index=False)
            )

        raise RuntimeError(
            f"{league_label} inference stopped: "
            "blank or invalid game_id."
        )

    if not current["game_id"].duplicated().any():
        return current.reset_index(drop=True)

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
            consolidated_rows.append(group.iloc[0].copy())
            continue

        duplicate_game_count += 1
        removed_row_count += len(group) - 1

        validate_duplicate_identity(
            str(game_id),
            group,
            identity_columns=identity_columns,
            league_label=league_label,
        )

        ranked_positions = sorted(
            range(len(group)),
            key=lambda position: (
                row_model_completeness(
                    group.iloc[position],
                    model_odds_columns=model_odds_columns,
                )[0],
                row_model_completeness(
                    group.iloc[position],
                    model_odds_columns=model_odds_columns,
                )[1],
                -position,
            ),
            reverse=True,
        )

        row = group.iloc[ranked_positions[0]].copy()
        row = fill_identity_from_group(
            row,
            group,
            identity_columns=identity_columns,
        )

        conflicting_odds = []

        for column in model_odds_columns:
            if column not in group.columns:
                continue

            valid = valid_numeric_odds(group[column])
            unique_valid = pd.unique(valid.astype(float))

            if len(unique_valid) > 1:
                conflicting_odds.append(column)

            base_value = pd.to_numeric(
                pd.Series([row[column]]),
                errors="coerce",
            ).iloc[0]

            base_is_valid = (
                pd.notna(base_value)
                and float(base_value) > 1.0
            )

            if not base_is_valid and not valid.empty:
                row[column] = float(valid.iloc[0])

        consolidated_rows.append(row)

        message = (
            f"{league_label} ML inference: consolidated "
            f"{len(group)} sportsbook rows for game_id "
            f"{game_id} into 1 row"
        )

        if conflicting_odds:
            message += (
                "; differing available odds in "
                + ", ".join(conflicting_odds)
                + " â€” kept values from the most-complete "
                "row and used other rows only to fill "
                "missing model inputs"
            )

        print(message + ".")

    consolidated = pd.DataFrame(
        consolidated_rows,
        columns=current.columns,
    ).reset_index(drop=True)

    if consolidated["game_id"].duplicated().any():
        raise RuntimeError(
            f"{league_label} inference stopped: "
            "duplicate game_id remained after consolidation."
        )

    print(
        f"{league_label} ML inference: sportsbook "
        "duplicate consolidation complete: "
        f"{duplicate_game_count} game(s), "
        f"{removed_row_count} duplicate row(s) removed."
    )

    return consolidated


def make_feature_frame(
    frame: pd.DataFrame,
    *,
    role_source_columns: dict[str, str],
    league_label: str,
) -> pd.DataFrame:
    dates = pd.to_datetime(
        frame["match_date"]
        .astype("string")
        .str.strip()
        .str.replace("_", "-", regex=False),
        errors="coerce",
    )

    if dates.isna().any():
        bad = frame.loc[
            dates.isna(),
            [
                "game_id",
                "match_date",
                "home_team",
                "away_team",
            ],
        ]
        raise RuntimeError(
            f"{league_label} inference stopped: "
            "unparseable match_date rows:\n"
            + bad.to_string(index=False)
        )

    features = pd.DataFrame(index=frame.index)
    features["_date_ordinal"] = dates.map(
        lambda date_value: int(date_value.toordinal())
    )
    features["_home_team_clean"] = frame[
        "home_team"
    ].map(clean_team)
    features["_away_team_clean"] = frame[
        "away_team"
    ].map(clean_team)

    for role, source_col in role_source_columns.items():
        if source_col not in frame.columns:
            features[role] = np.nan
            continue
        values = pd.to_numeric(
            frame[source_col],
            errors="coerce",
        )
        features[role] = values.mask(values <= 1.0)

    return features


def validate_predictions(
    predicted: pd.DataFrame,
    *,
    probability_columns: tuple[str, ...],
    goal_columns: tuple[str, ...],
    league_label: str,
) -> None:
    required = list(
        probability_columns
        + goal_columns
    )

    missing = [
        required_column
        for required_column in required
        if required_column not in predicted.columns
    ]

    if missing:
        raise RuntimeError(
            f"{league_label} inference stopped: "
            f"required model outputs absent: {missing}"
        )

    for column in probability_columns:
        values = pd.to_numeric(
            predicted[column],
            errors="coerce",
        ).to_numpy(float)

        if (
            not np.isfinite(values).all()
            or (values < 0.0).any()
            or (values > 1.0).any()
        ):
            raise RuntimeError(
                f"{league_label} inference stopped: "
                f"invalid probability output in {column}."
            )

    for column in goal_columns:
        values = pd.to_numeric(
            predicted[column],
            errors="coerce",
        ).to_numpy(float)

        if (
            not np.isfinite(values).all()
            or (values < 0.0).any()
        ):
            raise RuntimeError(
                f"{league_label} inference stopped: "
                f"invalid goal output in {column}."
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
            f"{league_label} inference stopped: "
            "1X2 probabilities do not sum to 1."
        )

    for first, second in (
        ("ml_over25_prob", "ml_under25_prob"),
        ("ml_over35_prob", "ml_under35_prob"),
        ("ml_btts_yes_prob", "ml_btts_no_prob"),
    ):
        pair_sum = predicted[
            [first, second]
        ].sum(axis=1).to_numpy(float)

        if not np.allclose(
            pair_sum,
            1.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"{league_label} inference stopped: "
                f"{first} + {second} does not equal 1."
            )


def enrich_merge_file(
    path: Path,
    predictions: pd.DataFrame,
    *,
    league_label: str,
) -> int:
    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    if frame.empty:
        return 0

    if "game_id" not in frame.columns:
        raise RuntimeError(
            f"{league_label} inference stopped: "
            f"game_id absent from merged file {path}"
        )

    frame["game_id"] = frame[
        "game_id"
    ].map(normalize_game_id)

    if frame["game_id"].isna().any():
        raise RuntimeError(
            f"{league_label} inference stopped: "
            f"blank game_id in merged file {path}"
        )

    if frame["game_id"].duplicated().any():
        dupes = frame.loc[
            frame["game_id"].duplicated(
                keep=False
            ),
            "game_id",
        ].tolist()

        raise RuntimeError(
            f"{league_label} inference stopped: "
            "duplicate game_id in merged "
            f"file {path}: {dupes}"
        )

    pred = predictions.copy()
    pred["game_id"] = pred[
        "game_id"
    ].map(normalize_game_id)

    ml_cols = [
        column
        for column in pred.columns
        if column != "game_id"
    ]

    existing = [
        column
        for column in ml_cols
        if column in frame.columns
    ]

    if existing:
        frame = frame.drop(columns=existing)

    out = frame.merge(
        pred,
        how="left",
        on="game_id",
        validate="one_to_one",
    )

    missing_predictions = out[
        ml_cols
    ].isna().all(axis=1)

    if missing_predictions.any():
        bad_columns = [
            column
            for column in (
                "game_id",
                "home_team",
                "away_team",
            )
            if column in out.columns
        ]

        bad = out.loc[
            missing_predictions,
            bad_columns,
        ]

        raise RuntimeError(
            f"{league_label} inference stopped: "
            "merged rows have no model prediction "
            f"in {path}:\n"
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
    bundles,
    *,
    league_key: str,
    league_label: str,
    merge_suffixes: tuple[str, ...],
    predict_func,
    enrich_func,
) -> list[tuple[str, int]]:
    merge_dir = soccer_root / "01_merge"

    merge_paths = [
        merge_dir
        / f"{date_text}_{league_key}_{suffix}.csv"
        for suffix in merge_suffixes
    ]

    existing_merge_paths = [
        candidate_path
        for candidate_path in merge_paths
        if candidate_path.exists()
    ]

    if not existing_merge_paths:
        return []

    sportsbook_path = (
        soccer_root
        / "00_intake"
        / "sportsbook"
        / "normalized"
        / f"{date_text}_{league_key}.csv"
    )

    if not sportsbook_path.exists():
        raise FileNotFoundError(
            f"{league_label} inference stopped: "
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
            f"{league_label} inference stopped: "
            "normalized sportsbook file is empty "
            f"for {date_text}: {sportsbook_path}"
        )

    predictions = predict_func(
        bundles,
        current,
    )

    updated = []

    for path in existing_merge_paths:
        rows = enrich_func(
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
        f"{league_label} ML inference complete for "
        f"{date_text}: {len(predictions)} match "
        f"prediction(s), {len(updated)} merge "
        "file(s) enriched."
    )

    for name, rows in updated:
        print(
            f"  enriched {name}: "
            f"{rows} row(s)"
        )

    return updated


class InferenceSupport:
    def __init__(
        self,
        *,
        league_key: str,
        league_label: str,
        merge_file_re,
        model_registry: dict,
        model_keys: tuple,
        model_odds_columns: tuple[str, ...],
        identity_columns: tuple[str, ...],
        role_source_columns: dict[str, str],
        probability_columns: tuple[str, ...],
        goal_columns: tuple[str, ...],
        merge_suffixes: tuple[str, ...],
        detailed_bad_ids: bool,
    ):
        self.league_key = league_key
        self.league_label = league_label
        self.merge_file_re = merge_file_re
        self.model_registry = model_registry
        self.model_keys = model_keys
        self.model_odds_columns = model_odds_columns
        self.identity_columns = identity_columns
        self.role_source_columns = role_source_columns
        self.probability_columns = probability_columns
        self.goal_columns = goal_columns
        self.merge_suffixes = merge_suffixes
        self.detailed_bad_ids = detailed_bad_ids
        self.predict_func = None

    @classmethod
    def from_module(
        cls,
        namespace: dict,
        league_key: str,
        league_label: str,
        detailed_bad_ids: bool,
    ):
        goal_keys = namespace.get(
            "GOAL_MODEL_KEYS",
            (),
        )

        model_keys = (
            namespace["BASE_MODEL_KEYS"]
            + goal_keys
            + namespace["SECOND_STAGE_KEYS"]
        )

        return cls(
            league_key=league_key,
            league_label=league_label,
            merge_file_re=namespace["MERGE_FILE_RE"],
            model_registry=namespace["MODEL_REGISTRY"],
            model_keys=model_keys,
            model_odds_columns=namespace["MODEL_ODDS_COLUMNS"],
            identity_columns=namespace["IDENTITY_COLUMNS"],
            role_source_columns=namespace["ROLE_SOURCE_COLUMNS"],
            probability_columns=namespace.get("PROBABILITY_COLUMNS", ()),
            goal_columns=namespace.get("GOAL_COLUMNS", ()),
            merge_suffixes=namespace["MERGE_SUFFIXES"],
            detailed_bad_ids=detailed_bad_ids,
        )

    @staticmethod
    def resolve_date(
        raw: str | None,
    ) -> str:
        return resolve_date(raw)

    def discover_merge_dates(
        self,
        merge_dir: Path,
        cutoff_date: str,
    ) -> list[str]:
        return discover_merge_dates(
            merge_dir,
            cutoff_date,
            merge_file_re=self.merge_file_re,
        )

    def load_all_bundles(
        self,
        joblib,
        root: Path,
    ) -> dict[str, object]:
        return load_all_bundles(
            joblib,
            root,
            model_registry=self.model_registry,
            model_keys=self.model_keys,
            league_label=self.league_label,
        )

    def consolidate_duplicate_games(
        self,
        current: pd.DataFrame,
    ) -> pd.DataFrame:
        return consolidate_duplicate_games(
            current,
            model_odds_columns=self.model_odds_columns,
            identity_columns=self.identity_columns,
            league_label=self.league_label,
            detailed_bad_ids=self.detailed_bad_ids,
        )

    def make_feature_frame(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        return make_feature_frame(
            frame,
            role_source_columns=self.role_source_columns,
            league_label=self.league_label,
        )

    def validate_predictions(
        self,
        predicted: pd.DataFrame,
    ) -> None:
        validate_predictions(
            predicted,
            probability_columns=self.probability_columns,
            goal_columns=self.goal_columns,
            league_label=self.league_label,
        )

    def enrich_merge_file(
        self,
        path: Path,
        predictions: pd.DataFrame,
    ) -> int:
        return enrich_merge_file(
            path,
            predictions,
            league_label=self.league_label,
        )

    def process_date(
        self,
        date_text: str,
        soccer_root: Path,
        bundles,
    ) -> list[tuple[str, int]]:
        if self.predict_func is None:
            raise RuntimeError(
                f"{self.league_label} inference support "
                "has no prediction function bound."
            )

        return process_date(
            date_text,
            soccer_root,
            bundles,
            league_key=self.league_key,
            league_label=self.league_label,
            merge_suffixes=self.merge_suffixes,
            predict_func=self.predict_func,
            enrich_func=self.enrich_merge_file,
        )
