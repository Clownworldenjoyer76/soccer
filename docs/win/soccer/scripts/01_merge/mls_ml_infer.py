#!/usr/bin/env python3
# docs/win/soccer/scripts/01_merge/mls_ml_infer.py
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
    "1x2": ("1x2", "random_forest", "wrapper_calibrated.joblib"),
    "draw": ("draw", "logistic", "wrapper_raw.joblib"),
    "over25": ("over25", "logistic", "wrapper_raw.joblib"),
    "over35": ("over35", "catboost", "wrapper_calibrated.joblib"),
    "btts": ("btts", "xgboost", "wrapper_raw.joblib"),
    "goals_extra_trees": ("goals", "extra_trees", "goal_bundle.joblib"),
    "goals_random_forest": ("goals", "random_forest", "goal_bundle.joblib"),
    "1x2_predictability": ("1x2_predictability", "random_forest", "wrapper_raw.joblib"),
    "1x2_skip": ("1x2_skip", "random_forest", "wrapper_raw.joblib"),
    "draw_predictability": ("draw_predictability", "logistic", "wrapper_raw.joblib"),
    "draw_skip": ("draw_skip", "logistic", "wrapper_raw.joblib"),
    "over25_predictability": ("over25_predictability", "logistic", "wrapper_raw.joblib"),
    "over25_skip": ("over25_skip", "logistic", "wrapper_raw.joblib"),
    "over35_predictability": ("over35_predictability", "catboost", "wrapper_raw.joblib"),
    "over35_skip": ("over35_skip", "catboost", "wrapper_raw.joblib"),
    "btts_predictability": ("btts_predictability", "xgboost", "wrapper_raw.joblib"),
    "btts_skip": ("btts_skip", "xgboost", "wrapper_raw.joblib"),
}

ROLE_SOURCE_COLUMNS = {
    "role_1x2_home_odds": "dk_home_decimal",
    "role_1x2_draw_odds": "dk_draw_decimal",
    "role_1x2_away_odds": "dk_away_decimal",
    "role_over25_odds": "dk_over25_decimal",
    "role_under25_odds": "dk_under25_decimal",
}
MODEL_ODDS_COLUMNS = tuple(ROLE_SOURCE_COLUMNS.values())

BASE_MODEL_KEYS = ("1x2", "draw", "over25", "over35", "btts")
GOAL_MODEL_KEYS = ("goals_extra_trees", "goals_random_forest")
SECOND_STAGE_KEYS = (
    "1x2_predictability", "1x2_skip",
    "draw_predictability", "draw_skip",
    "over25_predictability", "over25_skip",
    "over35_predictability", "over35_skip",
    "btts_predictability", "btts_skip",
)

MERGE_SUFFIXES = ("match_odds", "total_25", "total_35", "btts")
REQUIRED_CURRENT_COLUMNS = ("game_id", "league", "match_date", "home_team", "away_team")
IDENTITY_COLUMNS = ("league", "match_date", "home_team", "away_team")

MERGE_FILE_RE = re.compile(
    r"^(\d{4}_\d{2}_\d{2})_mls_(match_odds|total_25|total_35|btts)\.csv$",
    re.IGNORECASE,
)

PROBABILITY_COLUMNS = (
    "ml_home_prob", "ml_draw_prob", "ml_away_prob",
    "ml_over25_prob", "ml_under25_prob",
    "ml_over35_prob", "ml_under35_prob",
    "ml_btts_yes_prob", "ml_btts_no_prob",
    "ml_1x2_predictability", "ml_1x2_skip_prob",
    "ml_draw_predictability", "ml_draw_skip_prob",
    "ml_over25_predictability", "ml_over25_skip_prob",
    "ml_over35_predictability", "ml_over35_skip_prob",
    "ml_btts_predictability", "ml_btts_skip_prob",
)
GOAL_COLUMNS = ("ml_home_goals", "ml_away_goals")


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
        parsed = pd.to_datetime(text.replace("_", "-"), errors="coerce")
        return text if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")
    return text


def resolve_date(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        value = datetime.now(ZoneInfo("America/New_York")).strftime("%Y_%m_%d")
    value = value.replace("-", "_")
    datetime.strptime(value, "%Y_%m_%d")
    return value


def discover_merge_dates(merge_dir: Path, cutoff_date: str) -> list[str]:
    cutoff = datetime.strptime(cutoff_date, "%Y_%m_%d").date()
    dates = set()
    if not merge_dir.exists():
        return []
    for path in merge_dir.glob("*_mls_*.csv"):
        match = MERGE_FILE_RE.match(path.name)
        if not match:
            continue
        date_text = match.group(1)
        if datetime.strptime(date_text, "%Y_%m_%d").date() <= cutoff:
            dates.add(date_text)
    return sorted(dates)


def model_path(root: Path, task: str, algorithm: str, filename: str) -> Path:
    return root / "models" / task / "production-compatible" / algorithm / filename


def load_bundle(joblib, root: Path, item):
    task, algorithm, filename = item
    path = model_path(root, task, algorithm, filename)
    if not path.exists():
        raise FileNotFoundError(f"Required MLS model missing: {path}")
    return joblib.load(path)


def load_all_bundles(joblib, root: Path):
    return {
        key: load_bundle(joblib, root, MODEL_REGISTRY[key])
        for key in BASE_MODEL_KEYS + GOAL_MODEL_KEYS + SECOND_STAGE_KEYS
    }


def valid_numeric_odds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[numeric > 1.0]


def row_model_completeness(row: pd.Series):
    model_count = 0
    for column in MODEL_ODDS_COLUMNS:
        if column not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 1.0:
            model_count += 1
    return model_count, int(row.notna().sum())


def validate_duplicate_identity(game_id: str, group: pd.DataFrame):
    conflicts = []
    for column in IDENTITY_COLUMNS:
        values = []
        for value in group[column]:
            normalized = normalize_identity_value(column, value)
            if normalized is not None:
                values.append(normalized)
        unique_values = list(dict.fromkeys(values))
        if len(unique_values) > 1:
            conflicts.append((column, unique_values))
    if conflicts:
        detail = "; ".join(f"{c}={v}" for c, v in conflicts)
        raise RuntimeError(
            "MLS inference stopped: "
            f"game_id {game_id} maps to conflicting match identities: {detail}"
        )


def fill_identity_from_group(row: pd.Series, group: pd.DataFrame):
    for column in IDENTITY_COLUMNS:
        if normalize_identity_value(column, row[column]) is not None:
            continue
        for candidate in group[column]:
            if normalize_identity_value(column, candidate) is not None:
                row[column] = candidate
                break
    return row


def consolidate_duplicate_games(current: pd.DataFrame) -> pd.DataFrame:
    current = current.copy()
    current["game_id"] = current["game_id"].map(normalize_game_id)
    if current["game_id"].isna().any():
        raise RuntimeError("MLS inference stopped: blank or invalid game_id.")
    if not current["game_id"].duplicated().any():
        return current.reset_index(drop=True)

    consolidated_rows = []
    duplicate_game_count = 0
    removed_row_count = 0

    for game_id, group in current.groupby("game_id", sort=False, dropna=False):
        group = group.copy()
        if len(group) == 1:
            consolidated_rows.append(group.iloc[0].copy())
            continue

        duplicate_game_count += 1
        removed_row_count += len(group) - 1
        validate_duplicate_identity(str(game_id), group)

        ranked_positions = sorted(
            range(len(group)),
            key=lambda pos: (
                row_model_completeness(group.iloc[pos])[0],
                row_model_completeness(group.iloc[pos])[1],
                -pos,
            ),
            reverse=True,
        )
        row = group.iloc[ranked_positions[0]].copy()
        row = fill_identity_from_group(row, group)
        conflicting_odds = []

        for column in MODEL_ODDS_COLUMNS:
            if column not in group.columns:
                continue
            valid = valid_numeric_odds(group[column])
            if len(pd.unique(valid.astype(float))) > 1:
                conflicting_odds.append(column)
            base = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if not (pd.notna(base) and float(base) > 1.0) and not valid.empty:
                row[column] = float(valid.iloc[0])

        consolidated_rows.append(row)
        message = (
            f"MLS ML inference: consolidated {len(group)} sportsbook rows "
            f"for game_id {game_id} into 1 row"
        )
        if conflicting_odds:
            message += (
                "; differing available odds in "
                + ", ".join(conflicting_odds)
                + " — kept values from the most-complete row and used other rows "
                  "only to fill missing model inputs"
            )
        print(message + ".")

    consolidated = pd.DataFrame(
        consolidated_rows, columns=current.columns
    ).reset_index(drop=True)

    if consolidated["game_id"].duplicated().any():
        raise RuntimeError(
            "MLS inference stopped: duplicate game_id remained after consolidation."
        )

    print(
        "MLS ML inference: sportsbook duplicate consolidation complete: "
        f"{duplicate_game_count} game(s), {removed_row_count} duplicate row(s) removed."
    )
    return consolidated


def make_feature_frame(mls: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(
        mls["match_date"].astype("string").str.strip().str.replace("_", "-", regex=False),
        errors="coerce",
    )
    if dates.isna().any():
        bad = mls.loc[
            dates.isna(), ["game_id", "match_date", "home_team", "away_team"]
        ]
        raise RuntimeError(
            "MLS inference stopped: unparseable match_date rows:\n"
            + bad.to_string(index=False)
        )

    X = pd.DataFrame(index=mls.index)
    X["_date_ordinal"] = dates.map(lambda d: int(d.toordinal()))
    X["_home_team_clean"] = mls["home_team"].map(clean_team)
    X["_away_team_clean"] = mls["away_team"].map(clean_team)

    for role, source_col in ROLE_SOURCE_COLUMNS.items():
        if source_col not in mls.columns:
            X[role] = np.nan
            continue
        values = pd.to_numeric(mls[source_col], errors="coerce")
        X[role] = values.mask(values <= 1.0)
    return X


def validate_predictions(predicted: pd.DataFrame):
    required = list(PROBABILITY_COLUMNS + GOAL_COLUMNS)
    missing = [c for c in required if c not in predicted.columns]
    if missing:
        raise RuntimeError(
            f"MLS inference stopped: required model outputs absent: {missing}"
        )

    for column in PROBABILITY_COLUMNS:
        values = pd.to_numeric(predicted[column], errors="coerce").to_numpy(float)
        if (
            not np.isfinite(values).all()
            or (values < 0.0).any()
            or (values > 1.0).any()
        ):
            raise RuntimeError(
                f"MLS inference stopped: invalid probability output in {column}."
            )

    for column in GOAL_COLUMNS:
        values = pd.to_numeric(predicted[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise RuntimeError(
                f"MLS inference stopped: invalid goal output in {column}."
            )

    if not np.allclose(
        predicted[["ml_home_prob", "ml_draw_prob", "ml_away_prob"]]
        .sum(axis=1).to_numpy(float),
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "MLS inference stopped: 1X2 probabilities do not sum to 1."
        )

    for a, b in (
        ("ml_over25_prob", "ml_under25_prob"),
        ("ml_over35_prob", "ml_under35_prob"),
        ("ml_btts_yes_prob", "ml_btts_no_prob"),
    ):
        if not np.allclose(
            predicted[[a, b]].sum(axis=1).to_numpy(float),
            1.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"MLS inference stopped: {a} + {b} does not equal 1."
            )


def predict_frame(bundles, current: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CURRENT_COLUMNS if c not in current.columns]
    if missing:
        raise RuntimeError(
            "MLS inference stopped: required sportsbook columns absent: "
            f"{missing}"
        )

    current = consolidate_duplicate_games(current)
    league = current["league"].astype("string").str.strip().str.casefold()
    if not current.loc[~league.eq("mls")].empty:
        raise RuntimeError(
            "MLS pipeline inference received non-MLS rows "
            "in a MLS sportsbook file."
        )

    X = make_feature_frame(current)
    predicted = current[["game_id"]].copy()

    # Validation winner for 1X2 supplies Home/Away structure.
    one_x_two = bundles["1x2"].predict(X)
    # Standalone raw Draw won binary Draw validation.
    draw = bundles["draw"].predict(X)

    home_raw = pd.to_numeric(one_x_two["ml_home_prob"], errors="coerce").to_numpy(float)
    away_raw = pd.to_numeric(one_x_two["ml_away_prob"], errors="coerce").to_numpy(float)
    draw_final = pd.to_numeric(draw["ml_draw_prob"], errors="coerce").to_numpy(float)
    home_away_total = home_raw + away_raw

    if (
        not np.isfinite(home_raw).all()
        or not np.isfinite(away_raw).all()
        or not np.isfinite(draw_final).all()
        or (home_away_total <= 0.0).any()
        or (draw_final < 0.0).any()
        or (draw_final > 1.0).any()
    ):
        raise RuntimeError(
            "MLS inference stopped: invalid component probabilities "
            "while combining calibrated Random Forest Home/Away with standalone Logistic Draw."
        )

    remaining = 1.0 - draw_final
    predicted["ml_home_prob"] = remaining * (home_raw / home_away_total)
    predicted["ml_draw_prob"] = draw_final
    predicted["ml_away_prob"] = remaining * (away_raw / home_away_total)

    for key in ("over25", "over35", "btts"):
        pred = bundles[key].predict(X)
        if len(pred) != len(current):
            raise RuntimeError(
                f"MLS inference stopped: {key} returned wrong row count."
            )
        for column in pred.columns:
            predicted[column] = pred[column].to_numpy()

    # Mixed validation goal winners.
    home_goals = bundles["goals_extra_trees"].home.predict(X)
    away_goals = bundles["goals_random_forest"].away.predict(X)
    predicted["ml_home_goals"] = home_goals["ml_home_goals"].to_numpy()
    predicted["ml_away_goals"] = away_goals["ml_away_goals"].to_numpy()

    for key in SECOND_STAGE_KEYS:
        pred = bundles[key].predict(X)
        if len(pred) != len(current):
            raise RuntimeError(
                f"MLS inference stopped: {key} returned wrong row count."
            )
        for column in pred.columns:
            predicted[column] = pred[column].to_numpy()

    validate_predictions(predicted)
    return predicted


def enrich_merge_file(path: Path, predictions: pd.DataFrame) -> int:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return 0

    if "game_id" not in df.columns:
        raise RuntimeError(
            f"MLS inference stopped: game_id absent from merged file {path}"
        )

    df["game_id"] = df["game_id"].map(normalize_game_id)
    if df["game_id"].isna().any():
        raise RuntimeError(
            f"MLS inference stopped: blank game_id in merged file {path}"
        )
    if df["game_id"].duplicated().any():
        dupes = df.loc[df["game_id"].duplicated(keep=False), "game_id"].tolist()
        raise RuntimeError(
            f"MLS inference stopped: duplicate game_id in merged file {path}: {dupes}"
        )

    pred = predictions.copy()
    pred["game_id"] = pred["game_id"].map(normalize_game_id)
    ml_cols = [c for c in pred.columns if c != "game_id"]

    existing = [c for c in ml_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    out = df.merge(pred, how="left", on="game_id", validate="one_to_one")
    missing_predictions = out[ml_cols].isna().all(axis=1)
    if missing_predictions.any():
        bad_columns = [
            c for c in ("game_id", "home_team", "away_team") if c in out.columns
        ]
        bad = out.loc[missing_predictions, bad_columns]
        raise RuntimeError(
            "MLS inference stopped: merged rows have no model prediction "
            f"in {path}:\n" + bad.to_string(index=False)
        )

    temp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(temp, index=False)
    temp.replace(path)
    return len(out)


def process_date(date_text: str, soccer_root: Path, bundles):
    merge_dir = soccer_root / "01_merge"
    merge_paths = [
        merge_dir / f"{date_text}_mls_{suffix}.csv"
        for suffix in MERGE_SUFFIXES
    ]
    existing_merge_paths = [p for p in merge_paths if p.exists()]
    if not existing_merge_paths:
        return []

    sportsbook_path = (
        soccer_root / "00_intake" / "sportsbook" / "normalized"
        / f"{date_text}_mls.csv"
    )
    if not sportsbook_path.exists():
        raise FileNotFoundError(
            "MLS inference stopped: merge files exist for "
            f"{date_text}, but normalized sportsbook input is missing: "
            f"{sportsbook_path}"
        )

    current = pd.read_csv(sportsbook_path, low_memory=False)
    if current.empty:
        raise RuntimeError(
            "MLS inference stopped: normalized sportsbook file is empty "
            f"for {date_text}: {sportsbook_path}"
        )

    predictions = predict_frame(bundles, current)
    updated = []
    for path in existing_merge_paths:
        rows = enrich_merge_file(path, predictions)
        updated.append((path.name, rows))

    print(
        "MLS ML inference complete for "
        f"{date_text}: {len(predictions)} match prediction(s), "
        f"{len(updated)} merge file(s) enriched."
    )
    for name, rows in updated:
        print(f"  enriched {name}: {rows} row(s)")
    return updated


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run MLS-only ML inference across all historical MLS "
            "merge dates through the requested cutoff date and attach outputs "
            "to MLS merge files."
        )
    )
    ap.add_argument("--soccer-root", default="docs/win/soccer")
    ap.add_argument("--mls-root", default="docs/win/soccer/ml/mls")
    ap.add_argument(
        "--date",
        default=os.environ.get("RUN_DATE", ""),
        help=(
            "Inclusive cutoff date (YYYY_MM_DD or YYYY-MM-DD). "
            "All MLS merge dates on or before this date are processed."
        ),
    )
    args = ap.parse_args()

    soccer_root = Path(args.soccer_root)
    mls_root = Path(args.mls_root)
    cutoff_date = resolve_date(args.date)
    merge_dir = soccer_root / "01_merge"

    wrapper_module = mls_root / "soccer_model_wrapper.py"
    if not wrapper_module.exists():
        raise FileNotFoundError(
            f"Required MLS wrapper module missing: {wrapper_module}"
        )

    sys.path.insert(0, str(mls_root))
    import joblib
    import soccer_model_wrapper  # noqa: F401

    dates = discover_merge_dates(merge_dir, cutoff_date)
    if not dates:
        print(
            "MLS ML inference: no MLS merge dates found "
            f"through {cutoff_date}; nothing to do."
        )
        return

    print(
        "MLS ML inference: processing "
        f"{len(dates)} MLS merge date(s) through {cutoff_date}."
    )
    bundles = load_all_bundles(joblib, mls_root)

    processed_dates = 0
    updated_files = 0
    updated_rows = 0

    for date_text in dates:
        updated = process_date(date_text, soccer_root, bundles)
        if updated:
            processed_dates += 1
            updated_files += len(updated)
            updated_rows += sum(rows for _, rows in updated)

    print(
        "MLS ML historical inference complete: "
        f"{processed_dates} date(s), {updated_files} merge file(s), "
        f"{updated_rows} total merged row(s) enriched."
    )


if __name__ == "__main__":
    main()
