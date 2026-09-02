#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
    "1x2_predictability", "1x2_skip",
    "over25_predictability", "over25_skip",
    "over35_predictability", "over35_skip",
    "btts_predictability", "btts_skip",
)
MERGE_SUFFIXES = ("match_odds", "total_25", "total_35", "btts")
REQUIRED_CURRENT_COLUMNS = ("game_id", "league", "match_date", "home_team", "away_team")


def clean_team(value):
    if pd.isna(value):
        return pd.NA
    return unicodedata.normalize("NFKC", str(value).strip()).casefold()


def resolve_date(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        value = datetime.now(ZoneInfo("America/New_York")).strftime("%Y_%m_%d")
    value = value.replace("-", "_")
    datetime.strptime(value, "%Y_%m_%d")
    return value


def model_path(root: Path, task: str, algorithm: str, filename: str) -> Path:
    return root / "models" / task / "production-compatible" / algorithm / filename


def load_bundle(joblib, root: Path, item):
    task, algorithm, filename = item
    path = model_path(root, task, algorithm, filename)
    if not path.exists():
        raise FileNotFoundError(f"Required EPL model missing: {path}")
    return joblib.load(path)


def make_feature_frame(epl: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(
        epl["match_date"].astype("string").str.strip().str.replace("_", "-", regex=False),
        errors="coerce",
    )
    if dates.isna().any():
        bad = epl.loc[dates.isna(), ["game_id", "match_date", "home_team", "away_team"]]
        raise RuntimeError(
            "EPL inference stopped: unparseable match_date rows:\n" + bad.to_string(index=False)
        )

    X = pd.DataFrame(index=epl.index)
    X["_date_ordinal"] = dates.map(lambda d: int(d.toordinal()))
    X["_home_team_clean"] = epl["home_team"].map(clean_team)
    X["_away_team_clean"] = epl["away_team"].map(clean_team)

    for role, source_col in ROLE_SOURCE_COLUMNS.items():
        if source_col not in epl.columns:
            X[role] = np.nan
            continue
        values = pd.to_numeric(epl[source_col], errors="coerce")
        values = values.mask(values <= 1.0)
        X[role] = values
    return X


def predict_frame(joblib, root: Path, current: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CURRENT_COLUMNS if c not in current.columns]
    if missing:
        raise RuntimeError(f"EPL inference stopped: required sportsbook columns absent: {missing}")

    league = current["league"].astype("string").str.strip().str.casefold()
    non_epl = current.loc[~league.eq("epl")]
    if not non_epl.empty:
        raise RuntimeError("EPL pipeline inference received non-EPL rows in an EPL sportsbook file.")

    if current["game_id"].astype("string").duplicated().any():
        dupes = current.loc[current["game_id"].astype("string").duplicated(False), "game_id"].tolist()
        raise RuntimeError(f"EPL inference stopped: duplicate game_id in sportsbook input: {dupes}")

    X = make_feature_frame(current)
    predicted = current[["game_id"]].copy()

    for key in BASE_MODEL_KEYS + SECOND_STAGE_KEYS:
        bundle = load_bundle(joblib, root, MODEL_REGISTRY[key])
        pred = bundle.predict(X)
        if len(pred) != len(current):
            raise RuntimeError(f"EPL inference stopped: {key} returned wrong row count.")
        for col in pred.columns:
            predicted[col] = pred[col].to_numpy()

    required = [
        "ml_home_prob", "ml_draw_prob", "ml_away_prob",
        "ml_over25_prob", "ml_under25_prob",
        "ml_over35_prob", "ml_under35_prob",
        "ml_btts_yes_prob", "ml_btts_no_prob",
        "ml_home_goals", "ml_away_goals",
    ]
    missing_outputs = [c for c in required if c not in predicted.columns]
    if missing_outputs:
        raise RuntimeError(f"EPL inference stopped: required model outputs absent: {missing_outputs}")

    if not np.allclose(
        predicted[["ml_home_prob", "ml_draw_prob", "ml_away_prob"]].sum(axis=1).to_numpy(float),
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError("EPL inference stopped: 1X2 probabilities do not sum to 1.")

    for a, b in (
        ("ml_over25_prob", "ml_under25_prob"),
        ("ml_over35_prob", "ml_under35_prob"),
        ("ml_btts_yes_prob", "ml_btts_no_prob"),
    ):
        if not np.allclose(predicted[[a, b]].sum(axis=1).to_numpy(float), 1.0, atol=1e-10):
            raise RuntimeError(f"EPL inference stopped: {a} + {b} does not equal 1.")

    return predicted


def enrich_merge_file(path: Path, predictions: pd.DataFrame) -> int:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return 0
    if "game_id" not in df.columns:
        raise RuntimeError(f"EPL inference stopped: game_id absent from merged file {path}")
    if df["game_id"].astype("string").duplicated().any():
        raise RuntimeError(f"EPL inference stopped: duplicate game_id in merged file {path}")

    pred = predictions.copy()
    pred["game_id"] = pred["game_id"].astype("string")
    df["game_id"] = df["game_id"].astype("string")

    ml_cols = [c for c in pred.columns if c != "game_id"]
    existing = [c for c in ml_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    out = df.merge(pred, how="left", on="game_id", validate="one_to_one")
    missing_predictions = out[ml_cols].isna().all(axis=1)
    if missing_predictions.any():
        bad = out.loc[missing_predictions, ["game_id", "home_team", "away_team"]]
        raise RuntimeError(
            f"EPL inference stopped: merged rows have no model prediction in {path}:\n"
            + bad.to_string(index=False)
        )

    temp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(temp, index=False)
    temp.replace(path)
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run EPL-only ML inference from normalized sportsbook data and attach outputs to EPL merge files."
    )
    ap.add_argument("--soccer-root", default="docs/win/soccer")
    ap.add_argument("--epl-root", default="docs/win/soccer/ml/epl")
    ap.add_argument("--date", default=os.environ.get("RUN_DATE", ""))
    args = ap.parse_args()

    soccer_root = Path(args.soccer_root)
    epl_root = Path(args.epl_root)
    run_date = resolve_date(args.date)

    wrapper_module = epl_root / "soccer_model_wrapper.py"
    if not wrapper_module.exists():
        raise FileNotFoundError(f"Required EPL wrapper module missing: {wrapper_module}")

    sys.path.insert(0, str(epl_root))
    import joblib
    import soccer_model_wrapper  # noqa: F401

    sportsbook_path = soccer_root / "00_intake" / "sportsbook" / "normalized" / f"{run_date}_epl.csv"
    if not sportsbook_path.exists():
        print(f"EPL ML inference: no EPL sportsbook file for {run_date}; nothing to do.")
        return

    current = pd.read_csv(sportsbook_path, low_memory=False)
    if current.empty:
        print(f"EPL ML inference: EPL sportsbook file is empty for {run_date}; nothing to do.")
        return

    predictions = predict_frame(joblib, epl_root, current)
    merge_dir = soccer_root / "01_merge"

    updated = []
    for suffix in MERGE_SUFFIXES:
        path = merge_dir / f"{run_date}_epl_{suffix}.csv"
        if not path.exists():
            continue
        rows = enrich_merge_file(path, predictions)
        updated.append((path.name, rows))

    if not updated:
        raise RuntimeError(
            f"EPL inference generated predictions for {run_date}, but no EPL merge files existed in {merge_dir}."
        )

    print(f"EPL ML inference complete for {run_date}: {len(predictions)} match prediction(s).")
    for name, rows in updated:
        print(f"  enriched {name}: {rows} row(s)")


if __name__ == "__main__":
    main()
