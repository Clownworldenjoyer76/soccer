#!/usr/bin/env python3
# docs/win/soccer/scripts/02_juice/apply_juice.py

import traceback
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# =========================
# PATHS
# =========================

THIS_FILE = Path(__file__).resolve()
SOCCER_ROOT = THIS_FILE.parents[2]          # .../docs/win/soccer

INPUT_DIR = SOCCER_ROOT / "01_merge"
OUTPUT_DIR = SOCCER_ROOT / "02_juice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_ROOT = SOCCER_ROOT / "config" / "juice"

ERROR_DIR = SOCCER_ROOT / "errors" / "02_juice"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "apply_juice_log.txt"


# =========================
# LEAGUE CONFIG MAPPING
# =========================

LEAGUE_TO_CONFIG = {
    "bundesliga": "bundesliga",
    "epl": "epl",
    "laliga": "la_liga",
    "ligue1": "ligue1",
    "mls": "mls",
    "seriea": "serie_a",
}

MARKETS = ["match_odds", "total_25", "total_35", "btts"]


# =========================
# LOGGING
# =========================

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")


# =========================
# HELPERS
# =========================

def safe_float(val):
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def safe_decimal(prob):
    if prob is None or pd.isna(prob) or prob <= 0:
        return None
    return 1.0 / prob


def normalize_probs(probs):
    vals = [p for p in probs if p is not None and not pd.isna(p)]
    total = sum(vals)
    if total <= 0:
        return [None for _ in probs]
    out = []
    for p in probs:
        if p is None or pd.isna(p):
            out.append(None)
        else:
            out.append(p / total)
    return out


def parse_stem(stem: str):
    for league in LEAGUE_TO_CONFIG:
        for market in MARKETS:
            suffix = f"_{league}_{market}"
            if stem.endswith(suffix):
                date_str = stem[: -len(suffix)]
                return date_str, league, market
    return None, None, None


# =========================
# CONFIG LOADERS
# =========================

@dataclass
class LeagueConfig:
    juice: pd.DataFrame
    engine: pd.DataFrame


def load_league_config(league: str) -> LeagueConfig:
    cfg_name = LEAGUE_TO_CONFIG[league]
    cfg_dir = CONFIG_ROOT / cfg_name

    juice_path = cfg_dir / "3way_juice.csv"
    engine_path = cfg_dir / "dc_soccer_pricing_engine.csv"

    if not juice_path.exists():
        raise FileNotFoundError(f"Missing juice config: {juice_path}")
    if not engine_path.exists():
        raise FileNotFoundError(f"Missing engine config: {engine_path}")

    juice = pd.read_csv(juice_path)
    engine = pd.read_csv(engine_path)

    required_juice_cols = {"side", "fair_prob", "extra_juice"}
    missing_juice = required_juice_cols - set(juice.columns)
    if missing_juice:
        raise ValueError(f"{juice_path} missing columns: {sorted(missing_juice)}")

    required_engine_cols = {
        "lambda_home", "lambda_away", "lambda_total", "rho",
        "home_win", "draw", "away_win",
        "over2_5", "under2_5", "btts_yes", "btts_no",
        "home_win_fair_odds", "draw_fair_odds", "away_win_fair_odds",
        "over2_5_fair_odds", "under2_5_fair_odds",
        "btts_yes_fair_odds", "btts_no_fair_odds",
        "under3_5", "over3_5", "over3_5_fair_odds", "under3_5_fair_odds",
    }
    missing_engine = required_engine_cols - set(engine.columns)
    if missing_engine:
        raise ValueError(f"{engine_path} missing columns: {sorted(missing_engine)}")

    juice["fair_prob"] = pd.to_numeric(juice["fair_prob"], errors="coerce")
    juice["extra_juice"] = pd.to_numeric(juice["extra_juice"], errors="coerce")
    juice["side"] = juice["side"].astype(str).str.strip().str.lower()

    for col in engine.columns:
        engine[col] = pd.to_numeric(engine[col], errors="coerce")

    return LeagueConfig(juice=juice, engine=engine)


def build_all_configs():
    configs = {}
    for league in LEAGUE_TO_CONFIG:
        configs[league] = load_league_config(league)
        log(
            f"CONFIG LOADED {league} | "
            f"juice_rows={len(configs[league].juice)} | "
            f"engine_rows={len(configs[league].engine)}"
        )
    return configs


# =========================
# LOOKUP FUNCTIONS
# =========================

def interp_extra_juice(juice_df: pd.DataFrame, side: str, fair_prob: float):
    if fair_prob is None or pd.isna(fair_prob):
        return None

    sub = juice_df[juice_df["side"] == side].copy()
    sub = sub.dropna(subset=["fair_prob", "extra_juice"]).sort_values("fair_prob")
    if sub.empty:
        return None

    xs = sub["fair_prob"].to_numpy(dtype=float)
    ys = sub["extra_juice"].to_numpy(dtype=float)

    # clamp outside range, interpolate inside range
    fair_prob = float(fair_prob)
    if fair_prob <= xs[0]:
        return float(ys[0])
    if fair_prob >= xs[-1]:
        return float(ys[-1])

    return float(np.interp(fair_prob, xs, ys))


def nearest_engine_row(engine_df: pd.DataFrame, home_xg: float, away_xg: float):
    if home_xg is None or away_xg is None:
        return None, None

    work = engine_df.dropna(subset=["lambda_home", "lambda_away"]).copy()
    if work.empty:
        return None, None

    dh = work["lambda_home"] - float(home_xg)
    da = work["lambda_away"] - float(away_xg)

    # 2D nearest-neighbor on the lambda grid
    work["_distance"] = (dh ** 2 + da ** 2)
    idx = work["_distance"].idxmin()
    row = work.loc[idx].to_dict()
    dist = float(work.loc[idx, "_distance"])
    return row, dist


# =========================
# MARKET PROCESSORS
# =========================

def process_match_odds(df: pd.DataFrame, cfg: LeagueConfig) -> pd.DataFrame:
    out_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()

        home_prob = safe_float(r.get("home_prob"))
        draw_prob = safe_float(r.get("draw_prob"))
        away_prob = safe_float(r.get("away_prob"))
        home_xg = safe_float(r.get("home_xg"))
        away_xg = safe_float(r.get("away_xg"))

        engine_row, engine_distance = nearest_engine_row(cfg.engine, home_xg, away_xg)

        home_extra = interp_extra_juice(cfg.juice, "home", home_prob)
        draw_extra = interp_extra_juice(cfg.juice, "draw", draw_prob)
        away_extra = interp_extra_juice(cfg.juice, "away", away_prob)

        raw_home = None if home_prob is None or home_extra is None else home_prob + home_extra
        raw_draw = None if draw_prob is None or draw_extra is None else draw_prob + draw_extra
        raw_away = None if away_prob is None or away_extra is None else away_prob + away_extra

        juiced_home_prob, juiced_draw_prob, juiced_away_prob = normalize_probs(
            [raw_home, raw_draw, raw_away]
        )

        r["home_extra_juice"] = home_extra
        r["draw_extra_juice"] = draw_extra
        r["away_extra_juice"] = away_extra

        r["juiced_home_prob"] = juiced_home_prob
        r["juiced_draw_prob"] = juiced_draw_prob
        r["juiced_away_prob"] = juiced_away_prob

        r["juiced_home_decimal"] = safe_decimal(juiced_home_prob)
        r["juiced_draw_decimal"] = safe_decimal(juiced_draw_prob)
        r["juiced_away_decimal"] = safe_decimal(juiced_away_prob)

        if engine_row:
            r["engine_lambda_home"] = engine_row["lambda_home"]
            r["engine_lambda_away"] = engine_row["lambda_away"]
            r["engine_match_distance"] = engine_distance
            r["engine_home_prob"] = engine_row["home_win"]
            r["engine_draw_prob"] = engine_row["draw"]
            r["engine_away_prob"] = engine_row["away_win"]
            r["engine_home_fair_decimal"] = engine_row["home_win_fair_odds"]
            r["engine_draw_fair_decimal"] = engine_row["draw_fair_odds"]
            r["engine_away_fair_decimal"] = engine_row["away_win_fair_odds"]
        else:
            r["engine_lambda_home"] = None
            r["engine_lambda_away"] = None
            r["engine_match_distance"] = None
            r["engine_home_prob"] = None
            r["engine_draw_prob"] = None
            r["engine_away_prob"] = None
            r["engine_home_fair_decimal"] = None
            r["engine_draw_fair_decimal"] = None
            r["engine_away_fair_decimal"] = None

        out_rows.append(r)

    return pd.DataFrame(out_rows)


def process_total_25(df: pd.DataFrame, cfg: LeagueConfig) -> pd.DataFrame:
    out_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()
        home_xg = safe_float(r.get("home_xg"))
        away_xg = safe_float(r.get("away_xg"))

        engine_row, engine_distance = nearest_engine_row(cfg.engine, home_xg, away_xg)

        if engine_row:
            r["engine_lambda_home"] = engine_row["lambda_home"]
            r["engine_lambda_away"] = engine_row["lambda_away"]
            r["engine_match_distance"] = engine_distance
            r["fair_over_decimal"] = engine_row["over2_5_fair_odds"]
            r["fair_under_decimal"] = engine_row["under2_5_fair_odds"]
            r["engine_over_prob"] = engine_row["over2_5"]
            r["engine_under_prob"] = engine_row["under2_5"]
        else:
            r["engine_lambda_home"] = None
            r["engine_lambda_away"] = None
            r["engine_match_distance"] = None
            r["fair_over_decimal"] = None
            r["fair_under_decimal"] = None
            r["engine_over_prob"] = None
            r["engine_under_prob"] = None

        out_rows.append(r)

    return pd.DataFrame(out_rows)


def process_total_35(df: pd.DataFrame, cfg: LeagueConfig) -> pd.DataFrame:
    out_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()
        home_xg = safe_float(r.get("home_xg"))
        away_xg = safe_float(r.get("away_xg"))

        engine_row, engine_distance = nearest_engine_row(cfg.engine, home_xg, away_xg)

        if engine_row:
            r["engine_lambda_home"] = engine_row["lambda_home"]
            r["engine_lambda_away"] = engine_row["lambda_away"]
            r["engine_match_distance"] = engine_distance
            r["fair_over_decimal"] = engine_row["over3_5_fair_odds"]
            r["fair_under_decimal"] = engine_row["under3_5_fair_odds"]
            r["engine_over_prob"] = engine_row["over3_5"]
            r["engine_under_prob"] = engine_row["under3_5"]
        else:
            r["engine_lambda_home"] = None
            r["engine_lambda_away"] = None
            r["engine_match_distance"] = None
            r["fair_over_decimal"] = None
            r["fair_under_decimal"] = None
            r["engine_over_prob"] = None
            r["engine_under_prob"] = None

        out_rows.append(r)

    return pd.DataFrame(out_rows)


def process_btts(df: pd.DataFrame, cfg: LeagueConfig) -> pd.DataFrame:
    out_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()
        home_xg = safe_float(r.get("home_xg"))
        away_xg = safe_float(r.get("away_xg"))

        engine_row, engine_distance = nearest_engine_row(cfg.engine, home_xg, away_xg)

        if engine_row:
            r["engine_lambda_home"] = engine_row["lambda_home"]
            r["engine_lambda_away"] = engine_row["lambda_away"]
            r["engine_match_distance"] = engine_distance
            r["fair_btts_yes_decimal"] = engine_row["btts_yes_fair_odds"]
            r["fair_btts_no_decimal"] = engine_row["btts_no_fair_odds"]
            r["engine_btts_yes_prob"] = engine_row["btts_yes"]
            r["engine_btts_no_prob"] = engine_row["btts_no"]
        else:
            r["engine_lambda_home"] = None
            r["engine_lambda_away"] = None
            r["engine_match_distance"] = None
            r["fair_btts_yes_decimal"] = None
            r["fair_btts_no_decimal"] = None
            r["engine_btts_yes_prob"] = None
            r["engine_btts_no_prob"] = None

        out_rows.append(r)

    return pd.DataFrame(out_rows)


# =========================
# MAIN FILE PROCESSOR
# =========================

def process_file(file_path: Path, configs: dict, summary: dict):
    try:
        date_str, league, market = parse_stem(file_path.stem)
        if not league or not market:
            log(f"SKIP unrecognized file: {file_path.name}")
            summary["skipped"] += 1
            return

        if league not in configs:
            log(f"SKIP no config for league={league}: {file_path.name}")
            summary["skipped"] += 1
            return

        df = pd.read_csv(file_path)
        if df.empty:
            log(f"EMPTY: {file_path.name}")
            summary["empty"] += 1
            return

        cfg = configs[league]

        if market == "match_odds":
            out_df = process_match_odds(df, cfg)
        elif market == "total_25":
            out_df = process_total_25(df, cfg)
        elif market == "total_35":
            out_df = process_total_35(df, cfg)
        elif market == "btts":
            out_df = process_btts(df, cfg)
        else:
            log(f"SKIP unsupported market={market}: {file_path.name}")
            summary["skipped"] += 1
            return

        out_path = OUTPUT_DIR / file_path.name
        out_df.to_csv(out_path, index=False)

        log(f"WROTE {out_path} ({len(out_df)} rows)")
        summary["files_written"] += 1
        summary["rows_written"] += len(out_df)

    except Exception as e:
        log(f"ERROR processing {file_path}: {e}\n{traceback.format_exc()}")
        summary["errors"] += 1


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_juice RUN {datetime.now(timezone.utc).isoformat()} ===\n")

    summary = {
        "files_written": 0,
        "rows_written": 0,
        "empty": 0,
        "skipped": 0,
        "errors": 0,
    }

    configs = build_all_configs()

    input_files = sorted(
        f for f in INPUT_DIR.glob("*.csv")
        if f.name.endswith(".csv")
    )

    for file_path in input_files:
        process_file(file_path, configs, summary)

    log(
        f"SUMMARY: files_written={summary['files_written']} | "
        f"rows_written={summary['rows_written']} | "
        f"empty={summary['empty']} | "
        f"skipped={summary['skipped']} | "
        f"errors={summary['errors']}"
    )
    log("COMPLETE")
    print("apply_juice complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"FATAL:\n{e}\n{traceback.format_exc()}")
        raise
