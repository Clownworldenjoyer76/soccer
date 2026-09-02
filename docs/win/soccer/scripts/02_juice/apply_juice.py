#!/usr/bin/env python3
# docs/win/soccer/scripts/02_juice/apply_juice.py
#
# PRODUCTION PRICING AUTHORITY
# ----------------------------
# Stage 1 merges normalized prediction/xG inputs with sportsbook odds and EPL
# ML inference adds ml_* fields to EPL merge rows.
#
# This stage is the single production pricing authority:
#   * EPL: selected trained ml_* probabilities are authoritative.
#   * Other leagues: existing Dixon-Coles probabilities remain authoritative.
#
# The authoritative values are written into the existing engine_* probability
# contract so Stage 3/4 do not need a competing pricing path. Explicit
# engine_*_prob_source / pricing_source columns identify the underlying source.
# Dixon-Coles is still calculated and validated for every league; for EPL it is
# retained as diagnostic dc_* output while the trained ML probabilities drive
# fair odds, EV, Kelly, and selection downstream.
#
# Legacy adjusted/juiced 1X2 fields remain diagnostic only.

import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
SOCCER_ROOT = THIS_FILE.parents[2]

INPUT_DIR = SOCCER_ROOT / "01_merge"
OUTPUT_DIR = SOCCER_ROOT / "02_juice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_ROOT = SOCCER_ROOT / "config" / "juice"

ERROR_DIR = SOCCER_ROOT / "errors" / "02_juice"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "apply_juice_log.txt"

LEAGUE_TO_CONFIG = {
    "bundesliga": "bundesliga",
    "epl": "epl",
    "laliga": "la_liga",
    "ligue1": "ligue1",
    "mls": "mls",
    "seriea": "serie_a",
}

MARKETS = ["match_odds", "total_25", "total_35", "btts"]

POISSON_TAIL_TOLERANCE = 1e-14
POISSON_MAX_GOALS = 100

XG_COLS = ["home_xg", "away_xg", "expected_total_goals"]
XG_TOTAL_TOLERANCE = 0.01

ADJUSTED_1X2_SUM_TOLERANCE = 1e-9
ENGINE_PROB_SUM_TOLERANCE = 1e-9
FAIR_DECIMAL_REL_TOLERANCE = 1e-9
FAIR_DECIMAL_ABS_TOLERANCE = 1e-12
HOME_AWAY_SYMMETRY_TOLERANCE = 1e-9

EPL_ML_MARKET_COLUMNS = {
    "match_odds": (
        ("home_win", "ml_home_prob"),
        ("draw", "ml_draw_prob"),
        ("away_win", "ml_away_prob"),
    ),
    "total_25": (
        ("over2_5", "ml_over25_prob"),
        ("under2_5", "ml_under25_prob"),
    ),
    "total_35": (
        ("over3_5", "ml_over35_prob"),
        ("under3_5", "ml_under35_prob"),
    ),
    "btts": (
        ("btts_yes", "ml_btts_yes_prob"),
        ("btts_no", "ml_btts_no_prob"),
    ),
}


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")


def safe_float(val):
    try:
        if pd.isna(val):
            return None
        number = float(val)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def safe_decimal(prob):
    probability = safe_float(prob)
    if probability is None or probability <= 0.0:
        return None
    decimal = 1.0 / probability
    if not math.isfinite(decimal) or decimal <= 0.0:
        return None
    return decimal


def valid_probability(prob) -> bool:
    value = safe_float(prob)
    return value is not None and 0.0 <= value <= 1.0


def validate_threeway_probs(probs, tolerance=ADJUSTED_1X2_SUM_TOLERANCE):
    if len(probs) != 3:
        return False
    parsed = []
    for prob in probs:
        value = safe_float(prob)
        if value is None or value < 0.0 or value > 1.0:
            return False
        parsed.append(value)
    total = sum(parsed)
    return math.isfinite(total) and abs(total - 1.0) <= tolerance


def normalize_probs(probs):
    if len(probs) != 3:
        return [None for _ in probs]
    parsed = []
    for prob in probs:
        value = safe_float(prob)
        if value is None or value < 0.0:
            return [None, None, None]
        parsed.append(value)
    total = sum(parsed)
    if not math.isfinite(total) or total <= 0.0:
        return [None, None, None]
    normalized = [value / total for value in parsed]
    return normalized if validate_threeway_probs(normalized) else [None, None, None]


def parse_stem(stem: str):
    for league in LEAGUE_TO_CONFIG:
        for market in MARKETS:
            suffix = f"_{league}_{market}"
            if stem.endswith(suffix):
                return stem[:-len(suffix)], league, market
    return None, None, None


def validate_xg_inputs(row):
    parsed = {}
    for col in XG_COLS:
        value = safe_float(row.get(col))
        if value is None:
            return False, {}, None, None, f"missing/non-numeric {col}"
        if value < 0.0:
            return False, {}, None, None, f"negative {col}={value}"
        parsed[col] = value

    component_total = round(parsed["home_xg"] + parsed["away_xg"], 6)
    total_difference = abs(
        round(parsed["expected_total_goals"] - component_total, 6)
    )
    if total_difference > XG_TOTAL_TOLERANCE:
        return (
            False,
            parsed,
            component_total,
            total_difference,
            f"inconsistent expected_total_goals={parsed['expected_total_goals']:.6f} "
            f"home_plus_away={component_total:.6f} "
            f"difference={total_difference:.6f} "
            f"tolerance={XG_TOTAL_TOLERANCE:.6f}",
        )
    return True, parsed, component_total, total_difference, ""


def validate_xg_dataframe(df: pd.DataFrame, file_path: Path, summary: dict) -> pd.DataFrame:
    clean_rows = []
    invalid_count = 0
    for source_index, row in df.iterrows():
        valid, parsed, component_total, total_difference, reason = validate_xg_inputs(row)
        if not valid:
            invalid_count += 1
            summary["rows_rejected_invalid_xg"] += 1
            log(
                "REJECT INVALID XG BEFORE PRICING | "
                f"file={file_path.name} | line={source_index + 2} | "
                f"game_id={row.get('game_id', '')} | "
                f"home_team={row.get('home_team', '')} | "
                f"away_team={row.get('away_team', '')} | reason={reason} | "
                f"home_plus_away={component_total!r} | difference={total_difference!r}"
            )
            continue
        clean_row = row.copy()
        for col in XG_COLS:
            clean_row[col] = parsed[col]
        clean_rows.append(clean_row)

    if invalid_count:
        raise ValueError(
            f"{file_path.name} contains {invalid_count} invalid xG row(s); "
            "pricing aborted for this file"
        )
    return pd.DataFrame(clean_rows).reset_index(drop=True)


@dataclass
class LeagueConfig:
    juice: pd.DataFrame
    rho: float


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
    engine = pd.read_csv(engine_path, usecols=["rho"])

    required_juice_cols = {"side", "fair_prob", "extra_juice"}
    missing_juice = required_juice_cols - set(juice.columns)
    if missing_juice:
        raise ValueError(f"{juice_path} missing columns: {sorted(missing_juice)}")

    juice["fair_prob"] = pd.to_numeric(juice["fair_prob"], errors="coerce")
    juice["extra_juice"] = pd.to_numeric(juice["extra_juice"], errors="coerce")
    juice["side"] = juice["side"].astype(str).str.strip().str.lower()

    rho_values = pd.to_numeric(engine["rho"], errors="coerce").dropna().unique()
    if len(rho_values) != 1:
        raise ValueError(
            f"{engine_path} must contain exactly one non-null rho value; "
            f"found {len(rho_values)}"
        )
    rho = safe_float(rho_values[0])
    if rho is None:
        raise ValueError(f"{engine_path} contains a non-finite rho value")
    return LeagueConfig(juice=juice, rho=rho)


def build_all_configs():
    configs = {}
    for league in LEAGUE_TO_CONFIG:
        configs[league] = load_league_config(league)
        log(
            f"CONFIG LOADED {league} | juice_rows={len(configs[league].juice)} | "
            f"rho={configs[league].rho}"
        )
    return configs


def interp_extra_juice(juice_df: pd.DataFrame, side: str, fair_prob: float):
    if fair_prob is None or pd.isna(fair_prob):
        return None
    sub = (
        juice_df[juice_df["side"] == side]
        .copy()
        .dropna(subset=["fair_prob", "extra_juice"])
        .sort_values("fair_prob")
    )
    if sub.empty:
        return None
    xs = sub["fair_prob"].to_numpy(dtype=float)
    ys = sub["extra_juice"].to_numpy(dtype=float)
    fair_prob = float(fair_prob)
    if fair_prob <= xs[0]:
        return float(ys[0])
    if fair_prob >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(fair_prob, xs, ys))


def poisson_score_probs(lam: float):
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0:
        raise ValueError(f"Invalid Poisson lambda: {lam}")
    p = math.exp(-lam)
    probs = [p]
    cumulative = p
    for goals in range(1, POISSON_MAX_GOALS + 1):
        if 1.0 - cumulative <= POISSON_TAIL_TOLERANCE:
            break
        p *= lam / goals
        probs.append(p)
        cumulative += p
    if 1.0 - cumulative > 1e-10:
        raise ValueError(
            f"Poisson tail did not converge for lambda={lam}; "
            f"remaining_mass={1.0 - cumulative}"
        )
    return probs


def dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    if home_goals == 0 and away_goals == 0:
        return 1.0 - (lambda_home * lambda_away * rho)
    if home_goals == 0 and away_goals == 1:
        return 1.0 + (lambda_home * rho)
    if home_goals == 1 and away_goals == 0:
        return 1.0 + (lambda_away * rho)
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


@lru_cache(maxsize=10000)
def price_from_xg(home_xg: float, away_xg: float, rho: float):
    lambda_home = float(home_xg)
    lambda_away = float(away_xg)
    rho = float(rho)
    if (
        not math.isfinite(lambda_home)
        or not math.isfinite(lambda_away)
        or not math.isfinite(rho)
        or lambda_home < 0
        or lambda_away < 0
    ):
        return None

    home_scores = poisson_score_probs(lambda_home)
    away_scores = poisson_score_probs(lambda_away)

    result = {
        "home_win": 0.0,
        "draw": 0.0,
        "away_win": 0.0,
        "over2_5": 0.0,
        "under2_5": 0.0,
        "over3_5": 0.0,
        "under3_5": 0.0,
        "btts_yes": 0.0,
        "btts_no": 0.0,
    }
    total_mass = 0.0

    for home_goals, p_home in enumerate(home_scores):
        for away_goals, p_away in enumerate(away_scores):
            tau = dixon_coles_tau(
                home_goals, away_goals, lambda_home, lambda_away, rho
            )
            if not math.isfinite(tau) or tau < 0:
                raise ValueError(
                    "Dixon-Coles produced invalid low-score multiplier: "
                    f"home_xg={lambda_home}, away_xg={lambda_away}, rho={rho}, "
                    f"score={home_goals}-{away_goals}, tau={tau}"
                )
            p = p_home * p_away * tau
            if not math.isfinite(p) or p < 0.0:
                raise ValueError("Dixon-Coles produced invalid probability mass")
            total_mass += p

            if home_goals > away_goals:
                result["home_win"] += p
            elif home_goals == away_goals:
                result["draw"] += p
            else:
                result["away_win"] += p

            total_goals = home_goals + away_goals
            result["over2_5" if total_goals > 2.5 else "under2_5"] += p
            result["over3_5" if total_goals > 3.5 else "under3_5"] += p
            result["btts_yes" if home_goals > 0 and away_goals > 0 else "btts_no"] += p

    if not math.isfinite(total_mass) or total_mass <= 0:
        raise ValueError(
            f"Invalid Dixon-Coles total probability mass for home_xg={lambda_home}, "
            f"away_xg={lambda_away}, rho={rho}, total_mass={total_mass}"
        )
    for key in result:
        result[key] /= total_mass
    result["lambda_home"] = lambda_home
    result["lambda_away"] = lambda_away
    return result


def get_pricing(home_xg, away_xg, expected_total_goals, rho):
    valid, parsed, _, _, _ = validate_xg_inputs(
        {
            "home_xg": home_xg,
            "away_xg": away_xg,
            "expected_total_goals": expected_total_goals,
        }
    )
    if not valid:
        return None
    return price_from_xg(parsed["home_xg"], parsed["away_xg"], float(rho))


def validate_dc_pricing(pricing, home_xg, away_xg, rho):
    if not isinstance(pricing, dict):
        return False, "pricing_result_missing", {"pricing": pricing}

    groups = {
        "engine_1x2_sum": ["home_win", "draw", "away_win"],
        "engine_total_25_sum": ["over2_5", "under2_5"],
        "engine_total_35_sum": ["over3_5", "under3_5"],
        "engine_btts_sum": ["btts_yes", "btts_no"],
    }
    for group_name, keys in groups.items():
        values = [safe_float(pricing.get(k)) for k in keys]
        if any(v is None or v < 0.0 or v > 1.0 for v in values):
            return False, "probability_range", {k: pricing.get(k) for k in keys}
        total = sum(values)
        if abs(total - 1.0) > ENGINE_PROB_SUM_TOLERANCE:
            return False, group_name, {**{k: pricing.get(k) for k in keys}, "sum": total}

    hx = safe_float(home_xg)
    ax = safe_float(away_xg)
    rr = safe_float(rho)
    if hx is None or ax is None or rr is None:
        return False, "home_away_symmetry_inputs", {"home_xg": home_xg, "away_xg": away_xg, "rho": rho}
    swapped = price_from_xg(ax, hx, rr)
    symmetry_pairs = [
        ("home_win", "away_win"),
        ("away_win", "home_win"),
        ("draw", "draw"),
        ("over2_5", "over2_5"),
        ("under2_5", "under2_5"),
        ("over3_5", "over3_5"),
        ("under3_5", "under3_5"),
        ("btts_yes", "btts_yes"),
        ("btts_no", "btts_no"),
        ("lambda_home", "lambda_away"),
        ("lambda_away", "lambda_home"),
    ]
    for original_key, swapped_key in symmetry_pairs:
        original_value = safe_float(pricing.get(original_key))
        swapped_value = safe_float(swapped.get(swapped_key)) if swapped else None
        if (
            original_value is None
            or swapped_value is None
            or not math.isclose(
                original_value,
                swapped_value,
                rel_tol=HOME_AWAY_SYMMETRY_TOLERANCE,
                abs_tol=HOME_AWAY_SYMMETRY_TOLERANCE,
            )
        ):
            return False, "home_away_symmetry", {
                "field": original_key,
                "original_value": original_value,
                "swapped_field": swapped_key,
                "swapped_value": swapped_value,
            }
    return True, "", {}


def validate_row_pricing(pricing, row, cfg, file_path, source_index, market, summary):
    valid, invariant, values = validate_dc_pricing(
        pricing, row.get("home_xg"), row.get("away_xg"), cfg.rho
    )
    if valid:
        return True
    summary["rows_rejected_pricing_invariant"] += 1
    log(
        "REJECT PRICING INVARIANT | "
        f"file={file_path.name} | line={source_index + 2} | "
        f"game_id={row.get('game_id', '')} | market={market} | "
        f"invariant={invariant} | values={values!r}"
    )
    return False


def _validate_authoritative_probs(values, market, row, file_path, source_index, summary):
    if any(value is None or value <= 0.0 or value > 1.0 for value in values):
        summary["rows_rejected_pricing_invariant"] += 1
        log(
            "REJECT AUTHORITATIVE PROBABILITY | "
            f"file={file_path.name} | line={source_index + 2} | "
            f"game_id={row.get('game_id', '')} | market={market} | values={values!r}"
        )
        return False
    total = sum(values)
    if abs(total - 1.0) > ENGINE_PROB_SUM_TOLERANCE:
        summary["rows_rejected_pricing_invariant"] += 1
        log(
            "REJECT AUTHORITATIVE PROBABILITY SUM | "
            f"file={file_path.name} | line={source_index + 2} | "
            f"game_id={row.get('game_id', '')} | market={market} | "
            f"sum={total} | values={values!r}"
        )
        return False
    return True


def authoritative_market_probs(dc_pricing, row, market, file_path, source_index, summary):
    league = str(row.get("league", "")).strip().casefold()
    if league == "epl":
        mapping = EPL_ML_MARKET_COLUMNS[market]
        values = [safe_float(row.get(column)) for _, column in mapping]
        if not _validate_authoritative_probs(
            values, market, row, file_path, source_index, summary
        ):
            return None, None, None
        probs = {key: value for (key, _), value in zip(mapping, values)}
        sources = {key: column for key, column in mapping}
        return probs, sources, "epl_ml"

    mapping = {
        "match_odds": ("home_win", "draw", "away_win"),
        "total_25": ("over2_5", "under2_5"),
        "total_35": ("over3_5", "under3_5"),
        "btts": ("btts_yes", "btts_no"),
    }[market]
    probs = {key: dc_pricing[key] for key in mapping}
    sources = {key: f"dixon_coles_{key}" for key in mapping}
    return probs, sources, "dixon_coles"


def validate_fair_pair(probability, decimal_price):
    p = safe_float(probability)
    d = safe_float(decimal_price)
    if p is None or p <= 0.0 or p > 1.0 or d is None or d <= 1.0:
        return False
    return math.isclose(
        d,
        1.0 / p,
        rel_tol=FAIR_DECIMAL_REL_TOLERANCE,
        abs_tol=FAIR_DECIMAL_ABS_TOLERANCE,
    )


def process_match_odds(df, cfg, file_path, summary):
    out_rows = []
    for source_index, row in df.iterrows():
        r = row.to_dict()
        r.pop("engine_match_distance", None)
        home_prob = safe_float(r.get("home_prob"))
        draw_prob = safe_float(r.get("draw_prob"))
        away_prob = safe_float(r.get("away_prob"))

        dc = get_pricing(r.get("home_xg"), r.get("away_xg"), r.get("expected_total_goals"), cfg.rho)
        if not validate_row_pricing(dc, r, cfg, file_path, source_index, "match_odds", summary):
            continue

        auth, sources, pricing_source = authoritative_market_probs(
            dc, r, "match_odds", file_path, source_index, summary
        )
        if auth is None:
            continue

        home_extra = interp_extra_juice(cfg.juice, "home", home_prob)
        draw_extra = interp_extra_juice(cfg.juice, "draw", draw_prob)
        away_extra = interp_extra_juice(cfg.juice, "away", away_prob)
        raw_probs = [
            None if home_prob is None or home_extra is None else home_prob + home_extra,
            None if draw_prob is None or draw_extra is None else draw_prob + draw_extra,
            None if away_prob is None or away_extra is None else away_prob + away_extra,
        ]
        juiced_probs = normalize_probs(raw_probs)
        if not validate_threeway_probs(juiced_probs):
            summary["rows_rejected_invalid_adjusted_1x2"] += 1
            log(
                "REJECT INVALID ADJUSTED 1X2 | "
                f"file={file_path.name} | line={source_index + 2} | "
                f"game_id={r.get('game_id', '')} | raw={raw_probs!r} | "
                f"normalized={juiced_probs!r}"
            )
            continue

        engine_home_decimal = safe_decimal(auth["home_win"])
        engine_draw_decimal = safe_decimal(auth["draw"])
        engine_away_decimal = safe_decimal(auth["away_win"])
        if not all(
            (
                validate_fair_pair(auth["home_win"], engine_home_decimal),
                validate_fair_pair(auth["draw"], engine_draw_decimal),
                validate_fair_pair(auth["away_win"], engine_away_decimal),
            )
        ):
            summary["rows_rejected_pricing_invariant"] += 1
            log(f"REJECT FAIR DECIMAL | file={file_path.name} | line={source_index + 2}")
            continue

        r.update(
            {
                "home_extra_juice": home_extra,
                "draw_extra_juice": draw_extra,
                "away_extra_juice": away_extra,
                "juiced_home_prob": juiced_probs[0],
                "juiced_draw_prob": juiced_probs[1],
                "juiced_away_prob": juiced_probs[2],
                "juiced_home_decimal": safe_decimal(juiced_probs[0]),
                "juiced_draw_decimal": safe_decimal(juiced_probs[1]),
                "juiced_away_decimal": safe_decimal(juiced_probs[2]),
                "engine_lambda_home": dc["lambda_home"],
                "engine_lambda_away": dc["lambda_away"],
                "dc_home_prob": dc["home_win"],
                "dc_draw_prob": dc["draw"],
                "dc_away_prob": dc["away_win"],
                "engine_home_prob": auth["home_win"],
                "engine_draw_prob": auth["draw"],
                "engine_away_prob": auth["away_win"],
                "engine_home_prob_source": sources["home_win"],
                "engine_draw_prob_source": sources["draw"],
                "engine_away_prob_source": sources["away_win"],
                "engine_home_fair_decimal": engine_home_decimal,
                "engine_draw_fair_decimal": engine_draw_decimal,
                "engine_away_fair_decimal": engine_away_decimal,
                "pricing_source": pricing_source,
            }
        )
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def _process_total(df, cfg, file_path, summary, market, over_key, under_key):
    out_rows = []
    for source_index, row in df.iterrows():
        r = row.to_dict()
        r.pop("engine_match_distance", None)
        dc = get_pricing(r.get("home_xg"), r.get("away_xg"), r.get("expected_total_goals"), cfg.rho)
        if not validate_row_pricing(dc, r, cfg, file_path, source_index, market, summary):
            continue
        auth, sources, pricing_source = authoritative_market_probs(
            dc, r, market, file_path, source_index, summary
        )
        if auth is None:
            continue
        fair_over = safe_decimal(auth[over_key])
        fair_under = safe_decimal(auth[under_key])
        if not validate_fair_pair(auth[over_key], fair_over) or not validate_fair_pair(auth[under_key], fair_under):
            summary["rows_rejected_pricing_invariant"] += 1
            log(f"REJECT FAIR DECIMAL | file={file_path.name} | line={source_index + 2}")
            continue
        r.update(
            {
                "engine_lambda_home": dc["lambda_home"],
                "engine_lambda_away": dc["lambda_away"],
                "dc_over_prob": dc[over_key],
                "dc_under_prob": dc[under_key],
                "fair_over_decimal": fair_over,
                "fair_under_decimal": fair_under,
                "engine_over_prob": auth[over_key],
                "engine_under_prob": auth[under_key],
                "engine_over_prob_source": sources[over_key],
                "engine_under_prob_source": sources[under_key],
                "pricing_source": pricing_source,
            }
        )
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def process_total_25(df, cfg, file_path, summary):
    return _process_total(df, cfg, file_path, summary, "total_25", "over2_5", "under2_5")


def process_total_35(df, cfg, file_path, summary):
    return _process_total(df, cfg, file_path, summary, "total_35", "over3_5", "under3_5")


def process_btts(df, cfg, file_path, summary):
    out_rows = []
    for source_index, row in df.iterrows():
        r = row.to_dict()
        r.pop("engine_match_distance", None)
        dc = get_pricing(r.get("home_xg"), r.get("away_xg"), r.get("expected_total_goals"), cfg.rho)
        if not validate_row_pricing(dc, r, cfg, file_path, source_index, "btts", summary):
            continue
        auth, sources, pricing_source = authoritative_market_probs(
            dc, r, "btts", file_path, source_index, summary
        )
        if auth is None:
            continue
        fair_yes = safe_decimal(auth["btts_yes"])
        fair_no = safe_decimal(auth["btts_no"])
        if not validate_fair_pair(auth["btts_yes"], fair_yes) or not validate_fair_pair(auth["btts_no"], fair_no):
            summary["rows_rejected_pricing_invariant"] += 1
            log(f"REJECT FAIR DECIMAL | file={file_path.name} | line={source_index + 2}")
            continue
        r.update(
            {
                "engine_lambda_home": dc["lambda_home"],
                "engine_lambda_away": dc["lambda_away"],
                "dc_btts_yes_prob": dc["btts_yes"],
                "dc_btts_no_prob": dc["btts_no"],
                "fair_btts_yes_decimal": fair_yes,
                "fair_btts_no_decimal": fair_no,
                "engine_btts_yes_prob": auth["btts_yes"],
                "engine_btts_no_prob": auth["btts_no"],
                "engine_btts_yes_prob_source": sources["btts_yes"],
                "engine_btts_no_prob_source": sources["btts_no"],
                "pricing_source": pricing_source,
            }
        )
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def process_file(file_path: Path, configs: dict, summary: dict):
    out_path = OUTPUT_DIR / file_path.name
    try:
        _, league, market = parse_stem(file_path.stem)
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

        if out_path.exists():
            out_path.unlink()

        df = validate_xg_dataframe(df, file_path, summary)
        cfg = configs[league]

        if market == "match_odds":
            out_df = process_match_odds(df, cfg, file_path, summary)
        elif market == "total_25":
            out_df = process_total_25(df, cfg, file_path, summary)
        elif market == "total_35":
            out_df = process_total_35(df, cfg, file_path, summary)
        elif market == "btts":
            out_df = process_btts(df, cfg, file_path, summary)
        else:
            log(f"SKIP unsupported market={market}: {file_path.name}")
            summary["skipped"] += 1
            return

        if len(out_df) != len(df):
            raise ValueError(
                "Pricing fail-safe rejected rows after pre-validation for "
                f"{file_path.name}: validated={len(df)} priced={len(out_df)}"
            )

        out_df.to_csv(out_path, index=False)
        log(f"WROTE {out_path} ({len(out_df)} rows)")
        summary["files_written"] += 1
        summary["rows_written"] += len(out_df)
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        log(f"ERROR processing {file_path}: {e}\n{traceback.format_exc()}")
        summary["errors"] += 1


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_juice RUN {datetime.now(timezone.utc).isoformat()} ===\n")

    summary = {
        "files_written": 0,
        "rows_written": 0,
        "rows_rejected_invalid_xg": 0,
        "rows_rejected_invalid_adjusted_1x2": 0,
        "rows_rejected_pricing_invariant": 0,
        "empty": 0,
        "skipped": 0,
        "errors": 0,
    }

    log(
        "PRODUCTION PRICING | epl=epl_ml | other_leagues=dixon_coles | "
        "engine_* remains downstream authority contract"
    )
    log(
        "xG/DC validation retained | "
        f"xg_total_tolerance={XG_TOTAL_TOLERANCE:.6f} | "
        f"probability_sum_tolerance={ENGINE_PROB_SUM_TOLERANCE:.12f} | "
        f"home_away_symmetry_tolerance={HOME_AWAY_SYMMETRY_TOLERANCE:.12f}"
    )

    configs = build_all_configs()
    input_files = sorted(INPUT_DIR.glob("*.csv"))
    for file_path in input_files:
        process_file(file_path, configs, summary)

    log(
        "SUMMARY: "
        f"files_written={summary['files_written']} | "
        f"rows_written={summary['rows_written']} | "
        f"rows_rejected_invalid_xg={summary['rows_rejected_invalid_xg']} | "
        f"rows_rejected_invalid_adjusted_1x2={summary['rows_rejected_invalid_adjusted_1x2']} | "
        f"rows_rejected_pricing_invariant={summary['rows_rejected_pricing_invariant']} | "
        f"empty={summary['empty']} | skipped={summary['skipped']} | errors={summary['errors']}"
    )

    if (
        summary["errors"]
        or summary["rows_rejected_invalid_xg"]
        or summary["rows_rejected_invalid_adjusted_1x2"]
        or summary["rows_rejected_pricing_invariant"]
    ):
        log("FAILED")
        raise RuntimeError(
            "apply_juice aborted because one or more input rows/files failed validation or pricing"
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
