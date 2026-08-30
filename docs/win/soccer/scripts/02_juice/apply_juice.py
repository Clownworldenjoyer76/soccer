#!/usr/bin/env python3
# docs/win/soccer/scripts/02_juice/apply_juice.py

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

XG_COLS = [
    "home_xg",
    "away_xg",
    "expected_total_goals",
]

XG_TOTAL_TOLERANCE = 0.01

ADJUSTED_1X2_SUM_TOLERANCE = 1e-9


def log(msg: str) -> None:
    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{msg}\n"
        )


def safe_float(val):
    try:
        if pd.isna(val):
            return None

        number = float(val)

        if not math.isfinite(number):
            return None

        return number

    except Exception:
        return None


def safe_decimal(prob):
    if (
        prob is None
        or pd.isna(prob)
        or prob <= 0
    ):
        return None

    return 1.0 / prob


def validate_threeway_probs(
    probs,
    tolerance=ADJUSTED_1X2_SUM_TOLERANCE,
):
    if len(probs) != 3:
        return False

    parsed = []

    for prob in probs:
        value = safe_float(prob)

        if (
            value is None
            or value < 0.0
            or value > 1.0
        ):
            return False

        parsed.append(value)

    total = sum(parsed)

    return (
        abs(total - 1.0)
        <= tolerance
    )


def normalize_probs(probs):
    """
    Strict three-way normalization.

    All three outcomes must be present,
    finite, and non-negative.
    """

    if len(probs) != 3:
        return [
            None
            for _ in probs
        ]

    parsed = []

    for prob in probs:
        value = safe_float(prob)

        if (
            value is None
            or value < 0.0
        ):
            return [
                None,
                None,
                None,
            ]

        parsed.append(value)

    total = sum(parsed)

    if (
        not math.isfinite(total)
        or total <= 0.0
    ):
        return [
            None,
            None,
            None,
        ]

    normalized = [
        value / total
        for value in parsed
    ]

    if not validate_threeway_probs(
        normalized
    ):
        return [
            None,
            None,
            None,
        ]

    return normalized


def parse_stem(stem: str):
    for league in LEAGUE_TO_CONFIG:
        for market in MARKETS:
            suffix = (
                f"_{league}_"
                f"{market}"
            )

            if stem.endswith(suffix):
                return (
                    stem[:-len(suffix)],
                    league,
                    market,
                )

    return (
        None,
        None,
        None,
    )


def validate_xg_inputs(row):
    parsed = {}

    for col in XG_COLS:
        value = safe_float(
            row.get(col)
        )

        if value is None:
            return (
                False,
                {},
                None,
                None,
                f"missing/non-numeric {col}",
            )

        if value < 0.0:
            return (
                False,
                {},
                None,
                None,
                f"negative {col}={value}",
            )

        parsed[col] = value

    component_total = round(
        parsed["home_xg"]
        + parsed["away_xg"],
        6,
    )

    total_difference = abs(
        round(
            parsed["expected_total_goals"]
            - component_total,
            6,
        )
    )

    if (
        total_difference
        > XG_TOTAL_TOLERANCE
    ):
        return (
            False,
            parsed,
            component_total,
            total_difference,
            f"inconsistent expected_total_goals="
            f"{parsed['expected_total_goals']:.6f} "
            f"home_plus_away="
            f"{component_total:.6f} "
            f"difference="
            f"{total_difference:.6f} "
            f"tolerance="
            f"{XG_TOTAL_TOLERANCE:.6f}",
        )

    return (
        True,
        parsed,
        component_total,
        total_difference,
        "",
    )


def validate_xg_dataframe(
    df: pd.DataFrame,
    file_path: Path,
    summary: dict,
) -> pd.DataFrame:

    clean_rows = []
    invalid_count = 0

    for (
        source_index,
        row,
    ) in df.iterrows():

        (
            valid,
            parsed,
            component_total,
            total_difference,
            reason,
        ) = validate_xg_inputs(
            row
        )

        if not valid:
            invalid_count += 1

            summary[
                "rows_rejected_invalid_xg"
            ] += 1

            raw_xg = {
                col: row.get(col)
                for col in XG_COLS
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
                "REJECT INVALID XG BEFORE PRICING | "
                f"file={file_path.name} | "
                f"line={source_index + 2} | "
                f"game_id="
                f"{row.get('game_id', '')} | "
                f"home_team="
                f"{row.get('home_team', '')} | "
                f"away_team="
                f"{row.get('away_team', '')} | "
                f"raw={raw_xg} | "
                f"parsed="
                f"{parsed or 'unavailable'} | "
                f"home_plus_away="
                f"{component_text} | "
                f"difference="
                f"{difference_text} | "
                f"tolerance="
                f"{XG_TOTAL_TOLERANCE:.6f} | "
                f"reason={reason}"
            )

            continue

        clean_row = row.copy()

        for col in XG_COLS:
            clean_row[col] = (
                parsed[col]
            )

        clean_rows.append(
            clean_row
        )

    if invalid_count:
        raise ValueError(
            f"{file_path.name} contains "
            f"{invalid_count} invalid xG row(s); "
            "pricing aborted for this file"
        )

    return pd.DataFrame(
        clean_rows
    ).reset_index(
        drop=True
    )


@dataclass
class LeagueConfig:
    juice: pd.DataFrame
    rho: float


def load_league_config(
    league: str,
) -> LeagueConfig:

    cfg_name = (
        LEAGUE_TO_CONFIG[
            league
        ]
    )

    cfg_dir = (
        CONFIG_ROOT
        / cfg_name
    )

    juice_path = (
        cfg_dir
        / "3way_juice.csv"
    )

    engine_path = (
        cfg_dir
        / "dc_soccer_pricing_engine.csv"
    )

    if not juice_path.exists():
        raise FileNotFoundError(
            f"Missing juice config: "
            f"{juice_path}"
        )

    if not engine_path.exists():
        raise FileNotFoundError(
            f"Missing engine config: "
            f"{engine_path}"
        )

    juice = pd.read_csv(
        juice_path
    )

    engine = pd.read_csv(
        engine_path,
        usecols=["rho"],
    )

    required_juice_cols = {
        "side",
        "fair_prob",
        "extra_juice",
    }

    missing_juice = (
        required_juice_cols
        - set(juice.columns)
    )

    if missing_juice:
        raise ValueError(
            f"{juice_path} missing columns: "
            f"{sorted(missing_juice)}"
        )

    juice["fair_prob"] = (
        pd.to_numeric(
            juice["fair_prob"],
            errors="coerce",
        )
    )

    juice["extra_juice"] = (
        pd.to_numeric(
            juice["extra_juice"],
            errors="coerce",
        )
    )

    juice["side"] = (
        juice["side"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    rho_values = (
        pd.to_numeric(
            engine["rho"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(rho_values) != 1:
        raise ValueError(
            f"{engine_path} must contain "
            "exactly one non-null rho value; "
            f"found {len(rho_values)}"
        )

    return LeagueConfig(
        juice=juice,
        rho=float(
            rho_values[0]
        ),
    )


def build_all_configs():
    configs = {}

    for league in LEAGUE_TO_CONFIG:
        configs[league] = (
            load_league_config(
                league
            )
        )

        log(
            f"CONFIG LOADED {league} | "
            f"juice_rows="
            f"{len(configs[league].juice)} | "
            f"rho="
            f"{configs[league].rho}"
        )

    return configs


def interp_extra_juice(
    juice_df: pd.DataFrame,
    side: str,
    fair_prob: float,
):
    if (
        fair_prob is None
        or pd.isna(fair_prob)
    ):
        return None

    sub = (
        juice_df[
            juice_df["side"]
            == side
        ]
        .copy()
    )

    sub = (
        sub
        .dropna(
            subset=[
                "fair_prob",
                "extra_juice",
            ]
        )
        .sort_values(
            "fair_prob"
        )
    )

    if sub.empty:
        return None

    xs = (
        sub["fair_prob"]
        .to_numpy(
            dtype=float
        )
    )

    ys = (
        sub["extra_juice"]
        .to_numpy(
            dtype=float
        )
    )

    fair_prob = float(
        fair_prob
    )

    if fair_prob <= xs[0]:
        return float(
            ys[0]
        )

    if fair_prob >= xs[-1]:
        return float(
            ys[-1]
        )

    return float(
        np.interp(
            fair_prob,
            xs,
            ys,
        )
    )


def poisson_score_probs(
    lam: float,
):
    lam = float(lam)

    if (
        not math.isfinite(lam)
        or lam < 0
    ):
        raise ValueError(
            f"Invalid Poisson lambda: "
            f"{lam}"
        )

    p = math.exp(-lam)

    probs = [p]
    cumulative = p

    for goals in range(
        1,
        POISSON_MAX_GOALS + 1,
    ):
        if (
            1.0 - cumulative
            <= POISSON_TAIL_TOLERANCE
        ):
            break

        p *= (
            lam / goals
        )

        probs.append(p)
        cumulative += p

    if (
        1.0 - cumulative
        > 1e-10
    ):
        raise ValueError(
            "Poisson tail did not converge "
            f"for lambda={lam}; "
            f"remaining_mass="
            f"{1.0 - cumulative}"
        )

    return probs


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:

    if (
        home_goals == 0
        and away_goals == 0
    ):
        return (
            1.0
            - (
                lambda_home
                * lambda_away
                * rho
            )
        )

    if (
        home_goals == 0
        and away_goals == 1
    ):
        return (
            1.0
            + (
                lambda_home
                * rho
            )
        )

    if (
        home_goals == 1
        and away_goals == 0
    ):
        return (
            1.0
            + (
                lambda_away
                * rho
            )
        )

    if (
        home_goals == 1
        and away_goals == 1
    ):
        return (
            1.0 - rho
        )

    return 1.0


@lru_cache(
    maxsize=10000
)
def price_from_xg(
    home_xg: float,
    away_xg: float,
    rho: float,
):
    lambda_home = float(
        home_xg
    )

    lambda_away = float(
        away_xg
    )

    rho = float(rho)

    if (
        not math.isfinite(
            lambda_home
        )
        or not math.isfinite(
            lambda_away
        )
        or lambda_home < 0
        or lambda_away < 0
    ):
        return None

    home_scores = (
        poisson_score_probs(
            lambda_home
        )
    )

    away_scores = (
        poisson_score_probs(
            lambda_away
        )
    )

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    over2_5 = 0.0
    under2_5 = 0.0

    over3_5 = 0.0
    under3_5 = 0.0

    btts_yes = 0.0
    btts_no = 0.0

    total_mass = 0.0

    for (
        home_goals,
        p_home,
    ) in enumerate(
        home_scores
    ):
        for (
            away_goals,
            p_away,
        ) in enumerate(
            away_scores
        ):
            tau = (
                dixon_coles_tau(
                    home_goals,
                    away_goals,
                    lambda_home,
                    lambda_away,
                    rho,
                )
            )

            if tau < 0:
                raise ValueError(
                    "Dixon-Coles produced a "
                    "negative low-score multiplier: "
                    f"home_xg={lambda_home}, "
                    f"away_xg={lambda_away}, "
                    f"rho={rho}, "
                    f"score="
                    f"{home_goals}-"
                    f"{away_goals}, "
                    f"tau={tau}"
                )

            p = (
                p_home
                * p_away
                * tau
            )

            total_mass += p

            if (
                home_goals
                > away_goals
            ):
                home_win += p

            elif (
                home_goals
                == away_goals
            ):
                draw += p

            else:
                away_win += p

            total_goals = (
                home_goals
                + away_goals
            )

            if (
                total_goals
                > 2.5
            ):
                over2_5 += p

            else:
                under2_5 += p

            if (
                total_goals
                > 3.5
            ):
                over3_5 += p

            else:
                under3_5 += p

            if (
                home_goals > 0
                and away_goals > 0
            ):
                btts_yes += p

            else:
                btts_no += p

    if total_mass <= 0:
        raise ValueError(
            "Invalid Dixon-Coles total "
            "probability mass for "
            f"home_xg={lambda_home}, "
            f"away_xg={lambda_away}, "
            f"rho={rho}"
        )

    home_win /= total_mass
    draw /= total_mass
    away_win /= total_mass

    over2_5 /= total_mass
    under2_5 /= total_mass

    over3_5 /= total_mass
    under3_5 /= total_mass

    btts_yes /= total_mass
    btts_no /= total_mass

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over2_5": over2_5,
        "under2_5": under2_5,
        "over3_5": over3_5,
        "under3_5": under3_5,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
    }


def get_pricing(
    home_xg,
    away_xg,
    expected_total_goals,
    rho,
):
    (
        valid,
        parsed,
        _,
        _,
        _,
    ) = validate_xg_inputs(
        {
            "home_xg": home_xg,
            "away_xg": away_xg,
            "expected_total_goals":
                expected_total_goals,
        }
    )

    if not valid:
        return None

    return price_from_xg(
        parsed["home_xg"],
        parsed["away_xg"],
        float(rho),
    )


def process_match_odds(
    df: pd.DataFrame,
    cfg: LeagueConfig,
    file_path: Path,
    summary: dict,
) -> pd.DataFrame:

    out_rows = []

    for (
        source_index,
        row,
    ) in df.iterrows():

        r = row.to_dict()

        r.pop(
            "engine_match_distance",
            None,
        )

        home_prob = safe_float(
            r.get("home_prob")
        )

        draw_prob = safe_float(
            r.get("draw_prob")
        )

        away_prob = safe_float(
            r.get("away_prob")
        )

        pricing = get_pricing(
            safe_float(
                r.get("home_xg")
            ),
            safe_float(
                r.get("away_xg")
            ),
            safe_float(
                r.get(
                    "expected_total_goals"
                )
            ),
            cfg.rho,
        )

        if pricing is None:
            continue

        home_extra = (
            interp_extra_juice(
                cfg.juice,
                "home",
                home_prob,
            )
        )

        draw_extra = (
            interp_extra_juice(
                cfg.juice,
                "draw",
                draw_prob,
            )
        )

        away_extra = (
            interp_extra_juice(
                cfg.juice,
                "away",
                away_prob,
            )
        )

        raw_home = (
            None
            if (
                home_prob is None
                or home_extra is None
            )
            else (
                home_prob
                + home_extra
            )
        )

        raw_draw = (
            None
            if (
                draw_prob is None
                or draw_extra is None
            )
            else (
                draw_prob
                + draw_extra
            )
        )

        raw_away = (
            None
            if (
                away_prob is None
                or away_extra is None
            )
            else (
                away_prob
                + away_extra
            )
        )

        raw_probs = [
            raw_home,
            raw_draw,
            raw_away,
        ]

        juiced_probs = (
            normalize_probs(
                raw_probs
            )
        )

        if not validate_threeway_probs(
            juiced_probs
        ):
            summary[
                "rows_rejected_invalid_adjusted_1x2"
            ] += 1

            log(
                "REJECT INVALID ADJUSTED 1X2 | "
                f"file={file_path.name} | "
                f"line={source_index + 2} | "
                f"game_id="
                f"{r.get('game_id', '')} | "
                f"home_team="
                f"{r.get('home_team', '')} | "
                f"away_team="
                f"{r.get('away_team', '')} | "
                f"raw_home={raw_home!r} | "
                f"raw_draw={raw_draw!r} | "
                f"raw_away={raw_away!r} | "
                f"normalized="
                f"{juiced_probs!r}"
            )

            continue

        (
            juiced_home_prob,
            juiced_draw_prob,
            juiced_away_prob,
        ) = juiced_probs

        r["home_extra_juice"] = (
            home_extra
        )

        r["draw_extra_juice"] = (
            draw_extra
        )

        r["away_extra_juice"] = (
            away_extra
        )

        r["juiced_home_prob"] = (
            juiced_home_prob
        )

        r["juiced_draw_prob"] = (
            juiced_draw_prob
        )

        r["juiced_away_prob"] = (
            juiced_away_prob
        )

        r["juiced_home_decimal"] = (
            safe_decimal(
                juiced_home_prob
            )
        )

        r["juiced_draw_decimal"] = (
            safe_decimal(
                juiced_draw_prob
            )
        )

        r["juiced_away_decimal"] = (
            safe_decimal(
                juiced_away_prob
            )
        )

        r["engine_lambda_home"] = (
            pricing["lambda_home"]
        )

        r["engine_lambda_away"] = (
            pricing["lambda_away"]
        )

        r["engine_home_prob"] = (
            pricing["home_win"]
        )

        r["engine_draw_prob"] = (
            pricing["draw"]
        )

        r["engine_away_prob"] = (
            pricing["away_win"]
        )

        r["engine_home_fair_decimal"] = (
            safe_decimal(
                pricing["home_win"]
            )
        )

        r["engine_draw_fair_decimal"] = (
            safe_decimal(
                pricing["draw"]
            )
        )

        r["engine_away_fair_decimal"] = (
            safe_decimal(
                pricing["away_win"]
            )
        )

        out_rows.append(r)

    return pd.DataFrame(
        out_rows
    )


def process_total_25(
    df: pd.DataFrame,
    cfg: LeagueConfig,
) -> pd.DataFrame:

    out_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()

        r.pop(
            "engine_match_distance",
            None,
        )

        pricing = get_pricing(
            safe_float(
                r.get("home_xg")
            ),
            safe_float(
                r.get("away_xg")
            ),
            safe_float(
                r.get(
                    "expected_total_goals"
                )
            ),
            cfg.rho,
        )

        if pricing is None:
            continue

        r["engine_lambda_home"] = (
            pricing["lambda_home"]
        )

        r["engine_lambda_away"] = (
            pricing["lambda_away"]
        )

        r["fair_over_decimal"] = (
            safe_decimal(
                pricing["over2_5"]
            )
        )

        r["fair_under_decimal"] = (
            safe_decimal(
                pricing["under2_5"]
            )
        )

        r["engine_over_prob"] = (
            pricing["over2_5"]
        )

        r["engine_under_prob"] = (
            pricing["under2_5"]
        )

        out_rows.append(r)

    return pd.DataFrame(
        out_rows
    )


def process_total_35(
    df: pd.DataFrame,
    cfg: LeagueConfig,
) -> pd.DataFrame:

    out_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()

        r.pop(
            "engine_match_distance",
            None,
        )

        pricing = get_pricing(
            safe_float(
                r.get("home_xg")
            ),
            safe_float(
                r.get("away_xg")
            ),
            safe_float(
                r.get(
                    "expected_total_goals"
                )
            ),
            cfg.rho,
        )

        if pricing is None:
            continue

        r["engine_lambda_home"] = (
            pricing["lambda_home"]
        )

        r["engine_lambda_away"] = (
            pricing["lambda_away"]
        )

        r["fair_over_decimal"] = (
            safe_decimal(
                pricing["over3_5"]
            )
        )

        r["fair_under_decimal"] = (
            safe_decimal(
                pricing["under3_5"]
            )
        )

        r["engine_over_prob"] = (
            pricing["over3_5"]
        )

        r["engine_under_prob"] = (
            pricing["under3_5"]
        )

        out_rows.append(r)

    return pd.DataFrame(
        out_rows
    )


def process_btts(
    df: pd.DataFrame,
    cfg: LeagueConfig,
) -> pd.DataFrame:

    out_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()

        r.pop(
            "engine_match_distance",
            None,
        )

        pricing = get_pricing(
            safe_float(
                r.get("home_xg")
            ),
            safe_float(
                r.get("away_xg")
            ),
            safe_float(
                r.get(
                    "expected_total_goals"
                )
            ),
            cfg.rho,
        )

        if pricing is None:
            continue

        r["engine_lambda_home"] = (
            pricing["lambda_home"]
        )

        r["engine_lambda_away"] = (
            pricing["lambda_away"]
        )

        r["fair_btts_yes_decimal"] = (
            safe_decimal(
                pricing["btts_yes"]
            )
        )

        r["fair_btts_no_decimal"] = (
            safe_decimal(
                pricing["btts_no"]
            )
        )

        r["engine_btts_yes_prob"] = (
            pricing["btts_yes"]
        )

        r["engine_btts_no_prob"] = (
            pricing["btts_no"]
        )

        out_rows.append(r)

    return pd.DataFrame(
        out_rows
    )


def process_file(
    file_path: Path,
    configs: dict,
    summary: dict,
):
    out_path = (
        OUTPUT_DIR
        / file_path.name
    )

    try:
        (
            _,
            league,
            market,
        ) = parse_stem(
            file_path.stem
        )

        if (
            not league
            or not market
        ):
            log(
                "SKIP unrecognized file: "
                f"{file_path.name}"
            )

            summary["skipped"] += 1
            return

        if league not in configs:
            log(
                "SKIP no config for "
                f"league={league}: "
                f"{file_path.name}"
            )

            summary["skipped"] += 1
            return

        df = pd.read_csv(
            file_path
        )

        if df.empty:
            log(
                f"EMPTY: "
                f"{file_path.name}"
            )

            summary["empty"] += 1
            return

        # Prevent stale output from surviving
        # when validation fails on this run.
        if out_path.exists():
            out_path.unlink()

        df = validate_xg_dataframe(
            df,
            file_path,
            summary,
        )

        cfg = configs[
            league
        ]

        if market == "match_odds":
            out_df = (
                process_match_odds(
                    df,
                    cfg,
                    file_path,
                    summary,
                )
            )

        elif market == "total_25":
            out_df = (
                process_total_25(
                    df,
                    cfg,
                )
            )

        elif market == "total_35":
            out_df = (
                process_total_35(
                    df,
                    cfg,
                )
            )

        elif market == "btts":
            out_df = (
                process_btts(
                    df,
                    cfg,
                )
            )

        else:
            log(
                "SKIP unsupported "
                f"market={market}: "
                f"{file_path.name}"
            )

            summary["skipped"] += 1
            return

        if len(out_df) != len(df):
            raise ValueError(
                "Pricing fail-safe rejected rows "
                "after pre-validation for "
                f"{file_path.name}: "
                f"validated={len(df)} "
                f"priced={len(out_df)}"
            )

        out_df.to_csv(
            out_path,
            index=False,
        )

        log(
            f"WROTE {out_path} "
            f"({len(out_df)} rows)"
        )

        summary[
            "files_written"
        ] += 1

        summary[
            "rows_written"
        ] += len(out_df)

    except Exception as e:
        if out_path.exists():
            out_path.unlink()

        log(
            f"ERROR processing "
            f"{file_path}: "
            f"{e}\n"
            f"{traceback.format_exc()}"
        )

        summary[
            "errors"
        ] += 1


def main():
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== apply_juice RUN "
            f"{datetime.now(timezone.utc).isoformat()} ===\n"
        )

    summary = {
        "files_written": 0,
        "rows_written": 0,
        "rows_rejected_invalid_xg": 0,
        "rows_rejected_invalid_adjusted_1x2": 0,
        "empty": 0,
        "skipped": 0,
        "errors": 0,
    }

    log(
        "xG pricing fail-safe enabled | "
        f"total_tolerance="
        f"{XG_TOTAL_TOLERANCE:.6f} | "
        "negative_values=reject"
    )

    log(
        "adjusted 1X2 validation enabled | "
        "normalization_requires_all_three=true | "
        f"sum_tolerance="
        f"{ADJUSTED_1X2_SUM_TOLERANCE:.12f}"
    )

    configs = (
        build_all_configs()
    )

    input_files = sorted(
        f
        for f
        in INPUT_DIR.glob("*.csv")
        if f.name.endswith(
            ".csv"
        )
    )

    for file_path in input_files:
        process_file(
            file_path,
            configs,
            summary,
        )

    log(
        "SUMMARY: "
        f"files_written="
        f"{summary['files_written']} | "
        f"rows_written="
        f"{summary['rows_written']} | "
        f"rows_rejected_invalid_xg="
        f"{summary['rows_rejected_invalid_xg']} | "
        "rows_rejected_invalid_adjusted_1x2="
        f"{summary['rows_rejected_invalid_adjusted_1x2']} | "
        f"empty="
        f"{summary['empty']} | "
        f"skipped="
        f"{summary['skipped']} | "
        f"errors="
        f"{summary['errors']}"
    )

    if (
        summary["errors"]
        or summary[
            "rows_rejected_invalid_xg"
        ]
        or summary[
            "rows_rejected_invalid_adjusted_1x2"
        ]
    ):
        log("FAILED")

        raise RuntimeError(
            "apply_juice aborted because "
            "one or more input rows/files "
            "failed validation or pricing"
        )

    log("COMPLETE")

    print(
        "apply_juice complete."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        with open(
            LOG_FILE,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                f"FATAL:\n"
                f"{e}\n"
                f"{traceback.format_exc()}"
            )

        raise