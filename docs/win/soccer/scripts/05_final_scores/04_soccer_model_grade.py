#!/usr/bin/env python3
# docs/win/soccer/scripts/05_final_scores/04_soccer_model_grade.py

from __future__ import annotations

import math
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW_DIR = Path("docs/win/soccer/01_merge")
ENGINE_DIR = Path("docs/win/soccer/02_juice")
SCORES_DIR = Path("docs/win/soccer/05_final_scores/results/final_scores")
OUT_DIR = Path("docs/win/soccer/05_final_scores/model_evaluation")
ERR_DIR = Path("docs/win/soccer/05_final_scores/errors")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ERR_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = OUT_DIR / "SOCCER_model_results.csv"
ERROR_LOG = ERR_DIR / "soccer_model_grade_errors.txt"
SUMMARY_LOG = ERR_DIR / "soccer_model_grade_summary.txt"

RAW_REQUIRED = [
    "game_id",
    "sport",
    "league",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
    "home_prob",
    "draw_prob",
    "away_prob",
    "home_xg",
    "away_xg",
    "expected_total_goals",
]

ENGINE_REQUIRED = [
    "game_id",
    "engine_home_prob",
    "engine_draw_prob",
    "engine_away_prob",
    "engine_lambda_home",
    "engine_lambda_away",
]

SCORE_REQUIRED = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "match_time",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]

PROB_SUM_TOLERANCE = 0.001
LOG_EPSILON = 1e-15


def reset_logs():
    ERROR_LOG.write_text("", encoding="utf-8")
    SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def log_summary(msg):
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)

        if df.empty:
            log_error(f"EMPTY FILE | {path}")

        return df

    except Exception as e:
        log_error(f"READ ERROR | {path} | {e}")
        return pd.DataFrame()


def valid_headers(df, required, path):
    missing = [c for c in required if c not in df.columns]

    if missing:
        log_error(f"MISSING HEADERS | {path} | missing={missing}")
        return False

    return True


def game_id(value):
    if pd.isna(value):
        return ""

    s = str(value).strip()

    if s.lower() in {"", "nan", "none", "<na>"}:
        return ""

    return s[:-2] if s.endswith(".0") else s


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def validate_probs(h, d, a):
    probs = [number(h), number(d), number(a)]

    if any(v is None for v in probs):
        return False, "non_numeric_or_non_finite"

    if any(v < 0 or v > 1 for v in probs):
        return False, "outside_0_1"

    total = sum(probs)

    if abs(total - 1.0) > PROB_SUM_TOLERANCE:
        return False, f"sum={total:.12f}"

    return True, "valid"


def result_from_score(home, away):
    if home > away:
        return "home"

    if away > home:
        return "away"

    return "draw"


def one_hot(result):
    return (
        1.0 if result == "home" else 0.0,
        1.0 if result == "draw" else 0.0,
        1.0 if result == "away" else 0.0,
    )


def probability_metrics(h, d, a, result):
    probs = [h, d, a]
    actual = list(one_hot(result))

    brier = sum(
        (p - y) ** 2
        for p, y in zip(probs, actual)
    )

    idx = {
        "home": 0,
        "draw": 1,
        "away": 2,
    }[result]

    log_loss = -math.log(
        max(
            min(probs[idx], 1.0),
            LOG_EPSILON,
        )
    )

    rps = (
        (probs[0] - actual[0]) ** 2
        + (
            (probs[0] + probs[1])
            - (actual[0] + actual[1])
        ) ** 2
    ) / 2.0

    return brier, log_loss, rps


def build_score_index():
    idx = {}
    files = 0
    rows = 0

    for path in sorted(
        SCORES_DIR.glob("*/*.csv")
    ):
        df = read_csv(path)

        if df.empty:
            continue

        if not valid_headers(
            df,
            SCORE_REQUIRED,
            path,
        ):
            continue

        files += 1

        for _, r in df.iterrows():
            rows += 1

            gid = game_id(
                r.get("game_id")
            )

            league = str(
                r.get("league", "")
            ).lower().strip()

            hs = number(
                r.get("home_score")
            )

            aws = number(
                r.get("away_score")
            )

            if (
                not gid
                or not league
                or hs is None
                or aws is None
            ):
                log_error(
                    f"SCORE ROW EXCLUDED | "
                    f"{path} | "
                    f"game_id={gid}"
                )
                continue

            key = (
                league,
                gid,
            )

            if key in idx:
                log_error(
                    f"DUPLICATE FINAL SCORE | "
                    f"league={league} "
                    f"game_id={gid}"
                )
                continue

            idx[key] = {
                "score_game_date": str(
                    r.get("game_date", "")
                ).strip(),
                "score_match_time": str(
                    r.get("match_time", "")
                ).strip(),
                "score_home_team": str(
                    r.get("home_team", "")
                ).strip(),
                "score_away_team": str(
                    r.get("away_team", "")
                ).strip(),
                "home_score": hs,
                "away_score": aws,
            }

    log_summary(
        f"FINAL SCORES INDEXED | "
        f"files={files} "
        f"rows={rows} "
        f"valid_games={len(idx)}"
    )

    return idx


def build_engine_index(path):
    df = read_csv(path)

    if df.empty:
        return {}

    if not valid_headers(
        df,
        ENGINE_REQUIRED,
        path,
    ):
        return {}

    idx = {}

    for _, r in df.iterrows():
        gid = game_id(
            r.get("game_id")
        )

        if not gid:
            continue

        if gid in idx:
            log_error(
                f"DUPLICATE ENGINE GAME_ID | "
                f"{path} | "
                f"game_id={gid}"
            )
            continue

        idx[gid] = r.to_dict()

    return idx


def process():
    scores = build_score_index()

    raw_files = sorted(
        RAW_DIR.glob("*_match_odds.csv")
    )

    log_summary(
        f"RAW MATCH_ODDS FILES FOUND | "
        f"count={len(raw_files)}"
    )

    out = []
    seen = set()

    counts = {
        "raw_rows": 0,
        "known_scores": 0,
        "no_score": 0,
        "raw_invalid": 0,
        "engine_missing": 0,
        "engine_invalid": 0,
    }

    for raw_path in raw_files:
        raw_df = read_csv(
            raw_path
        )

        if raw_df.empty:
            continue

        if not valid_headers(
            raw_df,
            RAW_REQUIRED,
            raw_path,
        ):
            continue

        engine_idx = build_engine_index(
            ENGINE_DIR
            / raw_path.name
        )

        for _, r in raw_df.iterrows():
            counts["raw_rows"] += 1

            gid = game_id(
                r.get("game_id")
            )

            league = str(
                r.get("league", "")
            ).lower().strip()

            key = (
                league,
                gid,
            )

            if not gid or not league:
                log_error(
                    f"RAW ROW EXCLUDED | "
                    f"{raw_path} | "
                    f"blank league/game_id"
                )
                continue

            if key in seen:
                log_error(
                    f"DUPLICATE RAW MODEL GAME | "
                    f"league={league} "
                    f"game_id={gid}"
                )
                continue

            seen.add(key)

            score = scores.get(
                key
            )

            if score is None:
                counts["no_score"] += 1
                continue

            counts["known_scores"] += 1

            hs = score["home_score"]
            aws = score["away_score"]

            actual_result = (
                result_from_score(
                    hs,
                    aws,
                )
            )

            (
                actual_home,
                actual_draw,
                actual_away,
            ) = one_hot(
                actual_result
            )

            rh = number(
                r.get("home_prob")
            )

            rd = number(
                r.get("draw_prob")
            )

            ra = number(
                r.get("away_prob")
            )

            raw_valid, raw_status = (
                validate_probs(
                    rh,
                    rd,
                    ra,
                )
            )

            if raw_valid:
                (
                    raw_brier,
                    raw_log_loss,
                    raw_rps,
                ) = probability_metrics(
                    rh,
                    rd,
                    ra,
                    actual_result,
                )

            else:
                counts[
                    "raw_invalid"
                ] += 1

                raw_brier = None
                raw_log_loss = None
                raw_rps = None

                log_error(
                    f"INVALID RAW PROBABILITIES | "
                    f"league={league} "
                    f"game_id={gid} | "
                    f"{raw_status}"
                )

            e = engine_idx.get(
                gid
            )

            if e is None:
                counts[
                    "engine_missing"
                ] += 1

                eh = None
                ed = None
                ea = None
                elh = None
                ela = None

                engine_valid = False
                engine_status = (
                    "missing_engine_row"
                )

                engine_brier = None
                engine_log_loss = None
                engine_rps = None

            else:
                eh = number(
                    e.get(
                        "engine_home_prob"
                    )
                )

                ed = number(
                    e.get(
                        "engine_draw_prob"
                    )
                )

                ea = number(
                    e.get(
                        "engine_away_prob"
                    )
                )

                elh = number(
                    e.get(
                        "engine_lambda_home"
                    )
                )

                ela = number(
                    e.get(
                        "engine_lambda_away"
                    )
                )

                (
                    engine_valid,
                    engine_status,
                ) = validate_probs(
                    eh,
                    ed,
                    ea,
                )

                if engine_valid:
                    (
                        engine_brier,
                        engine_log_loss,
                        engine_rps,
                    ) = probability_metrics(
                        eh,
                        ed,
                        ea,
                        actual_result,
                    )

                else:
                    counts[
                        "engine_invalid"
                    ] += 1

                    engine_brier = None
                    engine_log_loss = None
                    engine_rps = None

                    log_error(
                        f"INVALID ENGINE PROBABILITIES | "
                        f"league={league} "
                        f"game_id={gid} | "
                        f"{engine_status}"
                    )

            hxg = number(
                r.get("home_xg")
            )

            axg = number(
                r.get("away_xg")
            )

            txg = number(
                r.get(
                    "expected_total_goals"
                )
            )

            actual_total = (
                hs + aws
            )

            he = (
                hxg - hs
                if hxg is not None
                else None
            )

            ae = (
                axg - aws
                if axg is not None
                else None
            )

            te = (
                txg - actual_total
                if txg is not None
                else None
            )

            out.append(
                {
                    "game_id": gid,
                    "sport": str(
                        r.get(
                            "sport",
                            "",
                        )
                    ).strip(),
                    "league": league,
                    "match_date": str(
                        r.get(
                            "match_date",
                            "",
                        )
                    ).strip(),
                    "match_time": str(
                        r.get(
                            "match_time",
                            "",
                        )
                    ).strip(),
                    "home_team": str(
                        r.get(
                            "home_team",
                            "",
                        )
                    ).strip(),
                    "away_team": str(
                        r.get(
                            "away_team",
                            "",
                        )
                    ).strip(),
                    **score,
                    "actual_total_goals": actual_total,
                    "actual_result": actual_result,
                    "actual_home": actual_home,
                    "actual_draw": actual_draw,
                    "actual_away": actual_away,
                    "raw_home_prob": rh,
                    "raw_draw_prob": rd,
                    "raw_away_prob": ra,
                    "raw_probs_valid": raw_valid,
                    "raw_probs_status": raw_status,
                    "raw_brier": raw_brier,
                    "raw_log_loss": raw_log_loss,
                    "raw_rps": raw_rps,
                    "engine_home_prob": eh,
                    "engine_draw_prob": ed,
                    "engine_away_prob": ea,
                    "engine_probs_valid": engine_valid,
                    "engine_probs_status": engine_status,
                    "engine_brier": engine_brier,
                    "engine_log_loss": engine_log_loss,
                    "engine_rps": engine_rps,
                    "home_xg": hxg,
                    "away_xg": axg,
                    "expected_total_goals": txg,
                    "engine_lambda_home": elh,
                    "engine_lambda_away": ela,
                    "home_xg_error": he,
                    "home_xg_abs_error": (
                        abs(he)
                        if he is not None
                        else None
                    ),
                    "home_xg_squared_error": (
                        he ** 2
                        if he is not None
                        else None
                    ),
                    "away_xg_error": ae,
                    "away_xg_abs_error": (
                        abs(ae)
                        if ae is not None
                        else None
                    ),
                    "away_xg_squared_error": (
                        ae ** 2
                        if ae is not None
                        else None
                    ),
                    "total_xg_error": te,
                    "total_xg_abs_error": (
                        abs(te)
                        if te is not None
                        else None
                    ),
                    "total_xg_squared_error": (
                        te ** 2
                        if te is not None
                        else None
                    ),
                }
            )

    if not out:
        if MASTER_FILE.exists():
            MASTER_FILE.unlink()

        log_error(
            "NO MODEL EVALUATION ROWS "
            "WITH KNOWN FINAL SCORES"
        )
        return

    result = (
        pd.DataFrame(out)
        .sort_values(
            [
                "match_date",
                "league",
                "game_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result.to_csv(
        MASTER_FILE,
        index=False,
    )

    log_summary(
        f"MODEL MASTER WRITTEN | "
        f"rows={len(result)} | "
        f"{MASTER_FILE}"
    )

    log_summary(
        "ACCOUNTING | "
        + " | ".join(
            f"{k}={v}"
            for k, v
            in counts.items()
        )
    )


def main():
    reset_logs()

    log_summary(
        f"=== START "
        f"04_soccer_model_grade.py "
        f"{datetime.now().isoformat()} ==="
    )

    try:
        process()

        log_summary(
            f"=== END "
            f"04_soccer_model_grade.py "
            f"{datetime.now().isoformat()} ==="
        )

        print(
            "Soccer model grading complete."
        )

    except Exception as e:
        log_error(
            f"FATAL | {e}\n"
            f"{traceback.format_exc()}"
        )
        raise


if __name__ == "__main__":
    main()