#!/usr/bin/env python3
# docs/win/soccer/scripts/05_final_scores/05_soccer_model_reports.py

from __future__ import annotations

import math
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(
    "docs/win/soccer/05_final_scores/model_evaluation"
)

ERR_DIR = Path(
    "docs/win/soccer/05_final_scores/errors"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ERR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MASTER_FILE = (
    MODEL_DIR
    / "SOCCER_model_results.csv"
)

METRICS_FILE = (
    MODEL_DIR
    / "SOCCER_model_metrics.csv"
)

CALIBRATION_FILE = (
    MODEL_DIR
    / "SOCCER_model_calibration.csv"
)

XG_FILE = (
    MODEL_DIR
    / "SOCCER_xg_metrics.csv"
)

ERROR_LOG = (
    ERR_DIR
    / "soccer_model_reports_errors.txt"
)

SUMMARY_LOG = (
    ERR_DIR
    / "soccer_model_reports_summary.txt"
)

REQUIRED = [
    "game_id",
    "league",
    "match_date",
    "actual_result",
    "actual_home",
    "actual_draw",
    "actual_away",
    "raw_home_prob",
    "raw_draw_prob",
    "raw_away_prob",
    "raw_probs_valid",
    "raw_brier",
    "raw_log_loss",
    "raw_rps",
    "engine_home_prob",
    "engine_draw_prob",
    "engine_away_prob",
    "engine_probs_valid",
    "engine_brier",
    "engine_log_loss",
    "engine_rps",
    "home_xg",
    "away_xg",
    "expected_total_goals",
    "home_score",
    "away_score",
    "actual_total_goals",
]

MODELS = {
    "raw": (
        "raw_probs_valid",
        "raw_home_prob",
        "raw_draw_prob",
        "raw_away_prob",
        "raw_brier",
        "raw_log_loss",
        "raw_rps",
    ),
    "engine": (
        "engine_probs_valid",
        "engine_home_prob",
        "engine_draw_prob",
        "engine_away_prob",
        "engine_brier",
        "engine_log_loss",
        "engine_rps",
    ),
}

OUTCOMES = {
    "home": "actual_home",
    "draw": "actual_draw",
    "away": "actual_away",
}


def reset_logs():
    ERROR_LOG.write_text(
        "",
        encoding="utf-8",
    )

    SUMMARY_LOG.write_text(
        "",
        encoding="utf-8",
    )


def log_error(msg):
    with open(
        ERROR_LOG,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"[{datetime.now().isoformat()}] "
            f"{msg}\n"
        )


def log_summary(msg):
    with open(
        SUMMARY_LOG,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"[{datetime.now().isoformat()}] "
            f"{msg}\n"
        )


def read_master():
    try:
        if not MASTER_FILE.exists():
            log_error(
                f"MISSING FILE | "
                f"{MASTER_FILE}"
            )
            return pd.DataFrame()

        df = pd.read_csv(
            MASTER_FILE
        )

        missing = [
            c
            for c in REQUIRED
            if c not in df.columns
        ]

        if missing:
            log_error(
                f"MISSING HEADERS | "
                f"{MASTER_FILE} | "
                f"missing={missing}"
            )
            return pd.DataFrame()

        return df

    except Exception as e:
        log_error(
            f"READ ERROR | "
            f"{MASTER_FILE} | "
            f"{e}"
        )
        return pd.DataFrame()


def true_mask(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


def groups(df):
    yield (
        "all",
        "ALL",
        df,
    )

    for league, part in df.groupby(
        "league",
        dropna=False,
    ):
        yield (
            "league",
            str(league),
            part,
        )


def build_model_metrics(df):
    rows = []

    for scope, league, part in groups(
        df
    ):
        for model, spec in MODELS.items():
            (
                valid_col,
                _,
                _,
                _,
                brier_col,
                log_col,
                rps_col,
            ) = spec

            valid = part.loc[
                true_mask(
                    part[valid_col]
                )
            ]

            brier = pd.to_numeric(
                valid[brier_col],
                errors="coerce",
            )

            logloss = pd.to_numeric(
                valid[log_col],
                errors="coerce",
            )

            rps = pd.to_numeric(
                valid[rps_col],
                errors="coerce",
            )

            ok = (
                brier.notna()
                & logloss.notna()
                & rps.notna()
            )

            rows.append(
                {
                    "scope": scope,
                    "league": league,
                    "model": model,
                    "sample_count": int(
                        ok.sum()
                    ),
                    "brier_score": (
                        float(
                            brier[
                                ok
                            ].mean()
                        )
                        if ok.any()
                        else None
                    ),
                    "log_loss": (
                        float(
                            logloss[
                                ok
                            ].mean()
                        )
                        if ok.any()
                        else None
                    ),
                    "rps": (
                        float(
                            rps[
                                ok
                            ].mean()
                        )
                        if ok.any()
                        else None
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_calibration(df):
    rows = []

    for scope, league, part in groups(
        df
    ):
        for model, spec in MODELS.items():
            (
                valid_col,
                hp,
                dp,
                ap,
                _,
                _,
                _,
            ) = spec

            valid = part.loc[
                true_mask(
                    part[valid_col]
                )
            ]

            prob_cols = {
                "home": hp,
                "draw": dp,
                "away": ap,
            }

            for outcome, actual_col in (
                OUTCOMES.items()
            ):
                temp = pd.DataFrame(
                    {
                        "p": pd.to_numeric(
                            valid[
                                prob_cols[
                                    outcome
                                ]
                            ],
                            errors="coerce",
                        ),
                        "y": pd.to_numeric(
                            valid[
                                actual_col
                            ],
                            errors="coerce",
                        ),
                    }
                ).dropna()

                if not temp.empty:
                    temp[
                        "bucket"
                    ] = temp[
                        "p"
                    ].apply(
                        lambda p: min(
                            9,
                            max(
                                0,
                                int(
                                    p * 10
                                ),
                            ),
                        )
                    )

                for i in range(
                    10
                ):
                    if temp.empty:
                        bucket = temp
                    else:
                        bucket = temp[
                            temp[
                                "bucket"
                            ]
                            == i
                        ]

                    n = len(
                        bucket
                    )

                    lo = i / 10
                    hi = (
                        i + 1
                    ) / 10

                    mean_p = (
                        float(
                            bucket[
                                "p"
                            ].mean()
                        )
                        if n
                        else None
                    )

                    observed = (
                        float(
                            bucket[
                                "y"
                            ].mean()
                        )
                        if n
                        else None
                    )

                    rows.append(
                        {
                            "scope": scope,
                            "league": league,
                            "model": model,
                            "outcome": outcome,
                            "bucket": (
                                f"{lo:.1f}"
                                f"_to_"
                                f"{hi:.1f}"
                            ),
                            "bucket_low": lo,
                            "bucket_high": hi,
                            "sample_count": n,
                            "mean_predicted_probability": mean_p,
                            "observed_rate": observed,
                            "calibration_gap": (
                                observed
                                - mean_p
                                if n
                                else None
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def build_xg_metrics(df):
    rows = []

    for scope, league, part in groups(
        df
    ):
        hxg = pd.to_numeric(
            part["home_xg"],
            errors="coerce",
        )

        axg = pd.to_numeric(
            part["away_xg"],
            errors="coerce",
        )

        txg = pd.to_numeric(
            part[
                "expected_total_goals"
            ],
            errors="coerce",
        )

        hs = pd.to_numeric(
            part["home_score"],
            errors="coerce",
        )

        aws = pd.to_numeric(
            part["away_score"],
            errors="coerce",
        )

        ats = pd.to_numeric(
            part[
                "actual_total_goals"
            ],
            errors="coerce",
        )

        ok = (
            hxg.notna()
            & axg.notna()
            & txg.notna()
            & hs.notna()
            & aws.notna()
            & ats.notna()
        )

        he = (
            hxg[ok]
            - hs[ok]
        )

        ae = (
            axg[ok]
            - aws[ok]
        )

        te = (
            txg[ok]
            - ats[ok]
        )

        def mae(s):
            return (
                float(
                    s.abs().mean()
                )
                if len(s)
                else None
            )

        def rmse(s):
            return (
                math.sqrt(
                    float(
                        (
                            s ** 2
                        ).mean()
                    )
                )
                if len(s)
                else None
            )

        def bias(s):
            return (
                float(
                    s.mean()
                )
                if len(s)
                else None
            )

        rows.append(
            {
                "scope": scope,
                "league": league,
                "sample_count": int(
                    ok.sum()
                ),
                "home_xg_mae": mae(
                    he
                ),
                "home_xg_rmse": rmse(
                    he
                ),
                "home_xg_bias": bias(
                    he
                ),
                "away_xg_mae": mae(
                    ae
                ),
                "away_xg_rmse": rmse(
                    ae
                ),
                "away_xg_bias": bias(
                    ae
                ),
                "total_xg_mae": mae(
                    te
                ),
                "total_xg_rmse": rmse(
                    te
                ),
                "total_xg_bias": bias(
                    te
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def process():
    df = read_master()

    if df.empty:
        return

    df[
        "league"
    ] = (
        df["league"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    metrics = (
        build_model_metrics(
            df
        )
    )

    calibration = (
        build_calibration(
            df
        )
    )

    xg = build_xg_metrics(
        df
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_FILE,
        index=False,
    )

    xg.to_csv(
        XG_FILE,
        index=False,
    )

    log_summary(
        f"MODEL METRICS WRITTEN | "
        f"rows={len(metrics)} | "
        f"{METRICS_FILE}"
    )

    log_summary(
        f"CALIBRATION WRITTEN | "
        f"rows={len(calibration)} | "
        f"{CALIBRATION_FILE}"
    )

    log_summary(
        f"XG METRICS WRITTEN | "
        f"rows={len(xg)} | "
        f"{XG_FILE}"
    )

    overall = metrics[
        metrics[
            "scope"
        ]
        == "all"
    ]

    for _, r in overall.iterrows():
        log_summary(
            f"OVERALL MODEL | "
            f"model={r['model']} "
            f"n={int(r['sample_count'])} "
            f"brier={r['brier_score']} "
            f"log_loss={r['log_loss']} "
            f"rps={r['rps']}"
        )


def main():
    reset_logs()

    log_summary(
        f"=== START "
        f"05_soccer_model_reports.py "
        f"{datetime.now().isoformat()} ==="
    )

    try:
        process()

        log_summary(
            f"=== END "
            f"05_soccer_model_reports.py "
            f"{datetime.now().isoformat()} ==="
        )

        print(
            "Soccer model reports generated."
        )

    except Exception as e:
        log_error(
            f"FATAL | {e}\n"
            f"{traceback.format_exc()}"
        )
        raise


if __name__ == "__main__":
    main()