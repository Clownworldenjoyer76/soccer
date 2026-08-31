#!/usr/bin/env python3
# docs/win/soccer/scripts/01_merge/merge_intake.py
#
# PRODUCTION PRICING PATH
# -----------------------
# This stage only merges normalized prediction inputs with sportsbook prices.
# It does not calculate production model probabilities or fair odds.
# Raw home_prob/draw_prob/away_prob values are preserved as upstream model
# inputs/audit fields. The authoritative production pricing calculation occurs
# exactly once in scripts/02_juice/apply_juice.py from validated xG inputs.
# scripts/03_edges/build_edges.py then consumes that engine output only.

import csv
import traceback
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

PRED_DIR = Path(
    "docs/win/soccer/00_intake/predictions/normalized"
)
BOOK_DIR = Path(
    "docs/win/soccer/00_intake/sportsbook/normalized"
)
OUT_DIR = Path(
    "docs/win/soccer/01_merge"
)
LOG_DIR = Path(
    "docs/win/soccer/errors/01_merge"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = (
    LOG_DIR
    / "merge_intake.txt"
)

LEAGUES = [
    "epl",
    "laliga",
    "ligue1",
    "bundesliga",
    "seriea",
    "mls",
]

IDENTITY_REASONS = {
    "missing_sportsbook_game_id",
    "date_mismatch",
    "team_name_mismatch",
    "no_sportsbook_fixture",
}


def log(msg):
    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{msg}\n"
        )


def load_csv(path):
    rows = []

    if not path.exists():
        log(
            f"MISSING FILE: {path}"
        )
        return rows

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    return rows


def build_game_id_index(
    rows,
    label,
):
    idx = {}
    blank_ids = 0
    duplicate_ids = 0

    for row in rows:
        gid = (
            row.get("game_id")
            or ""
        ).strip()

        if not gid:
            blank_ids += 1
            continue

        if gid in idx:
            duplicate_ids += 1

            log(
                f"DUPLICATE "
                f"{label.upper()} "
                f"GAME_ID | "
                f"game_id={gid} | "
                f"existing="
                f"{idx[gid].get('home_team')} "
                f"vs "
                f"{idx[gid].get('away_team')} | "
                f"duplicate="
                f"{row.get('home_team')} "
                f"vs "
                f"{row.get('away_team')}"
            )

            continue

        idx[gid] = row

    return (
        idx,
        blank_ids,
        duplicate_ids,
    )


def prediction_label(row):
    return (
        f"{row.get('home_team', '')} "
        f"vs "
        f"{row.get('away_team', '')}"
    )


def clear_slate_outputs(
    date,
    league,
):
    for suffix in (
        "match_odds",
        "total_25",
        "total_35",
        "btts",
    ):
        path = (
            OUT_DIR
            / f"{date}_{league}_{suffix}.csv"
        )

        if path.exists():
            path.unlink()

            log(
                f"DELETED STALE OUTPUT: "
                f"{path}"
            )


def write_market_file(
    filename,
    header,
    rows,
    summary,
):
    path = (
        OUT_DIR
        / filename
    )

    if not rows:
        if path.exists():
            path.unlink()

            log(
                f"DELETED STALE OUTPUT: "
                f"{path}"
            )
        else:
            log(
                f"NO ROWS: {filename} — "
                f"no output written"
            )

        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(header)
        writer.writerows(rows)

    log(
        f"WROTE {path} "
        f"({len(rows)} rows)"
    )

    summary["files_written"] += 1


def process_slate(
    date,
    league,
    summary,
):
    try:
        pred_path = (
            PRED_DIR
            / f"{date}_{league}.csv"
        )

        book_path = (
            BOOK_DIR
            / f"{date}_{league}.csv"
        )

        preds = load_csv(
            pred_path
        )

        books = load_csv(
            book_path
        )

        if not preds:
            clear_slate_outputs(
                date,
                league,
            )

            log(
                f"SKIP {date} {league}: "
                f"no normalized predictions"
            )

            summary["skipped"] += 1
            return

        (
            book_idx,
            blank_book_ids,
            duplicate_book_ids,
        ) = build_game_id_index(
            books,
            "sportsbook",
        )

        matched = 0
        excluded_missing_identity = 0
        excluded_game_id_not_in_sportsbook = 0
        excluded_duplicate_prediction_id = 0

        reason_counts = Counter()

        seen_prediction_ids = set()
        matched_book_ids = set()

        match_odds_rows = []
        total_25_rows = []
        total_35_rows = []
        btts_rows = []

        for prediction in preds:
            gid = (
                prediction.get("game_id")
                or ""
            ).strip()

            identity_reason = (
                prediction.get(
                    "identity_reason"
                )
                or ""
            ).strip()

            identity_detail = (
                prediction.get(
                    "identity_detail"
                )
                or ""
            ).strip()

            if not gid:
                reason = (
                    identity_reason
                    or "missing_game_id_unclassified"
                )

                excluded_missing_identity += 1
                reason_counts[reason] += 1

                log(
                    "PREDICTION EXCLUDED | "
                    f"date={date} | "
                    f"league={league} | "
                    f"fixture="
                    f"{prediction_label(prediction)} | "
                    f"reason={reason} | "
                    f"detail={identity_detail}"
                )

                continue

            if gid in seen_prediction_ids:
                excluded_duplicate_prediction_id += 1

                reason_counts[
                    "duplicate_prediction_game_id"
                ] += 1

                log(
                    "PREDICTION EXCLUDED | "
                    f"date={date} | "
                    f"league={league} | "
                    f"game_id={gid} | "
                    f"fixture="
                    f"{prediction_label(prediction)} | "
                    f"reason="
                    f"duplicate_prediction_game_id"
                )

                continue

            seen_prediction_ids.add(
                gid
            )

            sportsbook = book_idx.get(
                gid
            )

            if sportsbook is None:
                excluded_game_id_not_in_sportsbook += 1

                reason_counts[
                    "game_id_not_in_sportsbook_slate"
                ] += 1

                log(
                    "PREDICTION EXCLUDED | "
                    f"date={date} | "
                    f"league={league} | "
                    f"game_id={gid} | "
                    f"fixture="
                    f"{prediction_label(prediction)} | "
                    f"reason="
                    f"game_id_not_in_sportsbook_slate"
                )

                continue

            matched += 1

            matched_book_ids.add(
                gid
            )

            reason_counts[
                "merged"
            ] += 1

            log(
                "PREDICTION MERGED | "
                f"date={date} | "
                f"league={league} | "
                f"game_id={gid} | "
                f"fixture="
                f"{prediction_label(prediction)}"
            )

            base = [
                sportsbook["game_id"],
                sportsbook["sport"],
                sportsbook["league"],
                sportsbook["match_date"],
                sportsbook["match_time"],
                sportsbook["home_team"],
                sportsbook["away_team"],
                prediction["home_prob"],
                prediction["draw_prob"],
                prediction["away_prob"],
                prediction["home_xg"],
                prediction["away_xg"],
                prediction[
                    "expected_total_goals"
                ],
            ]

            match_odds_rows.append(
                base
                + [
                    sportsbook[
                        "dk_home_decimal"
                    ],
                    sportsbook[
                        "dk_draw_decimal"
                    ],
                    sportsbook[
                        "dk_away_decimal"
                    ],
                ]
            )

            total_25_rows.append(
                base
                + [
                    sportsbook[
                        "dk_over25_decimal"
                    ],
                    sportsbook[
                        "dk_under25_decimal"
                    ],
                ]
            )

            total_35_rows.append(
                base
                + [
                    sportsbook[
                        "dk_over35_decimal"
                    ],
                    sportsbook[
                        "dk_under35_decimal"
                    ],
                ]
            )

            btts_rows.append(
                base
                + [
                    sportsbook[
                        "btts_yes"
                    ],
                    sportsbook[
                        "btts_no"
                    ],
                ]
            )

        prediction_rows = len(
            preds
        )

        excluded_total = (
            excluded_missing_identity
            + excluded_game_id_not_in_sportsbook
            + excluded_duplicate_prediction_id
        )

        if (
            matched
            + excluded_total
            != prediction_rows
        ):
            raise RuntimeError(
                f"prediction merge accounting "
                f"failure for "
                f"{date} {league}: "
                f"prediction_rows="
                f"{prediction_rows} | "
                f"matched={matched} | "
                f"excluded={excluded_total}"
            )

        sportsbook_without_prediction = sum(
            1
            for gid in book_idx
            if gid not in matched_book_ids
        )

        slate_status = (
            "SUCCESS WITH EXCLUSIONS"
            if excluded_total
            else "SUCCESS"
        )

        log(
            "SLATE SUMMARY | "
            f"date={date} | "
            f"league={league} | "
            f"status={slate_status} | "
            f"prediction_rows="
            f"{prediction_rows} | "
            f"merged={matched} | "
            f"excluded_total="
            f"{excluded_total} | "
            f"excluded_missing_identity="
            f"{excluded_missing_identity} | "
            f"excluded_game_id_not_in_sportsbook="
            f"{excluded_game_id_not_in_sportsbook} | "
            f"excluded_duplicate_prediction_id="
            f"{excluded_duplicate_prediction_id} | "
            f"no_sportsbook_fixture="
            f"{reason_counts['no_sportsbook_fixture']} | "
            f"team_name_mismatch="
            f"{reason_counts['team_name_mismatch']} | "
            f"date_mismatch="
            f"{reason_counts['date_mismatch']} | "
            f"missing_sportsbook_game_id="
            f"{reason_counts['missing_sportsbook_game_id']} | "
            f"sportsbook_rows="
            f"{len(books)} | "
            f"sportsbook_blank_game_id="
            f"{blank_book_ids} | "
            f"sportsbook_duplicate_game_id="
            f"{duplicate_book_ids} | "
            f"sportsbook_without_prediction="
            f"{sportsbook_without_prediction}"
        )

        summary[
            "total_prediction_rows"
        ] += prediction_rows

        summary[
            "total_matched"
        ] += matched

        summary[
            "total_excluded"
        ] += excluded_total

        summary[
            "total_missing_identity"
        ] += excluded_missing_identity

        summary[
            "total_game_id_not_in_sportsbook"
        ] += (
            excluded_game_id_not_in_sportsbook
        )

        summary[
            "total_duplicate_prediction_id"
        ] += (
            excluded_duplicate_prediction_id
        )

        summary[
            "sportsbook_without_prediction"
        ] += sportsbook_without_prediction

        summary[
            "reason_counts"
        ].update(
            reason_counts
        )

        base_header = [
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

        write_market_file(
            f"{date}_{league}_match_odds.csv",
            base_header
            + [
                "dk_home_decimal",
                "dk_draw_decimal",
                "dk_away_decimal",
            ],
            match_odds_rows,
            summary,
        )

        write_market_file(
            f"{date}_{league}_total_25.csv",
            base_header
            + [
                "dk_over25_decimal",
                "dk_under25_decimal",
            ],
            total_25_rows,
            summary,
        )

        write_market_file(
            f"{date}_{league}_total_35.csv",
            base_header
            + [
                "dk_over35_decimal",
                "dk_under35_decimal",
            ],
            total_35_rows,
            summary,
        )

        write_market_file(
            f"{date}_{league}_btts.csv",
            base_header
            + [
                "btts_yes",
                "btts_no",
            ],
            btts_rows,
            summary,
        )

    except Exception as e:
        log(
            f"ERROR {date} {league}: "
            f"{e}\n"
            f"{traceback.format_exc()}"
        )

        summary["errors"] += 1


def main():
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== merge_intake RUN "
            f"{datetime.now(timezone.utc).isoformat()} ===\n"
        )

    summary = {
        "slates_processed": 0,
        "skipped": 0,
        "files_written": 0,
        "total_prediction_rows": 0,
        "total_matched": 0,
        "total_excluded": 0,
        "total_missing_identity": 0,
        "total_game_id_not_in_sportsbook": 0,
        "total_duplicate_prediction_id": 0,
        "sportsbook_without_prediction": 0,
        "reason_counts": Counter(),
        "errors": 0,
    }

    try:
        for pred_file in sorted(
            PRED_DIR.glob("*.csv")
        ):
            stem = pred_file.stem
            league = None
            date = None

            for candidate_league in LEAGUES:
                if stem.endswith(
                    f"_{candidate_league}"
                ):
                    league = candidate_league

                    date = stem[
                        : -(
                            len(
                                candidate_league
                            )
                            + 1
                        )
                    ]

                    break

            if not league or not date:
                log(
                    f"SKIP unrecognized file: "
                    f"{pred_file.name}"
                )

                continue

            summary[
                "slates_processed"
            ] += 1

            process_slate(
                date,
                league,
                summary,
            )

        if (
            summary["total_matched"]
            + summary["total_excluded"]
            != summary[
                "total_prediction_rows"
            ]
        ):
            raise RuntimeError(
                "global merge accounting "
                "failure | "
                f"prediction_rows="
                f"{summary['total_prediction_rows']} | "
                f"matched="
                f"{summary['total_matched']} | "
                f"excluded="
                f"{summary['total_excluded']}"
            )

        log(
            "SUMMARY | "
            f"slates_processed="
            f"{summary['slates_processed']} | "
            f"skipped="
            f"{summary['skipped']} | "
            f"files_written="
            f"{summary['files_written']} | "
            f"prediction_rows="
            f"{summary['total_prediction_rows']} | "
            f"merged="
            f"{summary['total_matched']} | "
            f"excluded="
            f"{summary['total_excluded']} | "
            f"missing_identity="
            f"{summary['total_missing_identity']} | "
            f"game_id_not_in_sportsbook="
            f"{summary['total_game_id_not_in_sportsbook']} | "
            f"duplicate_prediction_game_id="
            f"{summary['total_duplicate_prediction_id']} | "
            f"sportsbook_without_prediction="
            f"{summary['sportsbook_without_prediction']} | "
            f"errors="
            f"{summary['errors']}"
        )

        log(
            "EXCLUSION REASONS | "
            + " | ".join(
                f"{reason}="
                f"{summary['reason_counts'][reason]}"
                for reason in sorted(
                    IDENTITY_REASONS
                    | {
                        "game_id_not_in_sportsbook_slate",
                        "duplicate_prediction_game_id",
                        "missing_game_id_unclassified",
                    }
                )
            )
        )

        if summary["errors"]:
            status = (
                "COMPLETED WITH ERRORS"
            )

        elif summary["total_excluded"]:
            status = (
                "SUCCESS WITH EXCLUSIONS"
            )

        else:
            status = "SUCCESS"

        log(
            f"STATUS: {status}"
        )

    except Exception as e:
        log(
            f"FATAL: {e}\n"
            f"{traceback.format_exc()}"
        )

        log(
            "STATUS: FAILED"
        )

        raise


if __name__ == "__main__":
    main()
