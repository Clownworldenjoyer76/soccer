#!/usr/bin/env python3
"""Shared parsing helpers for manual soccer intake scripts."""

from pathlib import Path
import re


DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
TIME_RE = re.compile(
    r"^\d{1,2}:\d{2}\s*(AM|PM)$",
    re.IGNORECASE,
)
RECORD_RE = re.compile(r"\s*\([^)]*\)\s*$")


def clean_market_for_path(value: str) -> str:
    return (
        (value or "")
        .strip()
        .replace("_", "")
        .replace(" ", "")
        .upper()
    )


def clean_market_for_league_value(value: str) -> str:
    return (
        (value or "")
        .strip()
        .replace("_", "")
        .replace(" ", "")
        .lower()
    )


def normalize_match_date(value: str) -> str:
    value = (value or "").strip()

    if not DATE_RE.match(value):
        raise ValueError(
            f"Invalid match date format: {value}"
        )

    month, day, year = value.split("/")

    return (
        f"{year}_"
        f"{month.zfill(2)}_"
        f"{day.zfill(2)}"
    )


def normalize_match_time(value: str) -> str:
    value = (value or "").strip().upper()

    if not TIME_RE.match(value):
        raise ValueError(
            f"Invalid match time format: {value}"
        )

    time_part, ampm = value.split()
    hour, minute = time_part.split(":")

    return f"{hour.zfill(2)}:{minute} {ampm}"


def clean_team(value: str) -> str:
    value = (value or "").strip()
    value = RECORD_RE.sub("", value)
    return value.strip()


def split_tabs(value: str) -> list[str]:
    return [
        part.strip()
        for part in (value or "").split("\t")
        if part.strip()
    ]


def read_raw_lines(raw_file: Path) -> list[str]:
    text = raw_file.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return text.splitlines()


def clean_raw_lines(
    raw_lines: list[str],
    ignore_lines: set[str],
) -> list[str]:
    cleaned = []

    for line in raw_lines:
        line = line.strip()

        if not line:
            continue

        if line in ignore_lines:
            continue

        cleaned.append(line)

    return cleaned


def split_into_blocks(
    lines: list[str],
) -> list[list[str]]:
    blocks = []
    current_block = []

    for line in lines:
        if DATE_RE.match(line):
            if current_block:
                blocks.append(current_block)

            current_block = [line]

        elif current_block:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    return blocks


def group_rows_by_match_date(
    rows: list[dict],
) -> dict[str, list[dict]]:
    grouped = {}

    for row in rows:
        match_date = row["match_date"]
        grouped.setdefault(
            match_date,
            [],
        ).append(row)

    return grouped


def prepare_manual_input(
    market: str,
    raw_file_value: str,
) -> tuple[str, str, list[str]]:
    raw_file = Path(raw_file_value)

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw file not found: {raw_file}"
        )

    market_path_value = clean_market_for_path(
        market
    )

    league_value = clean_market_for_league_value(
        market
    )

    if not market_path_value:
        raise ValueError(
            "Market value is empty after cleanup"
        )

    if not league_value:
        raise ValueError(
            "League value is empty after cleanup"
        )

    raw_lines = read_raw_lines(raw_file)

    return (
        market_path_value,
        league_value,
        raw_lines,
    )
