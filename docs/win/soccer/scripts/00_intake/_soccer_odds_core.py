#!/usr/bin/env python3
""" soccer odds intake.

Fetches soccer fixtures and odds for:
  - Bundesliga
  - La Liga
  - EPL
  - MLS
  - Serie A

For each requested calendar day and league, writes:
  docs/win/soccer/00_intake/sportsbook/{league}/
      YYYY_MM_DD_{league}_soccer.csv

Odds strategy (field-by-field):
  1. DraftKings, when ESPN exposes it.
  2. Other ESPN-carried providers as backfill.
  3. Leave the field blank if ESPN does not expose a trustworthy value.

No API key is required. ESPN's public endpoints are undocumented and may change,
so parsing is deliberately defensive and failures are isolated per endpoint/event.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
CORE_BASE = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"
CDN_GAME = "https://cdn.espn.com/core/soccer/game"

DEFAULT_OUTPUT_ROOT = Path("docs/win/soccer/00_intake/sportsbook")
DEFAULT_TIMEZONE = "America/New_York"

LEAGUES: dict[str, str] = {
    "bundesliga": "ger.1",
    "la_liga": "esp.1",
    "epl": "eng.1",
    "mls": "usa.1",
    "serie_a": "ita.1",
}

LEAGUE_ALIASES: dict[str, str] = {
    "bundesliga": "bundesliga",
    "ger.1": "bundesliga",
    "germany": "bundesliga",
    "la_liga": "la_liga",
    "laliga": "la_liga",
    "la-liga": "la_liga",
    "esp.1": "la_liga",
    "epl": "epl",
    "premier_league": "epl",
    "premier-league": "epl",
    "eng.1": "epl",
    "mls": "mls",
    "usa.1": "mls",
    "serie_a": "serie_a",
    "seriea": "serie_a",
    "serie-a": "serie_a",
    "ita.1": "serie_a",
}

CSV_FIELDS = [
    "sport",
    "league",
    "game_id",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
    "dk_home_decimal",
    "dk_draw_decimal",
    "dk_away_decimal",
    "dk_over25_decimal",
    "dk_under25_decimal",
    "dk_over35_decimal",
    "dk_under35_decimal",
    "btts_yes",
    "btts_no",
]

PROVIDER_PREFERENCE = [
    "draftkings",
    "draft kings",
    "espn bet",
    "bet365",
    "fan duel",
    "fanduel",
    "betmgm",
    "bet mgm",
    "caesars",
    "tab",
]

USER_AGENT = (
    "Mozilla/5.0 (compatible; SoccerOddsIntake/1.0; "
    "+https://www.espn.com/)"
)


@dataclass
class Offer:
    """A candidate value for one output field."""

    sort_key: tuple[int, int, int] = field(init=False, repr=False)
    value: float
    provider: str
    provider_priority: int = 9999
    path_priority: int = 9999
    source: str = ""

    def __post_init__(self) -> None:
        self.sort_key = (
            provider_rank(self.provider),
            safe_int(self.provider_priority, 9999),
            safe_int(self.path_priority, 9999),
        )


@dataclass
class EventOdds:
    offers: dict[str, list[Offer]] = field(
        default_factory=lambda: {
            "home": [],
            "draw": [],
            "away": [],
            "over25": [],
            "under25": [],
            "over35": [],
            "under35": [],
            "btts_yes": [],
            "btts_no": [],
        }
    )

    def add(
        self,
        field_name: str,
        value: Any,
        provider: str,
        provider_priority: Any,
        path_priority: int,
        source: str,
        *,
        american: bool = True,
    ) -> None:
        if field_name not in self.offers:
            return

        decimal_value = (
            odds_to_decimal(value)
            if american
            else coerce_decimal_odds(value)
        )

        if decimal_value is None:
            return

        self.offers[field_name].append(
            Offer(
                value=decimal_value,
                provider=provider or "unknown",
                provider_priority=safe_int(provider_priority, 9999),
                path_priority=path_priority,
                source=source,
            )
        )

    def best(self, field_name: str) -> Offer | None:
        candidates = self.offers.get(field_name, [])
        return min(
            candidates,
            key=lambda offer: offer.sort_key,
        ) if candidates else None


class ESPNClient:
    def __init__(
        self,
        *,
        timeout: float = 20.0,
        retries: int = 3,
        min_interval: float = 0.20,
        verbose: bool = False,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.min_interval = max(0.0, min_interval)
        self.verbose = verbose
        self._last_request = 0.0
        self._cache: dict[str, Any] = {}

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        use_cache: bool = True,
    ) -> Any:
        if params:
            query = urlencode(
                {
                    key: value
                    for key, value in params.items()
                    if value is not None
                },
                doseq=True,
            )
            full_url = f"{url}?{query}"
        else:
            full_url = url

        full_url = normalize_espn_ref(full_url)

        if use_cache and full_url in self._cache:
            return self._cache[full_url]

        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            elapsed = time.monotonic() - self._last_request

            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            request = Request(
                full_url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json,text/plain,*/*",
                    "Cache-Control": "no-cache",
                },
            )

            try:
                self.log(f"GET {full_url}")
                self._last_request = time.monotonic()

                with urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:
                    raw = response.read()

                payload = json.loads(raw.decode("utf-8"))

                if use_cache:
                    self._cache[full_url] = payload

                return payload

            except HTTPError as exc:
                last_error = exc

                if exc.code in {400, 404}:
                    break

            except (
                URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc

            except Exception as exc:
                last_error = exc

            if attempt < self.retries:
                time.sleep(min(2 ** (attempt - 1), 4))

        if self.verbose and last_error is not None:
            self.log(
                f"  failed: {type(last_error).__name__}: {last_error}"
            )

        return None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, str):
        text = value.strip().replace(",", "")

        if not text:
            return None

        match = re.fullmatch(
            r"[+\-]?\d+(?:\.\d+)?",
            text,
        )

        if match:
            try:
                number = float(text)
                return number if math.isfinite(number) else None
            except ValueError:
                return None

    return None


def odds_to_decimal(value: Any) -> float | None:
    """Convert ESPN-style American odds to decimal odds."""
    number = coerce_float(value)

    if number is None or number == 0:
        return None

    if 1.01 <= number < 100:
        return number

    if number >= 100:
        return 1.0 + number / 100.0

    if number <= -100:
        return 1.0 + 100.0 / abs(number)

    return None


def coerce_decimal_odds(value: Any) -> float | None:
    number = coerce_float(value)

    if number is None:
        return None

    if 1.0 < number < 1000:
        return number

    return odds_to_decimal(number)


def format_decimal(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip().lower()


def normalize_provider_name(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        normalize_text(value),
    ).strip()


def provider_rank(provider: str) -> int:
    normalized = normalize_provider_name(provider)

    for idx, preferred in enumerate(PROVIDER_PREFERENCE):
        if preferred in normalized:
            return idx

    return len(PROVIDER_PREFERENCE) + 100


def normalize_espn_ref(url: str) -> str:
    """Make Core API references public/reachable where possible."""
    url = url.replace(
        "sports.core.api.espn.pvt",
        "sports.core.api.espn.com",
    )

    parsed = urlparse(url)

    if parsed.scheme == "http":
        parsed = parsed._replace(scheme="https")
        url = urlunparse(parsed)

    return url


def walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    """Yield every dictionary in an arbitrary JSON tree."""
    stack = [value]
    seen_ids: set[int] = set()

    while stack:
        current = stack.pop()

        if id(current) in seen_ids:
            continue

        seen_ids.add(id(current))

        if isinstance(current, dict):
            yield current
            stack.extend(current.values())

        elif isinstance(current, list):
            stack.extend(current)


def first_nonempty(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        value = mapping.get(key)

        if value not in (
            None,
            "",
            [],
            {},
        ):
            return value

    return None


def provider_from_dict(
    obj: Mapping[str, Any],
) -> tuple[str, int]:
    provider = obj.get("provider")

    if isinstance(provider, dict):
        name = first_nonempty(
            provider,
            (
                "name",
                "displayName",
                "shortName",
                "abbreviation",
                "slug",
            ),
        )

        priority = first_nonempty(
            provider,
            (
                "priority",
                "rank",
            ),
        )

        return (
            str(name or "unknown"),
            safe_int(priority, 9999),
        )

    if isinstance(provider, str):
        return provider, 9999

    book = first_nonempty(
        obj,
        (
            "sportsbook",
            "bookmaker",
            "book",
            "providerName",
            "sportsbookName",
        ),
    )

    if isinstance(book, dict):
        name = first_nonempty(
            book,
            (
                "name",
                "displayName",
                "shortName",
            ),
        )

        return str(name or "unknown"), 9999

    return str(book or "unknown"), 9999


def json_text(
    obj: Mapping[str, Any],
    *,
    max_depth: int = 2,
) -> str:
    """Build searchable text from human-facing fields."""
    text_keys = {
        "name",
        "displayname",
        "shortname",
        "label",
        "description",
        "details",
        "title",
        "text",
        "marketname",
        "market",
        "selection",
        "outcome",
        "betname",
        "bettype",
        "type",
    }

    pieces: list[str] = []

    def visit(value: Any, depth: int) -> None:
        if depth > max_depth:
            return

        if isinstance(value, dict):
            for key, child in value.items():
                key_norm = normalize_text(key).replace("_", "")

                if (
                    key_norm in text_keys
                    and isinstance(child, (str, int, float))
                ):
                    pieces.append(str(child))

                elif (
                    depth < max_depth
                    and isinstance(child, (dict, list))
                ):
                    visit(child, depth + 1)

        elif isinstance(value, list):
            for child in value[:30]:
                visit(child, depth + 1)

    visit(obj, 0)

    return normalize_text(
        " | ".join(pieces)
    )


def extract_price(
    obj: Mapping[str, Any],
) -> Any:
    """Extract the most likely American/decimal odds value."""
    for key in (
        "moneyLine",
        "moneyline",
        "americanOdds",
        "american",
        "price",
        "odds",
    ):
        if key not in obj:
            continue

        value = obj.get(key)

        if isinstance(value, dict):
            nested = first_nonempty(
                value,
                (
                    "moneyLine",
                    "moneyline",
                    "americanOdds",
                    "american",
                    "price",
                    "odds",
                    "value",
                ),
            )

            if coerce_float(nested) is not None:
                return nested

        elif coerce_float(value) is not None:
            return value

    return None


def extract_espn_current_price(
    obj: Mapping[str, Any],
) -> tuple[Any, bool]:
    """Extract price from ESPN Core prop rows such as current.over.

    Returns:
        (value, american)

    ESPN exposes both decimal and American representations. Decimal is preferred
    because it avoids unnecessary conversion.
    """
    current = obj.get("current")

    if not isinstance(current, dict):
        return None, False

    for side in (
        "over",
        "under",
    ):
        price_obj = current.get(side)

        if not isinstance(price_obj, dict):
            continue

        decimal = price_obj.get("decimal")

        if coerce_float(decimal) is not None:
            return decimal, False

        american = first_nonempty(
            price_obj,
            (
                "american",
                "alternateDisplayValue",
                "value",
            ),
        )

        if coerce_float(american) is not None:
            return american, True

    return None, False


def parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()

    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(
                text[:-1] + "+00:00"
            )

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt

    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Match-odds extraction
# ---------------------------------------------------------------------------


def looks_like_match_odds(
    obj: Mapping[str, Any],
) -> bool:
    """Identify full-match 1X2/total objects."""
    keys = set(obj.keys())

    if {
        "homeTeamOdds",
        "awayTeamOdds",
    }.issubset(keys):
        return True

    if (
        "drawOdds" in keys
        and (
            "overUnder" in keys
            or "homeTeamOdds" in keys
        )
    ):
        return True

    return False


def extract_moneyline(team_odds: Any) -> Any:
    if not isinstance(team_odds, dict):
        return None

    return first_nonempty(
        team_odds,
        (
            "moneyLine",
            "moneyline",
            "currentMoneyLine",
            "closeMoneyLine",
        ),
    )


def ingest_match_odds_object(
    target: EventOdds,
    obj: Mapping[str, Any],
    *,
    path_priority: int,
    source: str,
) -> None:
    if not looks_like_match_odds(obj):
        return

    provider, provider_priority = provider_from_dict(obj)

    target.add(
        "home",
        extract_moneyline(obj.get("homeTeamOdds")),
        provider,
        provider_priority,
        path_priority,
        source,
    )

    target.add(
        "draw",
        extract_moneyline(obj.get("drawOdds")),
        provider,
        provider_priority,
        path_priority,
        source,
    )

    target.add(
        "away",
        extract_moneyline(obj.get("awayTeamOdds")),
        provider,
        provider_priority,
        path_priority,
        source,
    )

    line = coerce_float(
        first_nonempty(
            obj,
            (
                "overUnder",
                "total",
                "totalLine",
                "line",
            ),
        )
    )

    if line is None:
        return

    over = first_nonempty(
        obj,
        (
            "overOdds",
            "overPrice",
            "overMoneyLine",
        ),
    )

    under = first_nonempty(
        obj,
        (
            "underOdds",
            "underPrice",
            "underMoneyLine",
        ),
    )

    if over is None and isinstance(obj.get("over"), dict):
        over = extract_price(obj["over"])

    if under is None and isinstance(obj.get("under"), dict):
        under = extract_price(obj["under"])

    if abs(line - 2.5) < 1e-9:
        target.add(
            "over25",
            over,
            provider,
            provider_priority,
            path_priority,
            source,
        )

        target.add(
            "under25",
            under,
            provider,
            provider_priority,
            path_priority,
            source,
        )

    elif abs(line - 3.5) < 1e-9:
        target.add(
            "over35",
            over,
            provider,
            provider_priority,
            path_priority,
            source,
        )

        target.add(
            "under35",
            under,
            provider,
            provider_priority,
            path_priority,
            source,
        )


def ingest_all_match_odds_objects(
    target: EventOdds,
    payload: Any,
    *,
    path_priority: int,
    source: str,
) -> None:
    for obj in walk_dicts(payload):
        ingest_match_odds_object(
            target,
            obj,
            path_priority=path_priority,
            source=source,
        )


# ---------------------------------------------------------------------------
# Prop-bet extraction
# ---------------------------------------------------------------------------

BTTS_MARKERS = (
    "both teams to score",
    "both teams score",
    "both team to score",
    "both team score",
    "btts",
)

EXCLUDED_TOTAL_MARKERS = (
    "team total",
    "total team",
    "team goals",
    "player",
    "corners",
    "corner",
    "cards",
    "bookings",
    "shots",
    "first half",
    "1st half",
    "1h",
    "second half",
    "2nd half",
    "2h",
)

FULL_MATCH_TOTAL_MARKERS = (
    "total goals",
    "match total",
    "game total",
    "over under",
    "over/under",
)


def text_contains_any(
    text: str,
    markers: Iterable[str],
) -> bool:
    return any(
        marker in text
        for marker in markers
    )


def infer_yes_no(text: str) -> str | None:
    tokens = set(
        re.findall(
            r"[a-z]+",
            text,
        )
    )

    if "yes" in tokens:
        return "yes"

    if "no" in tokens:
        return "no"

    return None


def infer_over_under(text: str) -> str | None:
    tokens = set(
        re.findall(
            r"[a-z]+",
            text,
        )
    )

    if "over" in tokens:
        return "over"

    if "under" in tokens:
        return "under"

    return None


def infer_total_line(text: str) -> float | None:
    if re.search(
        r"(?<!\d)2(?:\.5|½)(?!\d)",
        text,
    ):
        return 2.5

    if re.search(
        r"(?<!\d)3(?:\.5|½)(?!\d)",
        text,
    ):
        return 3.5

    return None


def iter_outcome_dicts(
    obj: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    for key in (
        "outcomes",
        "selections",
        "options",
        "prices",
        "choices",
        "bets",
    ):
        value = obj.get(key)

        if isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    yield child


def ingest_espn_core_btts_rows(
    target: EventOdds,
    payload: Any,
    *,
    path_priority: int,
    source: str,
) -> None:
    """Parse ESPN Core's special BTTS representation.

    ESPN provider propBets currently represents "Both Teams To Score" as two
    consecutive items with:

        type.id   == "152"
        type.name == "Both Teams To Score"

    The first row is YES and the second row is NO. The selection price is
    contained in current.over rather than in a named yes/no child outcome.

    Example:
        row 1: current.over.decimal = 1.36  -> YES
        row 2: current.over.decimal = 3.10  -> NO
    """
    if not isinstance(payload, dict):
        return

    items = payload.get("items")

    if not isinstance(items, list):
        return

    grouped: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        type_obj = item.get("type")

        if not isinstance(type_obj, dict):
            continue

        type_id = str(
            type_obj.get("id") or ""
        ).strip()

        type_name = normalize_text(
            type_obj.get("name")
        )

        if (
            type_id != "152"
            and not text_contains_any(
                type_name,
                BTTS_MARKERS,
            )
        ):
            continue

        provider_obj = item.get("provider")
        competition_obj = item.get("competition")

        provider_ref = ""

        if isinstance(provider_obj, dict):
            provider_ref = str(
                provider_obj.get("$ref") or ""
            )

        competition_ref = ""

        if isinstance(competition_obj, dict):
            competition_ref = str(
                competition_obj.get("$ref") or ""
            )

        group_key = (
            provider_ref,
            competition_ref,
        )

        grouped.setdefault(
            group_key,
            [],
        ).append(item)

    for rows in grouped.values():
        if not rows:
            continue

        # ESPN Core currently emits the BTTS selections in YES, NO order.
        for index, item in enumerate(rows[:2]):
            field_name = (
                "btts_yes"
                if index == 0
                else "btts_no"
            )

            provider, provider_priority = provider_from_dict(item)

            price, american = extract_espn_current_price(
                item
            )

            if price is None:
                continue

            target.add(
                field_name,
                price,
                provider,
                provider_priority,
                path_priority,
                source,
                american=american,
            )


def ingest_propbet_object(
    target: EventOdds,
    obj: Mapping[str, Any],
    *,
    path_priority: int,
    source: str,
) -> None:
    market_text = json_text(
        obj,
        max_depth=0,
    )

    broad_text = json_text(
        obj,
        max_depth=2,
    )

    provider, provider_priority = provider_from_dict(obj)

    children = list(
        iter_outcome_dicts(obj)
    )

    if children:
        if text_contains_any(
            market_text,
            BTTS_MARKERS,
        ):
            for child in children:
                child_text = (
                    f"{market_text} | "
                    f"{json_text(child)}"
                )

                side = infer_yes_no(
                    child_text
                )

                if side:
                    child_provider, child_priority = provider_from_dict(
                        child
                    )

                    target.add(
                        f"btts_{side}",
                        extract_price(child),
                        (
                            child_provider
                            if child_provider != "unknown"
                            else provider
                        ),
                        (
                            child_priority
                            if child_priority != 9999
                            else provider_priority
                        ),
                        path_priority,
                        source,
                    )

        if (
            text_contains_any(
                market_text,
                FULL_MATCH_TOTAL_MARKERS,
            )
            and not text_contains_any(
                market_text,
                EXCLUDED_TOTAL_MARKERS,
            )
        ):
            parent_line = infer_total_line(
                market_text
            )

            for child in children:
                child_text = (
                    f"{market_text} | "
                    f"{json_text(child)}"
                )

                if text_contains_any(
                    child_text,
                    EXCLUDED_TOTAL_MARKERS,
                ):
                    continue

                line = (
                    infer_total_line(child_text)
                    or parent_line
                )

                side = infer_over_under(
                    child_text
                )

                if (
                    line in {2.5, 3.5}
                    and side
                ):
                    field_name = (
                        f"{side}"
                        f"{'25' if line == 2.5 else '35'}"
                    )

                    child_provider, child_priority = provider_from_dict(
                        child
                    )

                    target.add(
                        field_name,
                        extract_price(child),
                        (
                            child_provider
                            if child_provider != "unknown"
                            else provider
                        ),
                        (
                            child_priority
                            if child_priority != 9999
                            else provider_priority
                        ),
                        path_priority,
                        source,
                    )

    own_price = extract_price(obj)

    if own_price is None:
        return

    if text_contains_any(
        broad_text,
        BTTS_MARKERS,
    ):
        side = infer_yes_no(
            broad_text
        )

        if side:
            target.add(
                f"btts_{side}",
                own_price,
                provider,
                provider_priority,
                path_priority,
                source,
            )

    if (
        text_contains_any(
            broad_text,
            FULL_MATCH_TOTAL_MARKERS,
        )
        and not text_contains_any(
            broad_text,
            EXCLUDED_TOTAL_MARKERS,
        )
    ):
        line = infer_total_line(
            broad_text
        )

        side = infer_over_under(
            broad_text
        )

        if (
            line in {2.5, 3.5}
            and side
        ):
            field_name = (
                f"{side}"
                f"{'25' if line == 2.5 else '35'}"
            )

            target.add(
                field_name,
                own_price,
                provider,
                provider_priority,
                path_priority,
                source,
            )


def ingest_propbets(
    target: EventOdds,
    payload: Any,
    *,
    path_priority: int,
    source: str,
) -> None:
    # ESPN Core represents BTTS differently from many other prop markets:
    # two type=152 rows, ordered YES then NO, with price in current.over.
    ingest_espn_core_btts_rows(
        target,
        payload,
        path_priority=path_priority,
        source=source,
    )

    # Retain the generic parser for ESPN variants that explicitly label
    # Yes/No outcomes and for safe full-match total props.
    for obj in walk_dicts(payload):
        ingest_propbet_object(
            target,
            obj,
            path_priority=path_priority,
            source=source,
        )


# ---------------------------------------------------------------------------
# ESPN event fetching
# ---------------------------------------------------------------------------


def dereference_core_items(
    client: ESPNClient,
    payload: Any,
) -> list[Any]:
    if not isinstance(payload, dict):
        return []

    items = payload.get("items")

    if not isinstance(items, list):
        return []

    resolved: list[Any] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        ref = item.get("$ref")

        if (
            isinstance(ref, str)
            and not item.get("provider")
        ):
            dereferenced = client.get_json(
                normalize_espn_ref(ref)
            )

            if dereferenced is not None:
                resolved.append(
                    dereferenced
                )
                continue

        resolved.append(item)

    return resolved


def fetch_event_odds(
    client: ESPNClient,
    league_slug: str,
    event_id: str,
    competition_id: str,
    scoreboard_event: Mapping[str, Any],
) -> EventOdds:
    result = EventOdds()

    ingest_all_match_odds_objects(
        result,
        scoreboard_event,
        path_priority=0,
        source="scoreboard",
    )

    core_url = (
        f"{CORE_BASE}/{league_slug}/events/{event_id}/"
        f"competitions/{competition_id}/odds"
    )

    core_payload = client.get_json(
        core_url,
        params={"limit": 100},
    )

    if core_payload is not None:
        ingest_all_match_odds_objects(
            result,
            core_payload,
            path_priority=1,
            source="core_odds",
        )

        for item in dereference_core_items(
            client,
            core_payload,
        ):
            ingest_all_match_odds_objects(
                result,
                item,
                path_priority=1,
                source="core_odds_ref",
            )

    summary_payload = client.get_json(
        f"{SITE_BASE}/{league_slug}/summary",
        params={"event": event_id},
    )

    if summary_payload is not None:
        ingest_all_match_odds_objects(
            result,
            summary_payload,
            path_priority=2,
            source="summary",
        )

        ingest_propbets(
            result,
            summary_payload,
            path_priority=4,
            source="summary_props",
        )

    cdn_payload = client.get_json(
        CDN_GAME,
        params={
            "xhr": 1,
            "gameId": event_id,
        },
    )

    if cdn_payload is not None:
        ingest_all_match_odds_objects(
            result,
            cdn_payload,
            path_priority=3,
            source="cdn_game",
        )

        ingest_propbets(
            result,
            cdn_payload,
            path_priority=5,
            source="cdn_props",
        )

    prop_url = (
        f"{CORE_BASE}/{league_slug}/events/{event_id}/"
        f"competitions/{competition_id}/propbets"
    )

    prop_payload = client.get_json(
        prop_url,
        params={"limit": 2000},
    )

    if prop_payload is not None:
        ingest_propbets(
            result,
            prop_payload,
            path_priority=3,
            source="core_propbets",
        )

        for item in dereference_core_items(
            client,
            prop_payload,
        ):
            ingest_propbets(
                result,
                item,
                path_priority=3,
                source="core_propbets_ref",
            )

    return result


def event_teams(
    event: Mapping[str, Any],
) -> tuple[str, str]:
    competitions = event.get("competitions")

    if (
        not isinstance(competitions, list)
        or not competitions
    ):
        return "", ""

    competition = competitions[0]

    if not isinstance(competition, dict):
        return "", ""

    home = ""
    away = ""

    for competitor in (
        competition.get("competitors")
        or []
    ):
        if not isinstance(competitor, dict):
            continue

        team = competitor.get("team")

        if not isinstance(team, dict):
            team = {}

        name = first_nonempty(
            team,
            (
                "displayName",
                "shortDisplayName",
                "name",
                "location",
                "abbreviation",
            ),
        )

        side = normalize_text(
            competitor.get("homeAway")
        )

        if side == "home":
            home = str(name or "")

        elif side == "away":
            away = str(name or "")

    return home, away


def event_datetime(
    event: Mapping[str, Any],
) -> datetime | None:
    return parse_iso_datetime(
        event.get("date")
    )


def event_competition_id(
    event: Mapping[str, Any],
) -> str:
    competitions = event.get("competitions")

    if (
        isinstance(competitions, list)
        and competitions
    ):
        first = competitions[0]

        if (
            isinstance(first, dict)
            and first.get("id") is not None
        ):
            return str(
                first["id"]
            )

    return str(
        event.get("id") or ""
    )


def row_for_event(
    client: ESPNClient,
    *,
    league_name: str,
    league_slug: str,
    requested_date: date,
    target_tz: ZoneInfo,
    event: Mapping[str, Any],
) -> dict[str, str] | None:
    event_id = str(
        event.get("id") or ""
    ).strip()

    if not event_id:
        return None

    kickoff = event_datetime(event)

    if kickoff is None:
        return None

    local_kickoff = kickoff.astimezone(
        target_tz
    )

    if local_kickoff.date() != requested_date:
        return None

    home_team, away_team = event_teams(event)

    competition_id = (
        event_competition_id(event)
        or event_id
    )

    odds = fetch_event_odds(
        client,
        league_slug,
        event_id,
        competition_id,
        event,
    )

    best = {
        field_name: odds.best(field_name)
        for field_name in odds.offers
    }

    if client.verbose:
        chosen = []

        for field_name, offer in best.items():
            if offer:
                chosen.append(
                    f"{field_name}={offer.value:.2f} "
                    f"[{offer.provider}/{offer.source}]"
                )

        client.log(
            f"  {league_name} {event_id} "
            f"{away_team} @ {home_team}: "
            + (
                ", ".join(chosen)
                if chosen
                else "no odds found"
            )
        )

    return {
        "sport": "soccer",
        "league": league_name,
        "game_id": event_id,
        "match_date": local_kickoff.strftime("%Y_%m_%d"),
        "match_time": local_kickoff.strftime("%I:%M %p"),
        "home_team": home_team,
        "away_team": away_team,
        "dk_home_decimal": format_decimal(
            best["home"].value
            if best["home"]
            else None
        ),
        "dk_draw_decimal": format_decimal(
            best["draw"].value
            if best["draw"]
            else None
        ),
        "dk_away_decimal": format_decimal(
            best["away"].value
            if best["away"]
            else None
        ),
        "dk_over25_decimal": format_decimal(
            best["over25"].value
            if best["over25"]
            else None
        ),
        "dk_under25_decimal": format_decimal(
            best["under25"].value
            if best["under25"]
            else None
        ),
        "dk_over35_decimal": format_decimal(
            best["over35"].value
            if best["over35"]
            else None
        ),
        "dk_under35_decimal": format_decimal(
            best["under35"].value
            if best["under35"]
            else None
        ),
        "btts_yes": format_decimal(
            best["btts_yes"].value
            if best["btts_yes"]
            else None
        ),
        "btts_no": format_decimal(
            best["btts_no"].value
            if best["btts_no"]
            else None
        ),
    }


def fetch_scoreboard(
    client: ESPNClient,
    league_slug: str,
    target_date: date,
) -> list[Mapping[str, Any]]:
    date_candidates = [
        target_date - timedelta(days=1),
        target_date,
        target_date + timedelta(days=1),
    ]

    events_by_id: dict[
        str,
        Mapping[str, Any],
    ] = {}

    for day in date_candidates:
        payload = client.get_json(
            f"{SITE_BASE}/{league_slug}/scoreboard",
            params={
                "dates": day.strftime("%Y%m%d"),
                "limit": 1000,
            },
        )

        if not isinstance(payload, dict):
            continue

        events = payload.get("events")

        if not isinstance(events, list):
            continue

        for event in events:
            if (
                isinstance(event, dict)
                and event.get("id") is not None
            ):
                events_by_id[
                    str(event["id"])
                ] = event

    return list(
        events_by_id.values()
    )


# ---------------------------------------------------------------------------
# Output / CLI
# ---------------------------------------------------------------------------


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def canonical_league(value: str) -> str:
    key = normalize_text(
        value
    ).replace(
        " ",
        "_",
    )

    canonical = LEAGUE_ALIASES.get(
        key
    )

    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"unknown league {value!r}; choose from: "
            f"{', '.join(LEAGUES)}"
        )

    return canonical


def parse_cli_date(value: str) -> date:
    text = (
        value.strip()
        .replace("_", "-")
        .replace("/", "-")
    )

    try:
        return date.fromisoformat(text)

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected "
            "YYYY-MM-DD or YYYY_MM_DD"
        ) from exc


def date_range(
    start: date,
    end: date,
) -> Iterator[date]:
    if end < start:
        raise ValueError(
            "end date must be on or after start date"
        )

    current = start

    while current <= end:
        yield current
        current += timedelta(days=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch ESPN soccer odds and write "
            "one CSV per day per league."
        )
    )

    dates = parser.add_mutually_exclusive_group()

    dates.add_argument(
        "--date",
        type=parse_cli_date,
        help=(
            "single date (YYYY-MM-DD or YYYY_MM_DD); "
            "default: today in --timezone"
        ),
    )

    dates.add_argument(
        "--start-date",
        type=parse_cli_date,
        help=(
            "start of inclusive date range; "
            "use with --end-date"
        ),
    )

    parser.add_argument(
        "--end-date",
        type=parse_cli_date,
        help=(
            "end of inclusive date range "
            "(requires --start-date)"
        ),
    )

    parser.add_argument(
        "--league",
        type=canonical_league,
        action="append",
        help=(
            "league to fetch; repeat for multiple. "
            "Default: all five. Choices: "
            "bundesliga, la_liga, epl, mls, serie_a"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            f"root output directory "
            f"(default: {DEFAULT_OUTPUT_ROOT})"
        ),
    )

    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=(
            "timezone used for match date/time and "
            f"file assignment (default: {DEFAULT_TIMEZONE})"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout seconds",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="HTTP attempts per request",
    )

    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.20,
        help=(
            "minimum seconds between ESPN requests "
            "(default: 0.20)"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "print endpoint failures and chosen "
            "provider/source per field"
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        target_tz = ZoneInfo(
            args.timezone
        )

    except Exception as exc:
        parser.error(
            f"invalid timezone {args.timezone!r}: {exc}"
        )

    if (
        args.start_date
        and not args.end_date
    ):
        parser.error(
            "--start-date requires --end-date"
        )

    if (
        args.end_date
        and not args.start_date
    ):
        parser.error(
            "--end-date requires --start-date"
        )

    if args.date:
        days = [args.date]

    elif args.start_date:
        try:
            days = list(
                date_range(
                    args.start_date,
                    args.end_date,
                )
            )

        except ValueError as exc:
            parser.error(
                str(exc)
            )

    else:
        days = [
            datetime.now(
                target_tz
            ).date()
        ]

    leagues = list(
        dict.fromkeys(
            args.league
            or list(LEAGUES.keys())
        )
    )

    client = ESPNClient(
        timeout=args.timeout,
        retries=max(
            1,
            args.retries,
        ),
        min_interval=max(
            0.0,
            args.min_interval,
        ),
        verbose=args.verbose,
    )

    total_rows = 0
    total_files = 0

    for target_date in days:
        for league_name in leagues:
            league_slug = LEAGUES[
                league_name
            ]

            client.log(
                f"\n=== {league_name} ({league_slug}) "
                f"{target_date.isoformat()} ==="
            )

            events = fetch_scoreboard(
                client,
                league_slug,
                target_date,
            )

            rows: list[
                dict[str, str]
            ] = []

            for event in events:
                try:
                    row = row_for_event(
                        client,
                        league_name=league_name,
                        league_slug=league_slug,
                        requested_date=target_date,
                        target_tz=target_tz,
                        event=event,
                    )

                    if row is not None:
                        rows.append(row)

                except Exception as exc:
                    event_id = (
                        event.get(
                            "id",
                            "unknown",
                        )
                        if isinstance(event, dict)
                        else "unknown"
                    )

                    print(
                        f"warning: {league_name} event "
                        f"{event_id}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

            rows.sort(
                key=lambda row: (
                    row["match_time"],
                    row["home_team"],
                    row["away_team"],
                )
            )

            file_date = target_date.strftime(
                "%Y_%m_%d"
            )

            output_path = (
                args.output_root
                / league_name
                / (
                    f"{file_date}_"
                    f"{league_name}_soccer.csv"
                )
            )

            write_csv(
                output_path,
                rows,
            )

            total_rows += len(rows)
            total_files += 1

            print(
                f"wrote {len(rows):>2} rows -> "
                f"{output_path}"
            )

    print(
        f"done: {total_rows} rows across "
        f"{total_files} CSV file(s)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
