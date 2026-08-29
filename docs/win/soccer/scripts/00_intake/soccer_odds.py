#!/usr/bin/env python3
"""Soccer odds intake using ESPN CDN fixture discovery and Core odds.

The ESPN retrieval/parsing implementation lives in _soccer_odds_core.py. This
wrapper preserves the public script path while adapting endpoint selection for
GitHub-hosted runners:

- Fixtures come from ESPN's CDN scoreboard because site.api.espn.com returns
  HTTP 403 from GitHub Actions.
- Primary markets come from the reachable Core competition /odds endpoint.
- Prop markets follow each odds provider's own propBets.$ref when ESPN exposes
  one, rather than the generic competition /propbets path.
- Provider prop trees are recursively dereferenced so nested markets such as
  Both Teams To Score can be discovered.
- Blocked Site summary and non-JSON CDN game fallbacks are not requested.

The target date plus adjacent dates must each return a valid CDN scoreboard
payload. A legitimate empty events list is accepted, but HTTP/network/malformed
responses are not. This prevents failed discovery from being mistaken for an
empty slate and silently producing a header-only success file.

Daily CSV persistence remains non-destructive: later runs never replace an
already-populated value with a blank and never drop previously captured games
that disappear from a later same-day ESPN response.

Ligue 1 is enabled here as ESPN league ``fra.1`` so it participates in the same
retrieval, provider fallback, persistence, and daily CSV workflow as the other
configured soccer leagues.
"""

from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import _soccer_odds_core as core
from _soccer_odds_core import *  # noqa: F401,F403 - preserve existing imports/API


CDN_SCOREBOARD = "https://cdn.espn.com/core/soccer/scoreboard"

_PROP_TREE_REF_MARKERS = (
    "/propbets",
    "/markets",
    "/market/",
    "/offers",
    "/offer/",
    "/selections",
    "/selection/",
    "/outcomes",
    "/outcome/",
)


# Extend the core league configuration without changing the underlying parsers.
# main() reads these dictionaries at runtime, so Ligue 1 is included by default
# and is also selectable explicitly through --league.
core.LEAGUES["ligue_1"] = "fra.1"
core.LEAGUE_ALIASES.update(
    {
        "ligue_1": "ligue_1",
        "ligue1": "ligue_1",
        "ligue-1": "ligue_1",
        "fra.1": "ligue_1",
        "france": "ligue_1",
    }
)


class ScoreboardFetchError(RuntimeError):
    """Raised when ESPN CDN fixture discovery is incomplete or malformed."""


def _cdn_events(payload: Any) -> list[Mapping[str, Any]] | None:
    """Return CDN scoreboard events, or None when the payload shape is invalid."""
    if not isinstance(payload, dict):
        return None

    content = payload.get("content")
    if not isinstance(content, dict):
        return None

    sb_data = content.get("sbData")
    if not isinstance(sb_data, dict):
        return None

    events = sb_data.get("events")
    if not isinstance(events, list):
        return None

    return [event for event in events if isinstance(event, dict)]


def fetch_scoreboard(
    client: core.ESPNClient,
    league_slug: str,
    target_date: date,
) -> list[Mapping[str, Any]]:
    """Fetch a complete three-day fixture window from ESPN's CDN scoreboard.

    The target date plus one day on either side are requested so local
    America/New_York assignment does not lose late MLS fixtures that cross a UTC
    date boundary. Every request must return a valid ``content.sbData.events``
    list. An empty list is a valid empty slate.
    """
    date_candidates = [
        target_date - timedelta(days=1),
        target_date,
        target_date + timedelta(days=1),
    ]

    events_by_id: dict[str, Mapping[str, Any]] = {}
    failed_days: list[str] = []

    for day in date_candidates:
        day_text = day.strftime("%Y%m%d")

        payload = client.get_json(
            CDN_SCOREBOARD,
            params={
                "xhr": 1,
                "league": league_slug,
                "dates": day_text,
            },
        )

        events = _cdn_events(payload)

        if events is None:
            client.log(
                f"  invalid CDN scoreboard payload for {league_slug} "
                f"{day_text}: missing content.sbData.events list"
            )
            failed_days.append(day_text)
            continue

        for event in events:
            event_id = str(event.get("id") or "").strip()
            if event_id:
                events_by_id[event_id] = event

    if failed_days:
        joined = ", ".join(failed_days)
        raise ScoreboardFetchError(
            f"ESPN CDN scoreboard discovery incomplete for {league_slug}; "
            f"failed date request(s): {joined}. No sportsbook CSV was written."
        )

    return list(events_by_id.values())


def _iter_provider_prop_refs(
    payload: Any,
) -> Iterator[tuple[str, str, int]]:
    """Yield unique provider-specific propBets refs with provider metadata."""
    seen: set[str] = set()

    for obj in core.walk_dicts(payload):
        prop_bets = obj.get("propBets")
        if not isinstance(prop_bets, dict):
            continue

        ref = prop_bets.get("$ref")
        if not isinstance(ref, str) or not ref.strip():
            continue

        normalized_ref = core.normalize_espn_ref(ref.strip())
        if normalized_ref in seen:
            continue

        seen.add(normalized_ref)

        provider, provider_priority = core.provider_from_dict(obj)
        yield normalized_ref, provider, provider_priority


def _prop_request_url(ref: str) -> str:
    """Add a large page limit to propBets collection refs, not leaf refs."""
    normalized = core.normalize_espn_ref(ref.strip())
    path_only = normalized.split("?", 1)[0].rstrip("/").lower()

    if path_only.endswith("/propbets") and "limit=" not in normalized.lower():
        separator = "&" if "?" in normalized else "?"
        return f"{normalized}{separator}limit=2000"

    return normalized


def _iter_provider_prop_tree(
    client: core.ESPNClient,
    root_ref: str,
    *,
    max_refs: int = 400,
) -> Iterator[tuple[Any, int, str]]:
    """Recursively follow prop/market refs for one ESPN odds provider.

    ESPN's provider propBets endpoint may return collections containing further
    $ref objects rather than all market/outcome information inline.

    This stays scoped to the same provider's odds tree so unrelated ESPN event,
    league, team, athlete, or venue references are not followed.

    Yields:
        payload, traversal depth, normalized reference URL
    """
    root = core.normalize_espn_ref(root_ref.strip())
    root_lower = root.lower()

    if "/propbets" in root_lower:
        provider_scope = root_lower.split("/propbets", 1)[0]
    else:
        provider_scope = root_lower.rstrip("/")

    queue: list[tuple[str, int]] = [(root, 0)]
    seen: set[str] = set()
    cursor = 0

    while cursor < len(queue):
        if len(seen) >= max_refs:
            client.log(
                f"  provider prop traversal stopped after {max_refs} refs: "
                f"{root}"
            )
            break

        ref, depth = queue[cursor]
        cursor += 1

        normalized_ref = core.normalize_espn_ref(ref.strip())

        if normalized_ref in seen:
            continue

        seen.add(normalized_ref)

        payload = client.get_json(_prop_request_url(normalized_ref))
        if payload is None:
            continue

        yield payload, depth, normalized_ref

        for obj in core.walk_dicts(payload):
            nested_ref = obj.get("$ref")

            if not isinstance(nested_ref, str) or not nested_ref.strip():
                continue

            nested_ref = core.normalize_espn_ref(nested_ref.strip())
            nested_lower = nested_ref.lower()

            if nested_ref in seen:
                continue

            # Stay inside this sportsbook/provider odds branch.
            if not nested_lower.startswith(provider_scope):
                continue

            # Follow only betting-market branches, avoiding unrelated Core refs.
            if not any(
                marker in nested_lower
                for marker in _PROP_TREE_REF_MARKERS
            ):
                continue

            queue.append((nested_ref, depth + 1))


def _ingest_propbets_with_provider_hint(
    target: core.EventOdds,
    payload: Any,
    *,
    provider: str,
    provider_priority: int,
    path_priority: int,
    source: str,
) -> None:
    """Ingest props and preserve the parent odds provider when children omit it."""
    before = {
        field_name: len(offers)
        for field_name, offers in target.offers.items()
    }

    core.ingest_propbets(
        target,
        payload,
        path_priority=path_priority,
        source=source,
    )

    for field_name, offers in target.offers.items():
        start = before[field_name]

        for offer in offers[start:]:
            if core.normalize_provider_name(offer.provider) != "unknown":
                continue

            offer.provider = provider or "unknown"
            offer.provider_priority = core.safe_int(
                provider_priority,
                9999,
            )
            offer.sort_key = (
                core.provider_rank(offer.provider),
                offer.provider_priority,
                core.safe_int(offer.path_priority, 9999),
            )


def fetch_event_odds(
    client: core.ESPNClient,
    league_slug: str,
    event_id: str,
    competition_id: str,
    scoreboard_event: Mapping[str, Any],
) -> core.EventOdds:
    """Fetch ESPN odds using surfaces proven reachable from GitHub Actions."""
    result = core.EventOdds()

    # The CDN scoreboard event may itself contain usable betting objects.
    core.ingest_all_match_odds_objects(
        result,
        scoreboard_event,
        path_priority=0,
        source="cdn_scoreboard",
    )

    core_url = (
        f"{core.CORE_BASE}/{league_slug}/events/{event_id}/"
        f"competitions/{competition_id}/odds"
    )

    core_payload = client.get_json(
        core_url,
        params={"limit": 100},
    )

    if core_payload is None:
        return result

    # Core /odds is the primary source for 1X2 and the main full-match total.
    core.ingest_all_match_odds_objects(
        result,
        core_payload,
        path_priority=1,
        source="core_odds",
    )

    resolved_items = core.dereference_core_items(
        client,
        core_payload,
    )

    for item in resolved_items:
        core.ingest_all_match_odds_objects(
            result,
            item,
            path_priority=1,
            source="core_odds_ref",
        )

    # ESPN exposes provider-specific prop endpoints from each odds object, e.g.
    # .../odds/100/propBets for DraftKings.
    #
    # Follow those instead of the generic competition /propbets endpoint, which
    # returns 404 for these events. Provider prop trees can contain additional
    # nested references, so recursively traverse the provider betting branch.
    prop_sources = [
        core_payload,
        *resolved_items,
    ]

    seen_refs: set[str] = set()

    for source_payload in prop_sources:
        for ref, provider, provider_priority in _iter_provider_prop_refs(
            source_payload
        ):
            if ref in seen_refs:
                continue

            seen_refs.add(ref)

            for prop_payload, depth, prop_ref in _iter_provider_prop_tree(
                client,
                ref,
            ):
                source_name = (
                    "provider_propbets"
                    if depth == 0
                    else "provider_propbets_nested"
                )

                _ingest_propbets_with_provider_hint(
                    result,
                    prop_payload,
                    provider=provider,
                    provider_priority=provider_priority,
                    path_priority=2,
                    source=source_name,
                )

                if client.verbose:
                    yes_offer = result.best("btts_yes")
                    no_offer = result.best("btts_no")

                    if yes_offer or no_offer:
                        yes_text = (
                            f"{yes_offer.value:.2f}"
                            if yes_offer
                            else "missing"
                        )
                        no_text = (
                            f"{no_offer.value:.2f}"
                            if no_offer
                            else "missing"
                        )

                        client.log(
                            f"  BTTS found for {event_id} via {prop_ref}: "
                            f"yes={yes_text}, "
                            f"no={no_text}, "
                            f"provider={provider}"
                        )

    return result


def _is_blank(value: Any) -> bool:
    return value is None or not str(value).strip()


def _read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)

            return [
                {
                    field: str(row.get(field) or "")
                    for field in core.CSV_FIELDS
                }
                for row in reader
            ]

    except (OSError, csv.Error) as exc:
        print(
            f"warning: could not read existing CSV {path}: {exc}",
            file=sys.stderr,
        )
        return []


def _merge_rows(
    path: Path,
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Merge fresh rows into a prior daily CSV without erasing captured data."""
    existing_rows = _read_existing(path)

    existing_by_id = {
        row["game_id"]: row
        for row in existing_rows
        if not _is_blank(row.get("game_id"))
    }

    merged_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for fresh in rows:
        merged = {
            field: str(fresh.get(field) or "")
            for field in core.CSV_FIELDS
        }

        game_id = merged["game_id"].strip()
        prior = existing_by_id.get(game_id) if game_id else None

        if prior is not None:
            for field in core.CSV_FIELDS:
                if (
                    _is_blank(merged[field])
                    and not _is_blank(prior.get(field))
                ):
                    merged[field] = prior[field]

            seen_ids.add(game_id)

        merged_rows.append(merged)

    # Keep previously captured games if ESPN no longer returns them later that
    # day.
    for prior in existing_rows:
        game_id = prior.get("game_id", "").strip()

        if game_id and game_id in seen_ids:
            continue

        merged_rows.append(prior)

    merged_rows.sort(
        key=lambda row: (
            row.get("match_time", ""),
            row.get("home_team", ""),
            row.get("away_team", ""),
        )
    )

    return merged_rows


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_rows = _merge_rows(
        path,
        rows,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=core.CSV_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(merged_rows)


# core.main() resolves these globals from the core module at execution time.
# Replace endpoint selection while retaining the existing parsers, CLI, row
# construction, field priority rules, and non-destructive persistence.
core.fetch_scoreboard = fetch_scoreboard
core.fetch_event_odds = fetch_event_odds
core.write_csv = write_csv

main = core.main


if __name__ == "__main__":
    raise SystemExit(main())
