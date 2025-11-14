"""
TheOddsAPI Client
=================

Wrapper minimale per interrogare il tier gratuito di TheOddsAPI e ottenere snapshot
aggiornati delle quote dei bookmaker. Usato dal BLOCCO 6 per arricchire l'odds tracker.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _normalize(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.lower().strip()


class TheOddsAPIClient:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(
        self,
        api_key: str,
        regions: str = "eu",
        markets: Optional[List[str]] = None,
        primary_market: str = "h2h",
        odds_format: str = "decimal",
        date_format: str = "iso",
        sport_mapping: Optional[Dict[str, str]] = None
    ):
        self.api_key = api_key
        self.enabled = bool(api_key)
        self.regions = regions
        self.markets = markets or ["h2h"]
        self.primary_market = primary_market or "h2h"
        self.odds_format = odds_format
        self.date_format = date_format
        self.sport_mapping = sport_mapping or {}

        if not self.enabled:
            logger.warning("⚠️ THEODDS_API_KEY non configurata - skipping live odds refresh.")

    def fetch_latest_snapshot(
        self,
        match: Dict[str, Any],
        market: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Scarica le ultime quote per il match richiesto."""
        if not self.enabled:
            return None

        sport_key = self._resolve_sport_key(match.get("league"))
        if not sport_key:
            return None

        market_key = market or self.primary_market
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": self.regions,
                    "markets": market_key,
                    "oddsFormat": self.odds_format,
                    "dateFormat": self.date_format,
                },
                timeout=8
            )
            response.raise_for_status()
            events = response.json()
        except requests.RequestException as exc:
            logger.debug(f"TheOddsAPI request failed: {exc}")
            return None

        event = self._match_event(events, match)
        if not event:
            return None

        prices = self._extract_prices(event, market_key, match)
        if not prices:
            return None

        return {
            "event_id": event.get("id"),
            "market": market_key,
            "prices": prices,
            "timestamp": datetime.utcnow().isoformat(),
            "bookmakers_queried": len(event.get("bookmakers", [])),
            "sport_key": sport_key,
        }

    def _resolve_sport_key(self, league: Optional[str]) -> Optional[str]:
        if not league:
            return None
        league_norm = _normalize(league)
        for keyword, sport_key in self.sport_mapping.items():
            if keyword in league_norm:
                return sport_key
        return None

    def _match_event(self, events: List[Dict[str, Any]], match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        home = _normalize(match.get("home"))
        away = _normalize(match.get("away"))
        if not home or not away:
            return None

        for event in events:
            if _normalize(event.get("home_team")) == home and _normalize(event.get("away_team")) == away:
                return event
        return None

    def _extract_prices(
        self,
        event: Dict[str, Any],
        market: str,
        match: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        bookmakers = event.get("bookmakers") or []
        if not bookmakers:
            return None

        selections = {
            "home": _normalize(match.get("home")),
            "away": _normalize(match.get("away")),
            "draw": "draw"
        }

        best_prices = {
            "home": None,
            "away": None,
            "draw": None,
        }
        best_sources = {
            "home": None,
            "away": None,
            "draw": None,
        }

        for bookmaker in bookmakers:
            markets = bookmaker.get("markets") or []
            market_entry = next((m for m in markets if m.get("key") == market), None)
            if not market_entry:
                continue

            for outcome in market_entry.get("outcomes", []):
                name_norm = _normalize(outcome.get("name"))
                price = outcome.get("price")
                if price is None:
                    continue

                target = None
                if name_norm == selections["home"]:
                    target = "home"
                elif name_norm == selections["away"]:
                    target = "away"
                elif name_norm in {"draw", "pareggio", "tie"}:
                    target = "draw"

                if target and (best_prices[target] is None or price > best_prices[target]):
                    best_prices[target] = price
                    best_sources[target] = bookmaker.get("title")

        if not any(best_prices.values()):
            return None

        return {
            "home": {
                "price": best_prices["home"],
                "bookmaker": best_sources["home"]
            },
            "away": {
                "price": best_prices["away"],
                "bookmaker": best_sources["away"]
            },
            "draw": {
                "price": best_prices["draw"],
                "bookmaker": best_sources["draw"]
            }
        }
