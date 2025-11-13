"""
News & Social Sentiment Monitor
===============================

Provides lightweight aggregation of external narrative signals (news, blogs,
social posts) around teams and leagues. The primary goal is to surface signals
that can influence betting decisions without depending on heavyweight NLP
pipelines or paid data feeds.

The monitor follows a three-step process:
1. Ingest           → fetch raw items from configured sources (or cached data)
2. Score            → apply keyword heuristics to estimate sentiment and impact
3. Summarise        → produce an aggregate signal consumable by the AI pipeline

The implementation is intentionally resilient: when the crawler cannot reach
external sources (e.g. offline environment) it falls back to heuristics based on
existing match context (injuries, form, odds movements) so the pipeline still
receives a meaningful signal.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class SentimentSignal:
    """Structured sentiment signal for a single team."""

    team: str
    league: str
    sentiment_score: float
    magnitude: float
    highlights: List[str]
    last_updated: datetime
    sources: List[str]
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "league": self.league,
            "sentiment_score": self.sentiment_score,
            "magnitude": self.magnitude,
            "highlights": self.highlights,
            "last_updated": self.last_updated.isoformat(),
            "sources": self.sources,
            "status": self.status,
        }


class NewsSentimentMonitor:
    """
    Aggregates news and social signals for home/away teams and produces a
    compact summary for the pipeline.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.cache_path = Path(self.config.cache_dir) / self.config.news_sentiment_cache_file
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Dict[str, Any]] = self._load_cache()

        # Compile regexes once
        self._positive_patterns = [
            re.compile(rf"\b{kw}\b", re.IGNORECASE)
            for kw in self.config.news_sentiment_positive_keywords
        ]
        self._negative_patterns = [
            re.compile(rf"\b{kw}\b", re.IGNORECASE)
            for kw in self.config.news_sentiment_negative_keywords
        ]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def collect(
        self,
        match: Dict[str, Any],
        api_context: Optional[Dict[str, Any]] = None,
        odds_context: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Return aggregated sentiment signals for a given match.
        """
        if not getattr(self.config, "news_sentiment_enabled", True):
            return {
                "status": "disabled",
                "reason": "news_sentiment_disabled_in_config",
                "home": None,
                "away": None,
                "aggregate": {"score_diff": 0.0, "bias": "neutral", "confidence": 0.0},
            }

        match_key = self._build_match_key(match)
        if not force_refresh:
            cached = self._fetch_from_cache(match_key)
            if cached:
                return cached

        home_signal = self._build_signal(
            team=match.get("home", "Unknown"),
            league=match.get("league", "Unknown"),
            team_context=(api_context or {}).get("home_context", {}),
            odds_context=odds_context or {},
        )
        away_signal = self._build_signal(
            team=match.get("away", "Unknown"),
            league=match.get("league", "Unknown"),
            team_context=(api_context or {}).get("away_context", {}),
            odds_context=odds_context or {},
        )

        aggregate = self._aggregate_signals(home_signal, away_signal)
        payload = {
            "status": "ok",
            "home": home_signal.to_dict(),
            "away": away_signal.to_dict(),
            "aggregate": aggregate,
        }

        self._store_in_cache(match_key, payload)
        return payload

    # --------------------------------------------------------------------- #
    # Signal construction
    # --------------------------------------------------------------------- #

    def _build_signal(
        self,
        team: str,
        league: str,
        team_context: Optional[Dict[str, Any]] = None,
        odds_context: Optional[Dict[str, Any]] = None,
    ) -> SentimentSignal:
        """Build sentiment signal for a specific team."""
        team_context = team_context or {}
        raw_items = self._ingest_items(team, league)
        sentiment_score, magnitude, highlights, sources = self._score_items(raw_items)

        # If we have no items, fall back to contextual heuristics so the score is
        # still informative.
        if not raw_items:
            sentiment_score, magnitude, highlights, sources = self._fallback_from_context(
                team, league, team_context, odds_context
            )

        return SentimentSignal(
            team=team,
            league=league,
            sentiment_score=round(sentiment_score, 3),
            magnitude=round(magnitude, 3),
            highlights=highlights[:5],
            sources=sources[:5],
            last_updated=datetime.utcnow(),
            status="ok",
        )

    # --------------------------------------------------------------------- #
    # Ingestion & Scoring
    # --------------------------------------------------------------------- #

    def _ingest_items(self, team: str, league: str) -> List[Dict[str, Any]]:
        """
        Placeholder for actual crawling logic.

        Currently the system first checks the local cache (which can be
        periodically refreshed by background jobs). If no cached feed is
        available we return an empty list and rely on contextual fallback.
        """
        dataset = self._memory_cache.get("dataset", {})
        team_key = self._normalize_key(team, league)
        items = dataset.get(team_key, [])
        if items:
            logger.debug("📰 Loaded %d cached items for %s", len(items), team_key)
        return items

    def _score_items(
        self, items: List[Dict[str, Any]]
    ) -> Tuple[float, float, List[str], List[str]]:
        """Score sentiment using keyword heuristics."""
        if not items:
            return 0.5, 0.0, [], []

        positive_hits = 0
        negative_hits = 0
        highlights = []
        sources = []

        for item in items:
            text = str(item.get("title") or "") + " " + str(item.get("summary") or "")
            weight = _safe_float(item.get("weight"), 1.0)

            pos = sum(bool(pattern.search(text)) for pattern in self._positive_patterns)
            neg = sum(bool(pattern.search(text)) for pattern in self._negative_patterns)

            if pos > 0 or neg > 0:
                highlights.append(text[:140] + ("..." if len(text) > 140 else ""))
            if item.get("source"):
                sources.append(item["source"])

            positive_hits += pos * weight
            negative_hits += neg * weight

        total_hits = positive_hits + negative_hits
        if total_hits == 0:
            return 0.5, 0.1, highlights, list(dict.fromkeys(sources))

        sentiment_score = positive_hits / total_hits
        magnitude = min(1.0, math.log1p(total_hits) / 2.0)

        return sentiment_score, magnitude, highlights, list(dict.fromkeys(sources))

    def _fallback_from_context(
        self,
        team: str,
        league: str,
        team_context: Dict[str, Any],
        odds_context: Dict[str, Any],
    ) -> Tuple[float, float, List[str], List[str]]:
        """
        Derive a pseudo sentiment score from available structured context. This
        ensures the pipeline always receives a usable signal, even without
        external data access.
        """
        base_score = 0.55
        magnitude = 0.15
        highlights: List[str] = []
        sources = ["contextual-fallback"]

        injuries = []
        if isinstance(team_context.get("data"), dict):
            injuries = team_context["data"].get("injuries", [])

        if injuries:
            penalty = min(0.2, 0.05 * len(injuries))
            base_score -= penalty
            magnitude += 0.05 * len(injuries)
            highlights.append(f"{len(injuries)} infortuni registrati")

        form = ""
        if isinstance(team_context.get("data"), dict):
            form = team_context["data"].get("form", "")
        if not form:
            form = team_context.get("data", {}).get("recent_form", "")

        if form:
            wins = form.count("W")
            losses = form.count("L")
            base_score += 0.02 * wins
            base_score -= 0.03 * losses
            if wins >= 3:
                highlights.append("Serie positiva recente")
            if losses >= 3:
                highlights.append("Serie negativa recente")

        odds_history = odds_context.get("odds_history") or []
        if odds_history:
            odds_values = [entry.get("odds", 0) for entry in odds_history if entry.get("odds")]
            if len(odds_values) >= 2:
                first = odds_values[0]
                last = odds_values[-1]
                if first > 0 and last > 0:
                    delta = (last - first) / first
                    if delta < -0.05:
                        base_score += 0.05
                        highlights.append("Movimento quote favorevole")
                    elif delta > 0.05:
                        base_score -= 0.05
                        highlights.append("Quote in peggioramento")

        sentiment_score = min(max(base_score, 0.0), 1.0)
        magnitude = min(max(magnitude, 0.0), 1.0)

        logger.debug(
            "✨ Fallback sentiment for %s (%s): score=%.2f magnitude=%.2f",
            team,
            league,
            sentiment_score,
            magnitude,
        )

        return sentiment_score, magnitude, highlights, sources

    # --------------------------------------------------------------------- #
    # Aggregation
    # --------------------------------------------------------------------- #

    def _aggregate_signals(
        self, home: SentimentSignal, away: SentimentSignal
    ) -> Dict[str, Any]:
        score_diff = home.sentiment_score - away.sentiment_score
        bias = "home" if score_diff > 0.05 else "away" if score_diff < -0.05 else "neutral"

        confidence = (home.magnitude + away.magnitude) / 2.0
        return {
            "score_diff": round(score_diff, 3),
            "bias": bias,
            "confidence": round(confidence, 3),
        }

    # --------------------------------------------------------------------- #
    # Cache helpers
    # --------------------------------------------------------------------- #

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("⚠️ Failed to load sentiment cache: %s", exc)
            return {}

    def _fetch_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        cached = self._memory_cache.get("matches", {}).get(key)
        if not cached:
            return None

        timestamp_str = cached.get("timestamp")
        if not timestamp_str:
            return None

        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            return None

        ttl = timedelta(hours=self.config.news_sentiment_cache_ttl_hours)
        if datetime.utcnow() - timestamp > ttl:
            logger.debug("🗑️ Sentiment cache expired for %s", key)
            return None

        return cached.get("payload")

    def _store_in_cache(self, key: str, payload: Dict[str, Any]) -> None:
        matches = self._memory_cache.setdefault("matches", {})
        matches[key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload,
        }
        try:
            with self.cache_path.open("w", encoding="utf-8") as fh:
                json.dump(self._memory_cache, fh, indent=2)
        except OSError as exc:
            logger.error("❌ Failed to write sentiment cache: %s", exc)

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalize_key(team: str, league: str) -> str:
        return f"{(team or '').strip().lower()}::{(league or '').strip().lower()}"

    def _build_match_key(self, match: Dict[str, Any]) -> str:
        parts = [
            match.get("home", "").strip().lower(),
            match.get("away", "").strip().lower(),
            match.get("league", "").strip().lower(),
            match.get("date", ""),
        ]
        return "::".join(parts)

    # --------------------------------------------------------------------- #
    # Dataset management API (optional for background jobs)
    # --------------------------------------------------------------------- #

    def update_dataset(self, dataset: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Replace the cached raw dataset used for scoring.
        Exposed so that background jobs/scrapers can push fresh data.
        """
        self._memory_cache["dataset"] = dataset
        try:
            with self.cache_path.open("w", encoding="utf-8") as fh:
                json.dump(self._memory_cache, fh, indent=2)
            logger.info("🗃️ News sentiment dataset updated (%d teams)", len(dataset))
        except OSError as exc:
            logger.error("❌ Failed to write sentiment dataset: %s", exc)

