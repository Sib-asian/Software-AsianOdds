"""
Minor League Data Pipeline
==========================

Provides lightweight ingestion and normalization for secondary leagues that
usually lack coverage from mainstream APIs. The goal is to enrich the AI
pipeline with niche information (fixtures density, travel, roster churn) while
staying resilient when external feeds are unavailable.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MinorLeagueSnapshot:
    league: str
    coverage_level: str
    freshness_hours: float
    data_quality: float
    notes: List[str]
    teams_available: List[str]
    dataset_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "league": self.league,
            "coverage_level": self.coverage_level,
            "freshness_hours": round(self.freshness_hours, 1),
            "data_quality": round(self.data_quality, 3),
            "notes": self.notes,
            "teams_available": sorted(self.teams_available),
            "dataset_version": self.dataset_version,
        }


class MinorLeagueDataPipeline:
    """
    Registry-driven enrichment engine for small leagues. It keeps a lightweight
    cache (JSON) that can be updated by separate crawlers. During runtime we
    simply normalize and expose the data to the AI pipeline.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.cache_path = Path(self.config.cache_dir) / self.config.minor_league_cache_file
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, Dict[str, Any]] = self._load_registry()
        self._snapshot = self._build_snapshot()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def enrich(
        self,
        match: Dict[str, Any],
        api_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enrich the pipeline with additional info when match belongs to a minor
        league. Returns a dict with enrichment info and adjustments that other
        blocks can consume.
        """
        if not getattr(self.config, "minor_league_enabled", True):
            return {"status": "disabled"}

        league = (match.get("league") or "").strip().lower()
        if not league:
            return {"status": "unknown_league"}

        league_key = self._normalize_league(league)
        dataset_entry = self._registry.get(league_key)

        if not dataset_entry:
            if league in self.config.major_leagues:
                return {"status": "major_league"}
            return self._build_generic_fallback(match, api_context)

        teams = dataset_entry.get("teams", {})
        home_team = (match.get("home") or "").strip().lower()
        away_team = (match.get("away") or "").strip().lower()

        home_data = teams.get(home_team, {})
        away_data = teams.get(away_team, {})

        enrichment = {
            "status": "ok",
            "league": dataset_entry.get("league"),
            "coverage_level": dataset_entry.get("coverage_level", "baseline"),
            "dataset_version": dataset_entry.get("dataset_version", "v0"),
            "data_quality": dataset_entry.get("data_quality", 0.6),
            "freshness_hours": dataset_entry.get("freshness_hours", 24.0),
            "home_snapshot": home_data,
            "away_snapshot": away_data,
            "adjustments": self._derive_adjustments(home_data, away_data, dataset_entry),
        }

        return enrichment

    def get_snapshot(self) -> MinorLeagueSnapshot:
        return self._snapshot

    # --------------------------------------------------------------------- #
    # Registry management
    # --------------------------------------------------------------------- #

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            logger.info("ℹ️ Minor league registry not found (%s). Using empty dataset.", self.cache_path)
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            logger.info("✅ Minor league registry loaded (%d leagues)", len(raw))
            return raw
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("❌ Failed to load minor league registry: %s", exc)
            return {}

    def update_registry(self, registry: Dict[str, Dict[str, Any]]) -> None:
        """Replace registry content (e.g. called by external crawler)."""
        self._registry = registry
        self._snapshot = self._build_snapshot()
        try:
            with self.cache_path.open("w", encoding="utf-8") as fh:
                json.dump(self._registry, fh, indent=2)
            logger.info("🗃️ Minor league registry updated (%d leagues)", len(registry))
        except OSError as exc:
            logger.error("❌ Failed to persist minor league registry: %s", exc)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _build_snapshot(self) -> MinorLeagueSnapshot:
        leagues = list(self._registry.values())
        if not leagues:
            return MinorLeagueSnapshot(
                league="N/A",
                coverage_level="empty",
                freshness_hours=float("inf"),
                data_quality=0.0,
                notes=["Registry empty"],
                teams_available=[],
                dataset_version="v0",
            )

        newest_timestamp = 0.0
        cumulative_quality = 0.0
        total_leagues = len(leagues)
        team_set = set()
        coverage_levels = defaultdict(int)

        for entry in leagues:
            dataset_ts = entry.get("last_updated_ts") or 0
            newest_timestamp = max(newest_timestamp, dataset_ts)
            cumulative_quality += entry.get("data_quality", 0.6)
            coverage_levels[entry.get("coverage_level", "baseline")] += 1
            team_set.update(entry.get("teams", {}).keys())

        freshest_dt = datetime.utcfromtimestamp(newest_timestamp) if newest_timestamp else datetime.utcfromtimestamp(0)
        freshness_hours = (datetime.utcnow() - freshest_dt).total_seconds() / 3600 if newest_timestamp else float("inf")

        dominant_coverage = max(coverage_levels.items(), key=lambda item: item[1])[0]
        dataset_version = leagues[0].get("dataset_version", "v0")

        return MinorLeagueSnapshot(
            league=f"{total_leagues} leagues",
            coverage_level=dominant_coverage,
            freshness_hours=freshness_hours,
            data_quality=cumulative_quality / max(total_leagues, 1),
            notes=[],
            teams_available=list(team_set),
            dataset_version=dataset_version,
        )

    def _derive_adjustments(
        self,
        home_data: Dict[str, Any],
        away_data: Dict[str, Any],
        league_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        adjustments = {
            "league_quality_override": league_entry.get("league_quality_override"),
            "data_quality_bonus": league_entry.get("data_quality_bonus", 0.0),
            "flagged_risks": [],
            "flagged_opportunities": [],
        }

        def _scan_team(team_data: Dict[str, Any], role: str) -> None:
            fatigue_idx = team_data.get("fatigue_index")
            travel_km = team_data.get("travel_km_last7")
            injuries = team_data.get("injuries", [])

            if fatigue_idx and fatigue_idx > self.config.minor_league_fatigue_high:
                adjustments["flagged_risks"].append(f"{role}: fatica elevata ({fatigue_idx:.1f})")
            elif fatigue_idx and fatigue_idx < self.config.minor_league_fatigue_low:
                adjustments["flagged_opportunities"].append(f"{role}: squadra fresca ({fatigue_idx:.1f})")

            if travel_km and travel_km > self.config.minor_league_travel_high_km:
                adjustments["flagged_risks"].append(f"{role}: viaggio lungo ({travel_km} km)")

            if injuries:
                adjustments["flagged_risks"].append(f"{role}: {len(injuries)} infortuni segnalati")

        _scan_team(home_data, "home")
        _scan_team(away_data, "away")

        return adjustments

    def _build_generic_fallback(
        self,
        match: Dict[str, Any],
        api_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Provide graceful fallback when league is genuinely minor but we have no
        curated dataset yet. We infer simple heuristics from base context.
        """
        api_context = api_context or {}
        metadata = api_context.get("metadata", {})
        data_quality = metadata.get("data_quality", 0.4)

        notes = ["Fallback heuristics"]
        if data_quality < 0.4:
            notes.append("Data quality bassa dalle API principali")
        if metadata.get("importance", 0.3) < 0.4:
            notes.append("Match a bassa priorità nella pipeline principale")

        fallback = {
            "status": "fallback",
            "league": match.get("league"),
            "coverage_level": "fallback",
            "dataset_version": "v0",
            "data_quality": data_quality,
            "freshness_hours": float("inf"),
            "home_snapshot": {},
            "away_snapshot": {},
            "adjustments": {
                "league_quality_override": self.config.minor_league_default_quality,
                "data_quality_bonus": -0.05,
                "flagged_risks": notes,
                "flagged_opportunities": [],
            },
        }
        return fallback

    @staticmethod
    def _normalize_league(league: str) -> str:
        return league.replace("  ", " ").strip().lower()

