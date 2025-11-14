"""
StatsBomb Open Data Client
==========================

Utility leggero per integrare i dati open di StatsBomb nella pipeline.
Scarica i match e gli eventi rilevanti, calcola metriche aggregate (xG, pressioni,
shot quality) e le mette a disposizione dei blocchi AI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from statsbombpy import sb

    STATS_BOMB_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    STATS_BOMB_AVAILABLE = False
    sb = None  # type: ignore


def _normalize_name(value: Optional[str]) -> str:
    if not value:
        return ""
    return (
        value.lower()
        .replace(".", "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )


def _safe_get(team_blob: Any, *keys: str) -> Optional[str]:
    if isinstance(team_blob, dict):
        for key in keys:
            if key in team_blob and team_blob[key]:
                return str(team_blob[key])
    if isinstance(team_blob, str):
        return team_blob
    return None


@dataclass
class StatsBombSettings:
    enabled: bool = True
    max_matches: int = 5
    cache_hours: int = 6
    min_matches_required: int = 2


class StatsBombClient:
    """Wrapper minimale sul pacchetto statsbombpy."""

    def __init__(self, settings: StatsBombSettings):
        self.settings = settings
        self.enabled = settings.enabled and STATS_BOMB_AVAILABLE

        self._matches_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._team_metrics_cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self._events_cache: Dict[int, Dict[str, Any]] = {}

        if not STATS_BOMB_AVAILABLE:
            logger.warning(
                "⚠️ statsbombpy non installato - esegui `pip install statsbombpy` per abilitare i dati StatsBomb."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_match_context(self, match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Restituisce metrics StatsBomb per home/away se disponibili."""
        if not self.enabled:
            return None

        league = match.get("league") or match.get("competition")
        if not league:
            return None

        match_date = self._parse_datetime(match.get("date") or match.get("match_date"))
        competition = self._resolve_competition(league, match.get("season"))
        if not competition:
            return None

        comp_id, season_id, comp_name, season_name = competition
        home = match.get("home")
        away = match.get("away")
        if not home or not away:
            return None

        home_metrics = self._get_team_metrics(home, comp_id, season_id, match_date)
        away_metrics = self._get_team_metrics(away, comp_id, season_id, match_date)

        if not home_metrics and not away_metrics:
            return None

        return {
            "competition_id": comp_id,
            "season_id": season_id,
            "competition": comp_name,
            "season": season_name,
            "home": home_metrics,
            "away": away_metrics,
            "source": "statsbomb_open_data",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_competition(
        self, league: str, season_label: Optional[str]
    ) -> Optional[Tuple[int, int, str, str]]:
        try:
            competitions = sb.competitions()  # type: ignore[call-arg]
            competitions = self._ensure_dataframe(competitions)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning(f"⚠️ StatsBomb competitions fetch failed: {exc}")
            return None

        league_norm = _normalize_name(league)
        if competitions.empty:
            return None

        mask = competitions["competition_name"].str.lower().str.contains(league_norm, na=False)
        subset = competitions[mask]
        if subset.empty:
            return None

        if season_label:
            season_mask = subset["season_name"].str.contains(
                str(season_label), case=False, na=False
            )
            if season_mask.any():
                subset = subset[season_mask]

        # Prendi la stagione più recente disponibile
        subset = subset.sort_values(["season_id"], ascending=False)
        row = subset.iloc[0]

        return (
            int(row["competition_id"]),
            int(row["season_id"]),
            str(row["competition_name"]),
            str(row["season_name"]),
        )

    def _get_team_metrics(
        self,
        team_name: str,
        comp_id: int,
        season_id: int,
        match_date: Optional[datetime],
    ) -> Optional[Dict[str, Any]]:
        cache_key = (_normalize_name(team_name), comp_id, season_id)
        cached = self._team_metrics_cache.get(cache_key)
        if cached and not self._is_cache_expired(cached["timestamp"]):
            return cached["data"]

        matches_df = self._get_matches(comp_id, season_id)
        if matches_df is None or matches_df.empty:
            return None

        team_matches = self._filter_team_matches(matches_df, team_name, match_date)
        if not team_matches:
            return None

        # Limita a max_matches per non appesantire gli eventi
        selected = team_matches[-self.settings.max_matches :]

        aggregate = {
            "matches_used": 0,
            "xg_total": 0.0,
            "xga_total": 0.0,
            "shots": 0,
            "shots_on_target": 0,
            "pressures": 0,
        }

        for info in selected:
            match_metrics = self._compute_match_metrics(info["match_id"], team_name)
            if not match_metrics:
                continue

            aggregate["matches_used"] += 1
            aggregate["xg_total"] += match_metrics["xg_for"]
            aggregate["xga_total"] += match_metrics["xg_against"]
            aggregate["shots"] += match_metrics["shots"]
            aggregate["shots_on_target"] += match_metrics["shots_on_target"]
            aggregate["pressures"] += match_metrics["pressures"]

        matches_used = aggregate["matches_used"]
        if matches_used < self.settings.min_matches_required:
            return None

        result = {
            "matches": matches_used,
            "avg_xg_per_match": round(aggregate["xg_total"] / matches_used, 3),
            "avg_xga_per_match": round(aggregate["xga_total"] / matches_used, 3),
            "avg_shots_per_match": round(aggregate["shots"] / matches_used, 2),
            "shots_on_target_pct": round(
                aggregate["shots_on_target"] / max(1, aggregate["shots"]), 3
            ),
            "pressures_per_match": round(aggregate["pressures"] / matches_used, 1),
            "last_update": datetime.utcnow().isoformat(),
        }

        self._team_metrics_cache[cache_key] = {
            "timestamp": datetime.utcnow(),
            "data": result,
        }

        return result

    def _get_matches(self, comp_id: int, season_id: int) -> Optional[pd.DataFrame]:
        cache_key = (comp_id, season_id)
        cached = self._matches_cache.get(cache_key)
        if cached and not self._is_cache_expired(cached["timestamp"]):
            return cached["data"]

        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)  # type: ignore[call-arg]
            matches = self._ensure_dataframe(matches)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning(f"⚠️ StatsBomb matches fetch failed: {exc}")
            return None

        self._matches_cache[cache_key] = {
            "timestamp": datetime.utcnow(),
            "data": matches,
        }
        return matches

    def _filter_team_matches(
        self,
        matches_df: pd.DataFrame,
        team_name: str,
        match_date: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        target = _normalize_name(team_name)
        results: List[Dict[str, Any]] = []

        for _, row in matches_df.iterrows():
            home_name = _normalize_name(
                _safe_get(row.get("home_team"), "home_team_name", "name")
            )
            away_name = _normalize_name(
                _safe_get(row.get("away_team"), "away_team_name", "name")
            )
            if target not in {home_name, away_name}:
                continue

            match_dt = self._parse_datetime(row.get("match_date"))
            if match_date and match_dt and match_dt > match_date:
                # Usa solo match già giocati rispetto alla data richiesta
                continue

            results.append(
                {
                    "match_id": int(row["match_id"]),
                    "match_date": match_dt,
                    "is_home": target == home_name,
                }
            )

        results.sort(key=lambda item: item["match_date"] or datetime.min)
        return results

    def _compute_match_metrics(self, match_id: int, team_name: str) -> Optional[Dict[str, Any]]:
        events = self._get_events(match_id)
        if events is None or events.empty:
            return None

        team_norm = _normalize_name(team_name)
        metrics = {
            "xg_for": 0.0,
            "xg_against": 0.0,
            "shots": 0,
            "shots_on_target": 0,
            "pressures": 0,
        }

        for _, event in events.iterrows():
            event_team = _normalize_name(_safe_get(event.get("team"), "name"))
            event_type = _safe_get(event.get("type"), "name")

            if event_type == "Shot":
                shot = event.get("shot") or {}
                xg = float(shot.get("statsbomb_xg") or 0.0)
                outcome = _safe_get(shot.get("outcome"), "name") or ""

                if event_team == team_norm:
                    metrics["xg_for"] += xg
                    metrics["shots"] += 1
                    if outcome in {"Goal", "Saved", "Saved To Post"}:
                        metrics["shots_on_target"] += 1
                else:
                    metrics["xg_against"] += xg

            elif event_type == "Pressure" and event_team == team_norm:
                metrics["pressures"] += 1

        return metrics

    def _get_events(self, match_id: int) -> Optional[pd.DataFrame]:
        cached = self._events_cache.get(match_id)
        if cached and not self._is_cache_expired(cached["timestamp"]):
            return cached["data"]

        try:
            events = sb.events(match_id=match_id)  # type: ignore[call-arg]
            events = self._ensure_dataframe(events)
        except Exception as exc:  # pragma: no cover - network errors
            logger.debug(f"StatsBomb events unavailable for match {match_id}: {exc}")
            return None

        self._events_cache[match_id] = {
            "timestamp": datetime.utcnow(),
            "data": events,
        }
        return events

    @staticmethod
    def _ensure_dataframe(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            return pd.DataFrame(data)
        return pd.DataFrame(data)

    def _is_cache_expired(self, timestamp: datetime) -> bool:
        return datetime.utcnow() - timestamp > timedelta(hours=self.settings.cache_hours)

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None
