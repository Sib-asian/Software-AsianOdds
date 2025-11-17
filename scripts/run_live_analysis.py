#!/usr/bin/env python3
"""
Esegue la pipeline AI su un evento reale proveniente da TheOddsAPI.

Usage:
    python scripts/run_live_analysis.py --sport soccer_epl --event-index 0

Richiede:
    - THEODDS_API_KEY definita (in .env o variabile d'ambiente)
    - Dipendenze installate via requirements.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_system.pipeline import AIPipeline  # pylint: disable=wrong-import-position
from ai_system.config import AIConfig  # pylint: disable=wrong-import-position


SPORT_LEAGUE_MAP = {
    "soccer_epl": "Premier League",
    "soccer_italy_serie_a": "Serie A",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_uefa_champs_league": "Champions League",
}


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("THEODDS_API_KEY")
    if not api_key:
        raise SystemExit(
            "THEODDS_API_KEY non trovata. Aggiungila al file .env o esportala nell'ambiente."
        )
    return api_key


def fetch_events(api_key: str, sport_key: str, regions: str, markets: str) -> List[Dict]:
    resp = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/",
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=10,
    )
    resp.raise_for_status()
    events = resp.json()
    if not isinstance(events, list) or not events:
        raise SystemExit(f"Nessun evento trovato per {sport_key}")
    return events


def build_odds_history(event: Dict, selection: str) -> List[Dict]:
    history = []
    for bookmaker in event.get("bookmakers", []):
        market = next(
            (m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None
        )
        if not market:
            continue
        price_entry = next(
            (o for o in market.get("outcomes", []) if o.get("name") == selection),
            None,
        )
        if not price_entry or price_entry.get("price") is None:
            continue
        history.append(
            {
                "timestamp": market.get("last_update"),
                "bookmaker": bookmaker.get("title"),
                "odds": float(price_entry["price"]),
            }
        )
    history.sort(key=lambda x: x.get("timestamp") or "")
    return history


def build_match(event: Dict, league_override: Optional[str]) -> Dict:
    league = league_override or SPORT_LEAGUE_MAP.get(event.get("sport_key"), "Premier League")
    return {
        "home": event.get("home_team"),
        "away": event.get("away_team"),
        "league": league,
        "date": event.get("commence_time"),
        "season": datetime.now().strftime("%Y/%Y"),
    }


def compute_hours_to_kickoff(commence_time: str) -> float:
    if not commence_time:
        return 0.0
    if commence_time.endswith("Z"):
        commence_time = commence_time[:-1] + "+00:00"
    commence_dt = datetime.fromisoformat(commence_time)
    if commence_dt.tzinfo is None:
        commence_dt = commence_dt.replace(tzinfo=timezone.utc)
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    return max((commence_dt - now).total_seconds() / 3600.0, 0.0)


def run_analysis(event: Dict, args: argparse.Namespace) -> Dict:
    match = build_match(event, args.league)
    selection_name = event.get("home_team") if args.selection == "home" else event.get("away_team")
    odds_history = build_odds_history(event, selection_name)
    if not odds_history:
        raise SystemExit("Impossibile costruire odds_history dall'evento selezionato.")

    odds_data = {
        "market": "h2h",
        "selection": args.selection,
        "odds_current": odds_history[-1]["odds"],
        "odds_history": odds_history,
        "time_to_kickoff_hours": compute_hours_to_kickoff(event.get("commence_time")),
        "historical_accuracy": args.historical_accuracy,
        "similar_bets_roi": args.similar_bets_roi,
        "similar_bets_count": args.similar_bets_count,
        "similar_bets_winrate": args.similar_bets_winrate,
    }

    pipeline = AIPipeline(config=AIConfig(log_level=args.log_level, use_ensemble=False))
    result = pipeline.analyze(
        match=match,
        prob_dixon_coles=args.base_probability,
        odds_data=odds_data,
        bankroll=args.bankroll,
    )

    return {
        "match": f"{match['home']} vs {match['away']}",
        "sport_key": event.get("sport_key"),
        "event_id": event.get("id"),
        "decision": result["risk_decision"]["decision"],
        "stake": result["risk_decision"]["final_stake"],
        "timing": result["timing"]["timing_recommendation"],
        "live_snapshot": result["timing"].get("live_odds_snapshot"),
        "chronos_prediction": result["timing"].get("predicted_odds_1h"),
        "precision_sources": result["api_context"]["match_data"].get("precision_sources", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live analysis via TheOddsAPI.")
    parser.add_argument("--sport", default="soccer_epl", help="Sport key TheOddsAPI (default: soccer_epl)")
    parser.add_argument("--regions", default="eu", help="Regions parameter for TheOddsAPI")
    parser.add_argument("--markets", default="h2h", help="Markets parameter (default: h2h)")
    parser.add_argument("--event-index", type=int, default=0, help="Index of the event to analyze")
    parser.add_argument("--selection", choices=["home", "away"], default="home", help="Selection to analyze")
    parser.add_argument("--league", default=None, help="Override league name passed to pipeline")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll for Kelly calculation")
    parser.add_argument("--base-probability", type=float, default=0.5, help="Base probability (Dixon-Coles)")
    parser.add_argument("--historical-accuracy", type=float, default=0.70, help="Historical accuracy input")
    parser.add_argument("--similar-bets-roi", type=float, default=0.05, help="Similar bets ROI input")
    parser.add_argument("--similar-bets-count", type=int, default=150, help="Similar bets count input")
    parser.add_argument("--similar-bets-winrate", type=float, default=0.57, help="Similar bets winrate input")
    parser.add_argument("--log-level", default="ERROR", help="Log level for AIConfig")
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = load_api_key()
    events = fetch_events(api_key, args.sport, args.regions, args.markets)
    if args.event_index >= len(events):
        raise SystemExit(f"Indice evento {args.event_index} fuori range (max {len(events)-1}).")
    event = events[args.event_index]
    summary = run_analysis(event, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
