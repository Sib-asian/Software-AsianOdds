#!/usr/bin/env python3
"""
Monitor live periodico: interroga TheOddsAPI ogni N secondi e lancia la pipeline
su una lista di eventi, salvando/mostrando lo snapshot risultante.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_live_analysis as live  # type: ignore  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live monitor basato su TheOddsAPI")
    parser.add_argument("--sport", default="soccer_epl", help="Sport key (es. soccer_epl)")
    parser.add_argument("--regions", default="eu", help="Param regions per TheOddsAPI")
    parser.add_argument("--markets", default="h2h", help="Param markets (default h2h)")
    parser.add_argument("--selection", choices=["home", "away"], default="home", help="Selezione da analizzare")
    parser.add_argument("--max-events", type=int, default=3, help="Numero massimo di eventi per ciclo")
    parser.add_argument("--interval", type=int, default=900, help="Intervallo in secondi fra i cicli (default 15min)")
    parser.add_argument("--iterations", type=int, default=0, help="Numero di cicli (0 = infinito)")
    parser.add_argument("--league", default=None, help="Override nome lega")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--base-probability", type=float, default=0.5)
    parser.add_argument("--historical-accuracy", type=float, default=0.7)
    parser.add_argument("--similar-bets-roi", type=float, default=0.05)
    parser.add_argument("--similar-bets-count", type=int, default=150)
    parser.add_argument("--similar-bets-winrate", type=float, default=0.57)
    parser.add_argument("--log-level", default="ERROR")
    parser.add_argument("--save-path", help="File JSONL dove salvare i risultati (append)")
    return parser.parse_args()


def build_run_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        selection=args.selection,
        league=args.league,
        historical_accuracy=args.historical_accuracy,
        similar_bets_roi=args.similar_bets_roi,
        similar_bets_count=args.similar_bets_count,
        similar_bets_winrate=args.similar_bets_winrate,
        log_level=args.log_level,
        base_probability=args.base_probability,
        bankroll=args.bankroll,
    )


def main():
    args = parse_args()
    load_dotenv()
    api_key = live.load_api_key()
    iteration = 0
    save_file = Path(args.save_path) if args.save_path else None

    while args.iterations <= 0 or iteration < args.iterations:
        iteration += 1
        print(f"\n=== Iterazione {iteration} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
        try:
            events = live.fetch_events(api_key, args.sport, args.regions, args.markets)
        except Exception as exc:  # pragma: no cover
            print(f"❌ Errore fetch events: {exc}")
            if args.iterations <= 0 or iteration < args.iterations:
                time.sleep(args.interval)
            continue

        run_args = build_run_args(args)
        for idx, event in enumerate(events[: args.max_events]):
            try:
                summary = live.run_analysis(event, run_args)
            except Exception as exc:  # pragma: no cover
                print(f"❌ Errore analisi evento {idx}: {exc}")
                continue

            summary["iteration"] = iteration
            summary["event_index"] = idx
            print(json.dumps(summary, indent=2, ensure_ascii=False))

            if save_file:
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with save_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        if args.iterations > 0 and iteration >= args.iterations:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
