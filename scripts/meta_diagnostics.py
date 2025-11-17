#!/usr/bin/env python3
"""
Utility CLI per ispezionare il meta-feature store e aggiornare gli outcome.

Esempi:
    python scripts/meta_diagnostics.py --show 5
    python scripts/meta_diagnostics.py --register-outcome MATCH_ID 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_system.config import AIConfig
from ai_system.models.meta_learner import MetaLearner
from ai_system.meta import AdaptiveOrchestrator


def _build_orchestrator(config: AIConfig) -> AdaptiveOrchestrator:
    meta_config = {
        "store_path": str(Path(config.history_dir) / config.meta_store_filename),
        "max_entries": config.meta_store_max_entries,
        "exploration_rate": config.meta_exploration_rate,
        "reliability_decay": config.meta_reliability_decay,
        "bootstrap_window": config.meta_bootstrap_window,
    }
    meta_learner = MetaLearner(num_models=3)
    return AdaptiveOrchestrator(meta_learner=meta_learner, config=meta_config)


def show_stats(orchestrator: AdaptiveOrchestrator, limit: int):
    store = orchestrator.feature_store
    entries = store.load_recent_entries(limit)
    reliabilities = orchestrator.registry.snapshot_reliability()

    print(f"\n📁 Feature store: {store.filepath}")
    print(f"   Totale entry (approx): {len(store.load_recent_entries(0))}")
    print(f"\n📊 Reliability snapshot:")
    if reliabilities:
        for name, score in reliabilities.items():
            print(f"   - {name:12s}: {score:.3f}")
    else:
        print("   Nessuna metrica disponibile (store vuoto).")

    if not entries:
        print("\nNessuna entry registrata.")
        return

    print(f"\n🧾 Ultime {min(limit, len(entries))} entry:")
    for entry in entries[-limit:]:
        match_id = entry.get("match_id")
        probability = entry.get("probability")
        actual = entry.get("actual_outcome")
        timestamp = entry.get("timestamp")
        weights = entry.get("weights", {})
        print(f" - {match_id} | prob={probability:.3f} | actual={actual}")
        print(f"   weights: {', '.join(f'{k}:{v:.2f}' for k, v in weights.items())}")
        print(f"   ts: {timestamp}")


def register_outcome(orchestrator: AdaptiveOrchestrator, match_id: str, outcome: float):
    updated = orchestrator.register_outcome(match_id, outcome)
    if updated:
        print(f"✅ Outcome registrato per {match_id} (valore={outcome})")
    else:
        print(f"⚠️  Nessuna entry trovata per match_id={match_id}")


def main():
    parser = argparse.ArgumentParser(description="Meta Layer Diagnostics")
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Numero di entry recenti da mostrare",
    )
    parser.add_argument(
        "--register-outcome",
        nargs=2,
        metavar=("MATCH_ID", "RESULT"),
        help="Aggiorna l'outcome di una predizione (RESULT 0-1)",
    )

    args = parser.parse_args()

    config = AIConfig()
    orchestrator = _build_orchestrator(config)

    if args.register_outcome:
        match_id, result = args.register_outcome
        register_outcome(orchestrator, match_id, float(result))

    show_stats(orchestrator, args.show)


if __name__ == "__main__":
    main()
