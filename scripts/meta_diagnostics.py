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
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Mostra statistiche aggregate sulle entry disponibili",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        help="Salva le statistiche aggregate in un file JSON",
    )

    args = parser.parse_args()

    config = AIConfig()
    orchestrator = _build_orchestrator(config)

    if args.register_outcome:
        match_id, result = args.register_outcome
        register_outcome(orchestrator, match_id, float(result))

    show_stats(orchestrator, args.show)

    if args.aggregate or args.export_json:
        stats = orchestrator.feature_store.aggregate()
        if args.aggregate:
            if not stats:
                print("\nNessuna entry per statistiche aggregate.")
            else:
                print("\n📊 STATISTICHE AGGREGATE")
                print(f"   Totale entry: {stats['total_entries']}")
                print(f"   Con outcome: {stats['entries_with_outcome']} ({stats['outcome_ratio']:.1%})")
                print(f"   Probabilità media: {stats['avg_probability']:.3f}")
                if stats["probability_rmse"] is not None:
                    print(f"   RMSE probabilità vs outcome: {stats['probability_rmse']:.3f}")
                if stats["weights"]:
                    print("\n   Peso medio per modello:")
                    for model, values in stats["weights"].items():
                        print(f"      - {model}: avg={values['avg']:.3f} min={values['min']:.3f} max={values['max']:.3f}")
                if stats["context_features"]:
                    top_features = list(stats["context_features"].items())[:5]
                    print("\n   Prime feature di contesto (medie):")
                    for feature, values in top_features:
                        print(f"      - {feature}: avg={values['avg']:.3f}")
        if args.export_json:
            import json

            Path(args.export_json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print(f"\n📝 Statistiche aggregate salvate in {args.export_json}")


if __name__ == "__main__":
    main()
