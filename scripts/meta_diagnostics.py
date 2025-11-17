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


def aggregate_stats(entries):
    if not entries:
        return {}

    total = len(entries)
    with_outcome = sum(1 for e in entries if e.get("actual_outcome") is not None)
    avg_prob = sum(e.get("probability", 0.0) for e in entries) / total
    prob_rmse = None
    diffs = [
        (e.get("probability", 0.0) - e.get("actual_outcome", 0.0)) ** 2
        for e in entries
        if e.get("actual_outcome") is not None
    ]
    if diffs:
        prob_rmse = (sum(diffs) / len(diffs)) ** 0.5

    weight_stats = {}
    prediction_stats = {}
    context_stats = {}

    for entry in entries:
        for model, weight in (entry.get("weights") or {}).items():
            bucket = weight_stats.setdefault(model, [])
            bucket.append(weight)
        for model, value in (entry.get("predictions") or {}).items():
            bucket = prediction_stats.setdefault(model, [])
            bucket.append(value)
        for feature, value in (entry.get("context_features") or {}).items():
            bucket = context_stats.setdefault(feature, [])
            bucket.append(value)

    def summarize(bucket_dict):
        return {
            key: {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for key, values in bucket_dict.items()
            if values
        }

    return {
        "total_entries": total,
        "entries_with_outcome": with_outcome,
        "outcome_ratio": with_outcome / total,
        "avg_probability": avg_prob,
        "probability_rmse": prob_rmse,
        "weights": summarize(weight_stats),
        "predictions": summarize(prediction_stats),
        "context_features": summarize(context_stats),
    }


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
        entries = orchestrator.feature_store.load_recent_entries(0)
        stats = aggregate_stats(entries)
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
