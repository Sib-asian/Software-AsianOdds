#!/usr/bin/env python3
"""
Meta health watcher.

Usage:
    python scripts/meta_alerts.py --aggregate --fail-on-alert --webhook-url https://hooks.slack.com/...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

from ai_system.config import AIConfig
from ai_system.models.meta_learner import MetaLearner
from ai_system.meta import AdaptiveOrchestrator, evaluate_meta_health
from ai_system.meta.reports import summarize_meta_health


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


def send_webhook(url: str, text: str):
    try:
        resp = requests.post(url, json={"text": text}, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[meta-alerts] Failed to send webhook: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Meta layer alert watcher")
    parser.add_argument("--limit", type=int, default=200, help="Numero di entry da considerare per il report")
    parser.add_argument("--webhook-url", type=str, default=os.getenv("META_ALERT_WEBHOOK"), help="Webhook HTTP per notifiche (es. Slack)")
    parser.add_argument("--fail-on-alert", action="store_true", help="Esce con codice 1 se esistono alert")
    parser.add_argument("--print-summary", action="store_true", help="Stampa il report completo in stdout")
    args = parser.parse_args()

    config = AIConfig()
    orchestrator = _build_orchestrator(config)
    health = evaluate_meta_health(
        store=orchestrator.feature_store,
        registry=orchestrator.registry,
        limit=args.limit,
        exploration_rate=orchestrator.meta_optimizer.exploration_rate,
    )

    summary_text = summarize_meta_health(health)
    if args.print-summary or True:
        print(summary_text)

    alerts = health.get("alerts") or []
    if alerts and args.webhook_url:
        send_webhook(args.webhook_url, summary_text)

    if alerts and args.fail_on_alert:
        sys.exit(1)


if __name__ == "__main__":
    main()
