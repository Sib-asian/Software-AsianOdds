#!/usr/bin/env python3
"""
Generate a simple static HTML report of meta health and store summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_system.config import AIConfig
from ai_system.meta import AdaptiveOrchestrator, evaluate_meta_health
from ai_system.models.meta_learner import MetaLearner


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


def main():
    parser = argparse.ArgumentParser(description="Generate meta layer dashboard")
    parser.add_argument("--output", type=str, default="meta_report.html", help="Output HTML file")
    parser.add_argument("--limit", type=int, default=200, help="Number of entries to analyze")
    args = parser.parse_args()

    config = AIConfig()
    orchestrator = _build_orchestrator(config)
    health = evaluate_meta_health(
        store=orchestrator.feature_store,
        registry=orchestrator.registry,
        limit=args.limit,
        exploration_rate=orchestrator.meta_optimizer.exploration_rate,
    )

    html = f"""
    <html>
    <head>
        <title>Meta Layer Health Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; }}
            .alert {{ margin: 0.5rem 0; }}
            .alert.INFO {{ color: #333; }}
            .alert.WARNING {{ color: #b58900; }}
            .alert.CRITICAL {{ color: #cb4b16; }}
            pre {{ background: #f7f7f7; padding: 1rem; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>Meta Layer Health</h1>
        <p><strong>Store:</strong> {health.get("store_path")}</p>
        <p><strong>Entries:</strong> {health.get("entries")}</p>
        <p><strong>Exploration rate:</strong> {health.get("exploration_rate")}</p>
        <h2>Summary</h2>
        <pre>{json.dumps(health.get("summary"), indent=2)}</pre>
        <h2>Window Summaries</h2>
        <pre>{json.dumps(health.get("summary"), indent=2)}</pre>
        <h2>Reliability</h2>
        <pre>{json.dumps(health.get("reliability"), indent=2)}</pre>
        <h2>Alerts</h2>
        {"".join(f"<div class='alert {a['level'].upper()}'><strong>{a['code']}</strong>: {a['message']} ({a.get('action','')})</div>" for a in health.get("alerts") or []) or "<p>Nessun alert.</p>"}
    </body>
    </html>
    """
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"[meta-dashboard] Report generato in {args.output}")


if __name__ == "__main__":
    main()
