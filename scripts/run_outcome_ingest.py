#!/usr/bin/env python3
"""
Batch job to ingest match outcomes from CSV into the meta layer.
"""

from __future__ import annotations

import argparse

from ai_system.config import AIConfig
from ai_system.meta import OutcomeManager
from ai_system.pipeline import AIPipeline


def main():
    parser = argparse.ArgumentParser(description="Apply pending match outcomes to the meta layer")
    parser.add_argument("--ttl-hours", type=float, default=None, help="Override TTL for outcome freshness")
    args = parser.parse_args()

    config = AIConfig()
    pipeline = AIPipeline(config=config)
    manager = OutcomeManager(config, pipeline=pipeline)
    manager.apply_pending_outcomes(ttl_hours=args.ttl_hours)


if __name__ == "__main__":
    main()
