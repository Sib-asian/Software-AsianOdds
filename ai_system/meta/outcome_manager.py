from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .feature_store import FeatureStore
from ..config import AIConfig
# NOTE: Pipeline is imported lazily to avoid circular dependencies
AIPipelineType = "AIPipeline"


class DataFreshnessManager:
    def __init__(self, status_path: Path):
        self.status_path = Path(status_path)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.status_path.exists():
            self._write({})

    def _read(self) -> Dict[str, float]:
        try:
            return json.loads(self.status_path.read_text())
        except Exception:
            return {}

    def _write(self, data: Dict[str, float]):
        self.status_path.write_text(json.dumps(data, indent=2))

    def is_stale(self, match_id: str, ttl_seconds: float) -> bool:
        data = self._read()
        timestamp = data.get(match_id)
        if not timestamp:
            return True
        return (time.time() - float(timestamp)) > ttl_seconds

    def mark_processed(self, match_id: str):
        data = self._read()
        data[match_id] = time.time()
        self._write(data)


class OutcomeManager:
    def __init__(self, config: AIConfig, pipeline: Optional[AIPipelineType] = None):
        self.config = config
        if pipeline is None:
            from ..pipeline import AIPipeline

            self.pipeline = AIPipeline(config=config)
        else:
            self.pipeline = pipeline
        self.outcomes_path = Path(config.outcomes_db)
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.outcomes_path.exists():
            with self.outcomes_path.open("w", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["match_id", "outcome", "timestamp"])
        self.status = DataFreshnessManager(config.outcome_status_db)

    def record_outcome(self, match_id: str, outcome: float):
        outcome = max(0.0, min(1.0, float(outcome)))
        with self.outcomes_path.open("a", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([match_id, outcome, time.time()])
        self.pipeline.register_outcome(match_id, outcome, {"source": "manual"})
        self.status.mark_processed(match_id)

    def apply_pending_outcomes(self, ttl_hours: float = 168):
        """Apply outcomes from CSV that are not yet recorded in the meta store."""
        fresh_ttl = ttl_hours * 3600
        with self.outcomes_path.open("r", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                match_id = row["match_id"]
                if not self.status.is_stale(match_id, fresh_ttl):
                    continue
                outcome = float(row["outcome"])
                self.pipeline.register_outcome(match_id, outcome, {"source": "ingest"})
                self.status.mark_processed(match_id)
