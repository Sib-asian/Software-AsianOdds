import csv
import json
import time
from pathlib import Path

import pytest

from ai_system.config import AIConfig
from ai_system.meta import AlertPlaybook, OutcomeManager, FeatureStore


class DummyPipeline:
    def __init__(self):
        self.calls = []

    def register_outcome(self, match_id, outcome, metadata=None):
        self.calls.append((match_id, outcome, metadata or {}))


def make_config(tmp_path):
    cfg = AIConfig()
    cfg.history_dir = tmp_path
    cfg.meta_store_filename = "meta_feature_store.jsonl"
    cfg.outcomes_db = str(tmp_path / "outcomes.csv")
    cfg.outcome_status_db = str(tmp_path / "statuses.json")
    cfg.outcome_ttl_hours = 0.001
    return cfg


def seed_feature_store(store_path: Path, match_id: str):
    fs = FeatureStore(store_path)
    fs.record_prediction(
        match_id=match_id,
        match={"home": "Home", "away": "Away"},
        context_features={
            "data_availability": 0.8,
            "form_reliability": 0.7,
            "historical_matches": 0.3,
            "h2h_relevance": 0.5,
        },
        predictions={"dixon_coles": 0.6},
        weights={"dixon_coles": 1.0},
        probability=0.6,
        metadata={},
    )
    return fs


def test_outcome_manager_validates_and_applies(tmp_path):
    cfg = make_config(tmp_path)
    store_path = Path(cfg.history_dir) / cfg.meta_store_filename
    seed_feature_store(store_path, "2024-home-away")
    seed_feature_store(store_path, "2024-home2-away2")

    pipeline = DummyPipeline()
    manager = OutcomeManager(cfg, pipeline=pipeline)

    assert manager.record_outcome("2024-home-away", 1.0) is True
    assert pipeline.calls and pipeline.calls[-1][0] == "2024-home-away"

    # Missing match should be rejected
    assert manager.record_outcome("missing-match", 0.0) is False
    assert len(pipeline.calls) == 1

    # Add a pending entry and ensure apply_pending_outcomes processes it
    with open(cfg.outcomes_db, "a", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["2024-home2-away2", "0", time.time() - 10_000])

    manager.apply_pending_outcomes(ttl_hours=0)
    assert len(pipeline.calls) == 2


def test_alert_playbook_executes_mapped_actions(tmp_path, capsys):
    playbook_data = {
        "low_outcome_feedback": "echo ingest",
        "high_probability_rmse": ["echo retrain", "echo notify"],
    }
    playbook_path = tmp_path / "playbook.json"
    playbook_path.write_text(json.dumps(playbook_data))

    playbook = AlertPlaybook(playbook_path)
    alerts = [{"code": "low_outcome_feedback"}, {"code": "high_probability_rmse"}]

    actions = playbook.actions_for_alerts(alerts)
    assert actions == ["echo ingest", "echo retrain", "echo notify"]

    # dry_run should not raise or execute commands
    playbook.execute(alerts, dry_run=True)
    captured = capsys.readouterr()
    assert "DRY-RUN" in captured.out
