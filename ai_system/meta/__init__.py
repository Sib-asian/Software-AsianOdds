"""
Meta-Adaptive Layer
===================

Utility classes che abilitano il layer meta-adattivo:
- Feature store per salvare features/risultati
- Registry dei modelli con affidabilità dinamica
- Performance tracker
- Meta optimizer per pesi dinamici
"""

from .adaptive_orchestrator import AdaptiveOrchestrator, build_match_id
from .feature_store import FeatureStore
from .model_registry import ModelRegistry, ModelDescriptor
from .performance_tracker import PerformanceTracker
from .meta_optimizer import MetaOptimizer
from .health import evaluate_meta_health
from .reports import summarize_meta_health, format_alert
from .outcome_manager import OutcomeManager, DataFreshnessManager
from .playbook import AlertPlaybook

__all__ = [
    "AdaptiveOrchestrator",
    "FeatureStore",
    "ModelRegistry",
    "ModelDescriptor",
    "PerformanceTracker",
    "MetaOptimizer",
    "evaluate_meta_health",
    "summarize_meta_health",
    "format_alert",
    "OutcomeManager",
    "DataFreshnessManager",
    "AlertPlaybook",
    "build_match_id",
]
