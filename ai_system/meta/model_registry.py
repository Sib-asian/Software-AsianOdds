"""
Simple in-memory registry that keeps metadata and reliability scores
for each predictive model plugged into the adaptive layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelDescriptor:
    name: str
    model_type: str = "generic"
    tags: List[str] = field(default_factory=list)
    priority: int = 1
    cost: float = 1.0
    supports_live: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelDescriptor] = {}
        self._reliability: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Registry management
    # ------------------------------------------------------------------ #

    def register(self, descriptor: ModelDescriptor):
        self._models[descriptor.name] = descriptor
        self._reliability.setdefault(descriptor.name, 0.5)

    def ensure(
        self,
        name: str,
        model_type: str = "generic",
        tags: Optional[List[str]] = None,
        priority: int = 1,
        cost: float = 1.0,
        supports_live: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if name in self._models:
            # merge small metadata updates
            if metadata:
                merged = self._models[name].metadata
                merged.update(metadata)
            return
        descriptor = ModelDescriptor(
            name=name,
            model_type=model_type,
            tags=tags or [],
            priority=priority,
            cost=cost,
            supports_live=supports_live,
            metadata=metadata or {},
        )
        self.register(descriptor)

    def list_models(self) -> List[ModelDescriptor]:
        return list(self._models.values())

    # ------------------------------------------------------------------ #
    # Reliability tracking
    # ------------------------------------------------------------------ #

    def get_reliability(self, name: str) -> float:
        return self._reliability.get(name, 0.5)

    def snapshot_reliability(self) -> Dict[str, float]:
        return {name: round(score, 4) for name, score in self._reliability.items()}

    def update_reliability(self, name: str, score: float, decay: float = 0.9):
        """
        Update reliability score (EWMA) given a new score in [0, 1],
        where 1 = prediction perfetta.
        """
        score = max(0.0, min(1.0, score))
        current = self._reliability.get(name, 0.5)
        updated = decay * current + (1 - decay) * score
        self._reliability[name] = round(updated, 6)


__all__ = ["ModelRegistry", "ModelDescriptor"]
