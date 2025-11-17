"""
Bayesian Fusion Model
=====================

Combina più segnali di probabilità (manuali, modelli ML, sentiment, ecc.)
utilizzando un approccio bayesiano gerarchico per produrre una stima finale
più robusta e un intervallo di confidenza esplicito.

L'obiettivo è dare maggior peso alle fonti storicamente affidabili per
lega/mercato, mantenendo comunque tutte le informazioni disponibili.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Mapping, Tuple, Optional, List
import math

from scipy.stats import beta


@dataclass
class ReliabilityStats:
    """Statistiche Beta per una singola fonte."""

    alpha: float = 2.0
    beta: float = 2.0

    def mean(self) -> float:
        total = self.alpha + self.beta
        if total <= 0:
            return 0.5
        return self.alpha / total

    def strength(self) -> float:
        return max(self.alpha + self.beta, 1e-3)

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha += max(weight, 0.0)
        else:
            self.beta += max(weight, 0.0)


class BayesianFusionModel:
    """
    Modello che fonde probabilità multiple sfruttando prior e confidenza storica.

    Args:
        default_strength: intensità di evidenza da assegnare al combinatore
        min_weight: peso minimo di una fonte (per non annullare segnali nuovi)
        global_prior: tuple (alpha, beta) per il prior di fallback
    """

    def __init__(
        self,
        default_strength: float = 20.0,
        min_weight: float = 0.05,
        global_prior: Tuple[float, float] = (2.0, 2.0),
    ):
        self.default_strength = default_strength
        self.min_weight = min_weight
        self.global_prior = global_prior
        self._stats: Dict[Tuple[str, str, str], ReliabilityStats] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        signals: Mapping[str, float],
        league: str = "GLOBAL",
        market: str = "1X2",
        fallback_prob: Optional[float] = None,
    ) -> Dict:
        """
        Combina i segnali e restituisce probabilità finale + intervallo di confidenza.
        """
        if not signals:
            raise ValueError("signals non può essere vuoto")

        weighted_sum = 0.0
        total_weight = 0.0
        breakdown: List[Dict[str, float]] = []

        for source, prob in signals.items():
            prob = float(max(0.0, min(1.0, prob)))
            stats = self._get_stats(league, market, source)
            reliability = max(stats.mean(), self.min_weight)
            weighted_sum += prob * reliability
            total_weight += reliability
            breakdown.append(
                {
                    "source": source,
                    "probability": prob,
                    "weight": reliability,
                }
            )

        if total_weight == 0:
            if fallback_prob is None:
                raise ValueError("Nessuna informazione utile per il fuse")
            fused_prob = fallback_prob
            evidence_strength = self.default_strength
        else:
            fused_prob = weighted_sum / total_weight
            evidence_strength = self.default_strength * (total_weight / len(signals))

        alpha_prior, beta_prior = self.global_prior
        alpha_post = alpha_prior + fused_prob * evidence_strength
        beta_post = beta_prior + (1.0 - fused_prob) * evidence_strength
        ci_low, ci_high = beta.interval(0.95, alpha_post, beta_post)
        if math.isnan(ci_low) or math.isnan(ci_high):
            ci_low, ci_high = fused_prob - 0.05, fused_prob + 0.05

        ci_low = max(0.0, ci_low)
        ci_high = min(1.0, ci_high)
        confidence = self._confidence_from_interval(ci_low, ci_high)

        return {
            "probability": fused_prob,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": confidence,
            "sources": breakdown,
        }

    def update_reliability(
        self,
        league: str,
        market: str,
        source: str,
        success: bool,
        weight: float = 1.0,
    ) -> None:
        """Aggiorna la confidenza di una fonte in base all'esito reale."""
        stats = self._get_stats(league, market, source)
        stats.update(success=success, weight=weight)

    def get_state(self) -> Dict:
        """Esporta lo stato interno (utile per salvataggio su file/cache)."""
        return {
            "default_strength": self.default_strength,
            "min_weight": self.min_weight,
            "global_prior": self.global_prior,
            "stats": {
                f"{league}|{market}|{source}": asdict(stats)
                for (league, market, source), stats in self._stats.items()
            },
        }

    @classmethod
    def from_state(cls, state: Dict) -> "BayesianFusionModel":
        """Ricostruisce il modello da uno stato precedentemente salvato."""
        model = cls(
            default_strength=state.get("default_strength", 20.0),
            min_weight=state.get("min_weight", 0.05),
            global_prior=tuple(state.get("global_prior", (2.0, 2.0))),
        )
        for key, stats_dict in state.get("stats", {}).items():
            league, market, source = key.split("|", 2)
            model._stats[(league, market, source)] = ReliabilityStats(
                alpha=stats_dict.get("alpha", 2.0),
                beta=stats_dict.get("beta", 2.0),
            )
        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_stats(self, league: str, market: str, source: str) -> ReliabilityStats:
        key = (league.upper(), market.upper(), source.lower())
        if key not in self._stats:
            self._stats[key] = ReliabilityStats()
        return self._stats[key]

    @staticmethod
    def _confidence_from_interval(ci_low: float, ci_high: float) -> float:
        """Trasforma l'ampiezza dell'intervallo credibile in un punteggio 0-100."""
        width = max(ci_high - ci_low, 1e-6)
        confidence = max(0.0, 1.0 - width) * 120  # scala empirica
        return max(0.0, min(100.0, confidence))
