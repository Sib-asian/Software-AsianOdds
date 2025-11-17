"""
Utility per integrare rapidamente il modello di Bayesian Fusion
all'interno della pipeline di analisi e delle notifiche.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from ai_system.models.bayesian_fusion import BayesianFusionModel

_DEFAULT_FUSION_MODEL = BayesianFusionModel()


def attach_bayesian_fusion(
    analysis_result: Dict,
    signals: Mapping[str, float],
    league: str = "GLOBAL",
    market: Optional[str] = None,
    fusion_model: Optional[BayesianFusionModel] = None,
) -> Dict:
    """
    Calcola la fusione bayesiana dei segnali e arricchisce il dizionario analisi.

    Args:
        analysis_result: dizionario prodotto dalla pipeline (sarà copiato)
        signals: mappa {nome_fonte: probabilità in [0,1]}
        league: lega/campionato (per il tracking della reliability)
        market: mercato specifico (es. "1X2", "Over25"). Default: analysis_result['market']
        fusion_model: istanza personalizzata, altrimenti usa singleton interno

    Returns:
        Nuovo dict con chiave 'bayesian_fusion' contenente output dettagliato.
    """
    if not signals:
        return dict(analysis_result)

    model = fusion_model or _DEFAULT_FUSION_MODEL
    target_market = market or analysis_result.get("market", "1X2")
    fusion = model.fuse(signals, league=league, market=str(target_market))

    enriched = dict(analysis_result)
    enriched["bayesian_fusion"] = fusion
    return enriched


def update_bayesian_reliability(
    source: str,
    outcome: bool,
    league: str = "GLOBAL",
    market: str = "1X2",
    weight: float = 1.0,
    fusion_model: Optional[BayesianFusionModel] = None,
) -> None:
    """
    Aggiorna la reliability di una fonte dopo aver conosciuto l'esito reale.
    Può essere richiamata da un job giornaliero/post-match.
    """
    model = fusion_model or _DEFAULT_FUSION_MODEL
    model.update_reliability(
        league=league,
        market=market,
        source=source,
        success=outcome,
        weight=weight,
    )
