"""
Sistema Predizioni Multi-Modello Intelligente
==============================================

Combina predizioni da tutte le AI disponibili per:
- Calcolare consensus score
- Identificare disaccordi (possibili value bet nascosti)
- Aumentare confidence quando tutti d'accordo
- Alert per opportunità speciali
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MultiModelConsensus:
    """
    Sistema che combina predizioni da tutte le AI per calcolare consensus.
    """
    
    def __init__(self):
        self.models_used = []
        self.consensus_history = []
    
    def analyze_consensus(
        self,
        predictions: Dict[str, Any],
        ai_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analizza consensus tra tutte le predizioni AI.
        
        Args:
            predictions: Dict con predizioni da vari modelli
            ai_results: Risultati completi dalla pipeline AI
            
        Returns:
            Dict con consensus score, agreement level, insights
        """
        try:
            # Raccogli tutte le predizioni disponibili
            all_predictions = {}
            
            # 1. Ensemble predictions
            if 'ensemble' in ai_results:
                ensemble = ai_results['ensemble']
                if 'model_predictions' in ensemble:
                    all_predictions['dixon_coles'] = ensemble['model_predictions'].get('dixon_coles')
                    all_predictions['xgboost'] = ensemble['model_predictions'].get('xgboost')
                    all_predictions['lstm'] = ensemble['model_predictions'].get('lstm')
                if 'probability' in ensemble:
                    all_predictions['ensemble_final'] = ensemble['probability']
            
            # 2. Bayesian Fusion
            if 'bayesian_fusion' in ai_results:
                bayesian = ai_results['bayesian_fusion']
                if 'probability' in bayesian:
                    all_predictions['bayesian'] = bayesian['probability']
            
            # 3. Calibrated probability
            if 'calibrated' in ai_results:
                calibrated = ai_results['calibrated']
                if 'prob_calibrated' in calibrated:
                    all_predictions['calibrated'] = calibrated['prob_calibrated']
            
            # 4. Sentiment-adjusted (se disponibile)
            if 'sentiment' in ai_results:
                sentiment = ai_results.get('sentiment', {})
                if 'adjusted_probability' in sentiment:
                    all_predictions['sentiment_adjusted'] = sentiment['adjusted_probability']
            
            # Filtra None e converte a float
            valid_predictions = {
                k: float(v) for k, v in all_predictions.items()
                if v is not None and isinstance(v, (int, float)) and 0 <= v <= 1
            }
            
            if not valid_predictions:
                return {
                    'consensus_score': 0.0,
                    'agreement_level': 'UNKNOWN',
                    'mean_prediction': 0.0,
                    'std_prediction': 0.0,
                    'models_count': 0,
                    'insights': []
                }
            
            # Calcola statistiche
            pred_values = list(valid_predictions.values())
            mean_pred = np.mean(pred_values)
            std_pred = np.std(pred_values)
            median_pred = np.median(pred_values)
            
            # Consensus score (0-1): più basso std = più alto consensus
            # Normalizza std (max std teorico = 0.5 per probabilità)
            max_std = 0.5
            consensus_score = max(0.0, 1.0 - (std_pred / max_std))
            
            # Agreement level
            if std_pred < 0.05:
                agreement_level = 'VERY_HIGH'
            elif std_pred < 0.10:
                agreement_level = 'HIGH'
            elif std_pred < 0.15:
                agreement_level = 'MEDIUM'
            elif std_pred < 0.20:
                agreement_level = 'LOW'
            else:
                agreement_level = 'VERY_LOW'
            
            # Identifica modelli outlier
            outliers = []
            for model, pred in valid_predictions.items():
                diff = abs(pred - mean_pred)
                if diff > 2 * std_pred and std_pred > 0.05:  # Solo se c'è variabilità
                    outliers.append({
                        'model': model,
                        'prediction': pred,
                        'diff_from_mean': diff,
                        'diff_percent': (diff / mean_pred * 100) if mean_pred > 0 else 0
                    })
            
            # Insights
            insights = []
            
            if consensus_score > 0.8:
                insights.append({
                    'type': 'HIGH_CONSENSUS',
                    'message': f'Tutti i modelli sono d\'accordo ({agreement_level})',
                    'confidence_boost': 10
                })
            elif consensus_score < 0.3:
                insights.append({
                    'type': 'LOW_CONSENSUS',
                    'message': f'Disaccordo tra modelli ({agreement_level}) - Possibile opportunità nascosta',
                    'confidence_penalty': -15,
                    'investigate': True
                })
            
            if outliers:
                outlier_names = [o['model'] for o in outliers]
                insights.append({
                    'type': 'OUTLIER_DETECTED',
                    'message': f'Modelli outlier: {", ".join(outlier_names)}',
                    'outliers': outliers,
                    'investigate': True
                })
            
            # Calcola weighted consensus (pesa per confidence se disponibile)
            weighted_pred = mean_pred
            if 'confidence' in ai_results:
                confidence = ai_results['confidence'].get('confidence_score', 50) / 100.0
                # Pesa di più i modelli quando confidence è alta
                weighted_pred = mean_pred * (0.5 + confidence * 0.5)
            
            result = {
                'consensus_score': float(consensus_score),
                'agreement_level': agreement_level,
                'mean_prediction': float(mean_pred),
                'median_prediction': float(median_pred),
                'std_prediction': float(std_pred),
                'weighted_prediction': float(weighted_pred),
                'models_count': len(valid_predictions),
                'models_used': list(valid_predictions.keys()),
                'predictions': valid_predictions,
                'outliers': outliers,
                'insights': insights,
                'recommendation': self._get_recommendation(consensus_score, std_pred, outliers)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating consensus: {e}", exc_info=True)
            return {
                'consensus_score': 0.0,
                'agreement_level': 'ERROR',
                'error': str(e)
            }
    
    def _get_recommendation(
        self,
        consensus_score: float,
        std_pred: float,
        outliers: List[Dict]
    ) -> str:
        """Genera raccomandazione basata su consensus."""
        if consensus_score > 0.8:
            return "HIGH_CONFIDENCE - Tutti i modelli concordano, alta affidabilità"
        elif consensus_score > 0.6:
            return "MODERATE_CONFIDENCE - Buon accordo tra modelli"
        elif consensus_score > 0.4:
            return "INVESTIGATE - Disaccordo moderato, valuta attentamente"
        elif consensus_score > 0.2:
            return "CAUTION - Alto disaccordo, possibile value bet nascosto o rischio"
        else:
            return "SKIP - Disaccordo estremo, evita questa scommessa"
    
    def should_boost_confidence(
        self,
        consensus_result: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Determina se boostare confidence basato su consensus.
        
        Returns:
            (should_boost, boost_amount)
        """
        consensus_score = consensus_result.get('consensus_score', 0.0)
        agreement_level = consensus_result.get('agreement_level', 'UNKNOWN')
        
        if agreement_level in ['VERY_HIGH', 'HIGH']:
            # Boost confidence se consensus alto
            boost = (consensus_score - 0.5) * 20  # Max +10 punti
            return True, max(0, min(boost, 15))
        elif agreement_level in ['VERY_LOW']:
            # Penalty se consensus basso
            penalty = (0.5 - consensus_score) * 20  # Max -15 punti
            return True, min(0, max(penalty, -20))
        
        return False, 0.0

