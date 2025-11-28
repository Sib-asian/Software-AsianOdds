"""
Sistema Analisi Pattern con LLM
================================

Analizza pattern nelle scommesse vincenti usando LLM per generare
insights strategici automatici e suggerimenti.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from .llm_analyst import LLMAnalyst

logger = logging.getLogger(__name__)


class PatternAnalyzerLLM:
    """
    Analizza pattern nelle performance usando LLM per insights.
    """
    
    def __init__(self, llm_analyst: Optional[LLMAnalyst] = None):
        self.llm_analyst = llm_analyst
        self.analysis_cache = {}
    
    def analyze_performance_patterns(
        self,
        betting_results: List[Dict[str, Any]],
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analizza pattern nelle performance e genera insights con LLM.
        
        Args:
            betting_results: Lista di risultati scommesse
            days: Giorni da analizzare
            
        Returns:
            Dict con pattern, insights, raccomandazioni
        """
        try:
            if not betting_results:
                return {
                    'patterns': {},
                    'insights': [],
                    'recommendations': [],
                    'summary': 'Nessun dato disponibile per analisi'
                }
            
            # Filtra per periodo
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_results = [
                r for r in betting_results
                if isinstance(r.get('timestamp'), datetime) and r['timestamp'] >= cutoff_date
            ]
            
            if not recent_results:
                return {
                    'patterns': {},
                    'insights': [],
                    'recommendations': [],
                    'summary': f'Nessun dato negli ultimi {days} giorni'
                }
            
            # Calcola statistiche base
            total_bets = len(recent_results)
            wins = sum(1 for r in recent_results if r.get('outcome') == 'W')
            losses = sum(1 for r in recent_results if r.get('outcome') == 'L')
            win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
            
            # Pattern per lega
            league_stats = self._calculate_league_stats(recent_results)
            
            # Pattern per market
            market_stats = self._calculate_market_stats(recent_results)
            
            # Pattern per orario
            time_stats = self._calculate_time_stats(recent_results)
            
            # Pattern per tipo (pre-match vs live)
            type_stats = self._calculate_type_stats(recent_results)
            
            # Prepara dati per LLM
            analysis_data = {
                'total_bets': total_bets,
                'win_rate': win_rate,
                'wins': wins,
                'losses': losses,
                'league_stats': league_stats,
                'market_stats': market_stats,
                'time_stats': time_stats,
                'type_stats': type_stats
            }
            
            # Genera insights con LLM (se disponibile)
            insights = []
            recommendations = []
            
            if self.llm_analyst:
                try:
                    llm_insights = self._generate_llm_insights(analysis_data)
                    insights.extend(llm_insights.get('insights', []))
                    recommendations.extend(llm_insights.get('recommendations', []))
                except Exception as e:
                    logger.warning(f"⚠️ LLM insights generation failed: {e}")
            
            # Aggiungi insights automatici
            auto_insights = self._generate_auto_insights(analysis_data)
            insights.extend(auto_insights)
            
            return {
                'patterns': {
                    'league': league_stats,
                    'market': market_stats,
                    'time': time_stats,
                    'type': type_stats
                },
                'insights': insights,
                'recommendations': recommendations,
                'summary': {
                    'total_bets': total_bets,
                    'win_rate': win_rate,
                    'period_days': days
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing patterns: {e}", exc_info=True)
            return {
                'error': str(e),
                'patterns': {},
                'insights': [],
                'recommendations': []
            }
    
    def _calculate_league_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calcola statistiche per lega."""
        league_data = {}
        
        for result in results:
            league = result.get('league', 'Unknown')
            if league not in league_data:
                league_data[league] = {'total': 0, 'wins': 0, 'losses': 0}
            
            league_data[league]['total'] += 1
            if result.get('outcome') == 'W':
                league_data[league]['wins'] += 1
            elif result.get('outcome') == 'L':
                league_data[league]['losses'] += 1
        
        # Calcola win rate
        stats = {}
        for league, data in league_data.items():
            win_rate = (data['wins'] / data['total'] * 100) if data['total'] > 0 else 0
            stats[league] = {
                'total': data['total'],
                'wins': data['wins'],
                'losses': data['losses'],
                'win_rate': win_rate
            }
        
        return stats
    
    def _calculate_market_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calcola statistiche per market."""
        market_data = {}
        
        for result in results:
            market = result.get('market', '1X2')
            if market not in market_data:
                market_data[market] = {'total': 0, 'wins': 0, 'losses': 0}
            
            market_data[market]['total'] += 1
            if result.get('outcome') == 'W':
                market_data[market]['wins'] += 1
            elif result.get('outcome') == 'L':
                market_data[market]['losses'] += 1
        
        stats = {}
        for market, data in market_data.items():
            win_rate = (data['wins'] / data['total'] * 100) if data['total'] > 0 else 0
            stats[market] = {
                'total': data['total'],
                'wins': data['wins'],
                'losses': data['losses'],
                'win_rate': win_rate
            }
        
        return stats
    
    def _calculate_time_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calcola statistiche per orario."""
        time_data = {
            'morning': {'total': 0, 'wins': 0},  # 6-12
            'afternoon': {'total': 0, 'wins': 0},  # 12-18
            'evening': {'total': 0, 'wins': 0},  # 18-24
            'night': {'total': 0, 'wins': 0}  # 0-6
        }
        
        for result in results:
            timestamp = result.get('timestamp')
            if not isinstance(timestamp, datetime):
                continue
            
            hour = timestamp.hour
            
            if 6 <= hour < 12:
                period = 'morning'
            elif 12 <= hour < 18:
                period = 'afternoon'
            elif 18 <= hour < 24:
                period = 'evening'
            else:
                period = 'night'
            
            time_data[period]['total'] += 1
            if result.get('outcome') == 'W':
                time_data[period]['wins'] += 1
        
        stats = {}
        for period, data in time_data.items():
            win_rate = (data['wins'] / data['total'] * 100) if data['total'] > 0 else 0
            stats[period] = {
                'total': data['total'],
                'wins': data['wins'],
                'win_rate': win_rate
            }
        
        return stats
    
    def _calculate_type_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calcola statistiche per tipo (pre-match vs live)."""
        type_data = {
            'pre_match': {'total': 0, 'wins': 0},
            'live': {'total': 0, 'wins': 0}
        }
        
        for result in results:
            match_type = result.get('match_type', 'pre_match')
            if match_type not in type_data:
                match_type = 'pre_match'
            
            type_data[match_type]['total'] += 1
            if result.get('outcome') == 'W':
                type_data[match_type]['wins'] += 1
        
        stats = {}
        for match_type, data in type_data.items():
            win_rate = (data['wins'] / data['total'] * 100) if data['total'] > 0 else 0
            stats[match_type] = {
                'total': data['total'],
                'wins': data['wins'],
                'win_rate': win_rate
            }
        
        return stats
    
    def _generate_auto_insights(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera insights automatici basati su pattern."""
        insights = []
        
        # Analizza leghe
        league_stats = data.get('league_stats', {})
        if league_stats:
            best_league = max(
                league_stats.items(),
                key=lambda x: x[1].get('win_rate', 0)
            )
            worst_league = min(
                league_stats.items(),
                key=lambda x: x[1].get('win_rate', 0) if x[1].get('total', 0) >= 5 else 100
            )
            
            if best_league[1].get('total', 0) >= 5:
                insights.append({
                    'type': 'BEST_LEAGUE',
                    'league': best_league[0],
                    'win_rate': best_league[1]['win_rate'],
                    'message': f"✅ Miglior performance: {best_league[0]} (Win rate: {best_league[1]['win_rate']:.1f}%)"
                })
            
            if worst_league[1].get('total', 0) >= 5 and worst_league[1].get('win_rate', 0) < 40:
                insights.append({
                    'type': 'WORST_LEAGUE',
                    'league': worst_league[0],
                    'win_rate': worst_league[1]['win_rate'],
                    'message': f"⚠️ Performance bassa: {worst_league[0]} (Win rate: {worst_league[1]['win_rate']:.1f}%)"
                })
        
        # Analizza markets
        market_stats = data.get('market_stats', {})
        if market_stats:
            for market, stats in market_stats.items():
                if stats.get('total', 0) >= 5:
                    win_rate = stats.get('win_rate', 0)
                    if win_rate < 40:
                        insights.append({
                            'type': 'POOR_MARKET',
                            'market': market,
                            'win_rate': win_rate,
                            'message': f"⚠️ Market '{market}' ha win rate basso: {win_rate:.1f}%"
                        })
        
        # Analizza orari
        time_stats = data.get('time_stats', {})
        if time_stats:
            best_time = max(
                time_stats.items(),
                key=lambda x: x[1].get('win_rate', 0)
            )
            if best_time[1].get('total', 0) >= 3:
                insights.append({
                    'type': 'BEST_TIME',
                    'period': best_time[0],
                    'win_rate': best_time[1]['win_rate'],
                    'message': f"✅ Miglior orario: {best_time[0]} (Win rate: {best_time[1]['win_rate']:.1f}%)"
                })
        
        return insights
    
    def _generate_llm_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera insights usando LLM."""
        if not self.llm_analyst:
            return {'insights': [], 'recommendations': []}
        
        try:
            # Prepara prompt per LLM
            prompt = f"""
Analizza questi pattern di performance betting e genera insights strategici:

Statistiche Generali:
- Totale scommesse: {data['total_bets']}
- Win rate: {data['win_rate']:.1f}%
- Vittorie: {data['wins']}
- Sconfitte: {data['losses']}

Performance per Lega:
{json.dumps(data['league_stats'], indent=2)}

Performance per Market:
{json.dumps(data['market_stats'], indent=2)}

Performance per Orario:
{json.dumps(data['time_stats'], indent=2)}

Performance per Tipo:
{json.dumps(data['type_stats'], indent=2)}

Genera:
1. 3-5 insights chiave sui pattern
2. 3-5 raccomandazioni strategiche concrete

Formato JSON:
{{
    "insights": [
        {{"type": "...", "message": "..."}},
        ...
    ],
    "recommendations": [
        {{"action": "...", "reason": "..."}},
        ...
    ]
}}
"""
            
            response = self.llm_analyst.analyze_patterns(prompt)
            
            # Parse response
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                # Prova a parse JSON
                try:
                    return json.loads(response)
                except:
                    return {'insights': [], 'recommendations': []}
            else:
                return {'insights': [], 'recommendations': []}
                
        except Exception as e:
            logger.warning(f"⚠️ LLM insight generation error: {e}")
            return {'insights': [], 'recommendations': []}

