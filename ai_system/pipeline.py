"""
AI Pipeline - Orchestratore Principale
=======================================

Integra tutti i 7 blocchi AI in una pipeline coerente e facile da usare.

Flusso:
BLOCCO 0 (API Data Engine) →
BLOCCO 1 (Probability Calibrator) →
BLOCCO 2 (Confidence Scorer) →
BLOCCO 3 (Value Detector) →
BLOCCO 4 (Smart Kelly Optimizer) →
BLOCCO 5 (Risk Manager) →
BLOCCO 6 (Odds Movement Tracker) →
DECISIONE FINALE

Usage:
    pipeline = AIPipeline()
    result = pipeline.analyze(match_data, odds_data, bankroll)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from .config import AIConfig
from .blocco_0_api_engine import APIDataEngine
from .blocco_1_calibrator import ProbabilityCalibrator
from .blocco_2_confidence import ConfidenceScorer
from .blocco_3_value_detector import ValueDetector
from .blocco_4_kelly import SmartKellyOptimizer
from .blocco_5_risk_manager import RiskManager
from .blocco_6_odds_tracker import OddsMovementTracker
from .models.ensemble import EnsembleMetaModel

logger = logging.getLogger(__name__)


class AIPipeline:
    """
    Pipeline AI completa per analisi betting.

    Integra tutti i 7 blocchi e fornisce un'interfaccia semplice.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Inizializza pipeline.

        Args:
            config: Configurazione AI (usa default se None)
        """
        self.config = config or AIConfig()

        # Initialize all blocks
        logger.info("🚀 Initializing AI Pipeline...")

        self.api_engine = APIDataEngine(self.config)
        self.calibrator = ProbabilityCalibrator(self.config)
        self.confidence_scorer = ConfidenceScorer(self.config)
        self.value_detector = ValueDetector(self.config)
        self.kelly_optimizer = SmartKellyOptimizer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.odds_tracker = OddsMovementTracker(self.config)

        # Initialize Ensemble Meta-Model (if enabled)
        self.ensemble = None
        if self.config.use_ensemble:
            try:
                logger.info("🤖 Initializing Ensemble Meta-Model...")
                self.ensemble = EnsembleMetaModel(config={'models_dir': self.config.ensemble_models_dir})

                # Load trained models if requested
                if self.config.ensemble_load_models:
                    self.ensemble.load_models(self.config.ensemble_models_dir)

                logger.info("   ✅ Ensemble initialized")
            except Exception as e:
                logger.warning(f"   ⚠️  Ensemble initialization failed: {e}")
                logger.warning("   → Falling back to Dixon-Coles only")
                self.ensemble = None

        # Pipeline state
        self.last_analysis = None
        self.analysis_history = []

        logger.info("✅ AI Pipeline initialized successfully")
        logger.info(f"   Configuration: {self.config.log_level}")
        logger.info(f"   Models dir: {self.config.models_dir}")
        logger.info(f"   Ensemble enabled: {self.config.use_ensemble and self.ensemble is not None}")

    def analyze(
        self,
        match: Dict[str, Any],
        prob_dixon_coles: float,
        odds_data: Dict[str, Any],
        bankroll: float,
        portfolio_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analisi completa di un match attraverso tutta la pipeline AI.

        Args:
            match: Info match (home, away, league, date, etc)
            prob_dixon_coles: Probabilità raw dal modello Dixon-Coles
            odds_data: Dati quote (current_odds, odds_history, etc)
            bankroll: Bankroll attuale
            portfolio_state: Stato portfolio (active_bets, etc)

        Returns:
            Dizionario completo con:
            - Tutti i risultati intermedi di ogni blocco
            - Decisione finale (BET/SKIP/WATCH)
            - Stake raccomandato
            - Timing ottimale
            - Reasoning dettagliato
        """
        analysis_start = datetime.now()
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 ANALYZING: {match.get('home')} vs {match.get('away')}")
        logger.info(f"{'='*70}")

        if portfolio_state is None:
            portfolio_state = {"bankroll": bankroll, "active_bets": []}

        try:
            # ========================================
            # BLOCCO 0: API Data Engine
            # ========================================
            logger.info("\n📡 BLOCCO 0: Collecting API data...")
            api_context = self.api_engine.collect(match)

            logger.info(
                f"   ✓ Data quality: {api_context['metadata']['data_quality']:.0%}"
            )
            logger.info(
                f"   ✓ API calls: {api_context['metadata']['api_calls_used']}"
            )

            # ========================================
            # ENSEMBLE META-MODEL (optional)
            # ========================================
            ensemble_result = None
            if self.ensemble is not None:
                logger.info("\n🤖 ENSEMBLE: Combining models...")
                try:
                    # Get match history for LSTM (if available)
                    match_history = match.get('history', [])

                    ensemble_result = self.ensemble.predict(
                        match_data=match,
                        prob_dixon_coles=prob_dixon_coles,
                        api_context=api_context,
                        match_history=match_history
                    )

                    # Use ensemble probability instead of Dixon-Coles alone
                    prob_to_use = ensemble_result['probability']

                    logger.info(
                        f"   ✓ Ensemble: {prob_to_use:.1%} (DC: {prob_dixon_coles:.1%})"
                    )
                    logger.info(
                        f"   ✓ Confidence: {ensemble_result['confidence']:.0f}/100"
                    )
                    logger.info(
                        f"   ✓ Uncertainty: {ensemble_result['uncertainty']:.3f}"
                    )
                    logger.info(
                        f"   ✓ Dominant: {ensemble_result['breakdown']['summary']['dominant_model']}"
                    )

                except Exception as e:
                    logger.warning(f"   ⚠️  Ensemble prediction failed: {e}")
                    logger.warning("   → Using Dixon-Coles only")
                    prob_to_use = prob_dixon_coles
            else:
                # No ensemble, use Dixon-Coles directly
                prob_to_use = prob_dixon_coles

            # ========================================
            # BLOCCO 1: Probability Calibrator
            # ========================================
            logger.info("\n🎯 BLOCCO 1: Calibrating probability...")

            calibration_context = {
                "league": match.get("league"),
                "market": odds_data.get("market", "1x2"),
                "data_quality": api_context["metadata"]["data_quality"],
                "api_context": {
                    "injuries_home": api_context["home_context"]["data"].get("injuries", []),
                    "injuries_away": api_context["away_context"]["data"].get("injuries", []),
                    "form_home": api_context["home_context"]["data"].get("form", "DDDDD"),
                    "form_away": api_context["away_context"]["data"].get("form", "DDDDD"),
                    "xg_home_last5": api_context["match_data"].get("xg_home", 8.0),
                    "xg_away_last5": api_context["match_data"].get("xg_away", 7.0),
                    "xga_home_last5": api_context["match_data"].get("xga_home", 5.0),
                    "xga_away_last5": api_context["match_data"].get("xga_away", 6.0),
                    "lineup_quality_home": api_context["match_data"].get("lineup_home", 0.85),
                    "lineup_quality_away": api_context["match_data"].get("lineup_away", 0.85),
                }
            }

            calibrated_result = self.calibrator.calibrate(
                prob_to_use,
                calibration_context
            )

            logger.info(
                f"   ✓ Probability: {prob_to_use:.1%} → {calibrated_result['prob_calibrated']:.1%}"
            )
            logger.info(
                f"   ✓ Shift: {calibrated_result['calibration_shift']:+.1%}"
            )

            # ========================================
            # BLOCCO 2: Confidence Scorer
            # ========================================
            logger.info("\n🔍 BLOCCO 2: Scoring confidence...")

            confidence_context = {
                "odds_current": odds_data.get("odds_current", 2.0),
                "odds_history": odds_data.get("odds_history", []),
                "historical_accuracy": odds_data.get("historical_accuracy", 0.70),
                "similar_bets_roi": odds_data.get("similar_bets_roi", 0.0),
                "league_quality": self._get_league_quality(match.get("league", "")),
                "market_liquidity": 0.8,  # Default
                "red_flags": [],
                "green_flags": []
            }

            confidence_result = self.confidence_scorer.score(
                calibrated_result,
                api_context,
                confidence_context
            )

            logger.info(
                f"   ✓ Confidence: {confidence_result['confidence_score']:.0f}/100 "
                f"({confidence_result['confidence_level']})"
            )

            # ========================================
            # BLOCCO 3: Value Detector
            # ========================================
            logger.info("\n💎 BLOCCO 3: Detecting value...")

            value_context = {
                "similar_bets_roi": odds_data.get("similar_bets_roi", 0.0),
                "similar_bets_count": odds_data.get("similar_bets_count", 0),
                "similar_bets_winrate": odds_data.get("similar_bets_winrate", 0.5),
                "league_quality": self._get_league_quality(match.get("league", "")),
                "market_efficiency": 0.75,  # Default
            }

            # Merge calibrated prob into value result for detector
            value_detector_input = {
                **calibrated_result,
                "data_quality": api_context["metadata"]["data_quality"]
            }

            value_result = self.value_detector.detect(
                value_detector_input,
                confidence_result,
                odds_data,
                value_context
            )

            logger.info(
                f"   ✓ Value: {value_result['value_score']:.0f}/100 "
                f"({value_result['value_type']})"
            )
            logger.info(
                f"   ✓ EV: {value_result['expected_value']:+.1%}"
            )

            # ========================================
            # BLOCCO 4: Smart Kelly Optimizer
            # ========================================
            logger.info("\n💰 BLOCCO 4: Optimizing stake...")

            # Merge results for Kelly
            kelly_input = {
                **value_detector_input,
                **value_result
            }

            kelly_result = self.kelly_optimizer.optimize(
                kelly_input,
                confidence_result,
                odds_data.get("odds_current", 2.0),
                bankroll,
                portfolio_state
            )

            logger.info(
                f"   ✓ Stake: €{kelly_result['optimal_stake']:.2f} "
                f"({kelly_result['stake_percentage']:.1f}% bankroll)"
            )
            logger.info(
                f"   ✓ Kelly fraction: {kelly_result['kelly_fraction']:.2f}"
            )

            # ========================================
            # BLOCCO 5: Risk Manager
            # ========================================
            logger.info("\n🛡️  BLOCCO 5: Risk management...")

            risk_decision = self.risk_manager.decide(
                value_result,
                confidence_result,
                kelly_result,
                match,
                portfolio_state
            )

            logger.info(
                f"   ✓ Decision: {risk_decision['decision']} "
                f"(Priority: {risk_decision['priority']})"
            )
            logger.info(
                f"   ✓ Final stake: €{risk_decision['final_stake']:.2f}"
            )

            if risk_decision["red_flags"]:
                logger.warning(f"   ⚠️  Red flags: {len(risk_decision['red_flags'])}")
                for flag in risk_decision["red_flags"][:3]:
                    logger.warning(f"      - {flag}")

            # ========================================
            # BLOCCO 6: Odds Movement Tracker
            # ========================================
            logger.info("\n📈 BLOCCO 6: Analyzing odds movement...")

            timing_result = self.odds_tracker.monitor(
                match,
                risk_decision,
                odds_data.get("odds_current", 2.0),
                odds_data.get("odds_history", []),
                odds_data.get("time_to_kickoff_hours", 24.0)
            )

            logger.info(
                f"   ✓ Timing: {timing_result['timing_recommendation']} "
                f"(Urgency: {timing_result['urgency']})"
            )

            if timing_result.get("sharp_money_detected"):
                logger.info("   ✓ Sharp money detected!")

            # ========================================
            # FINAL RESULT
            # ========================================
            analysis_time = (datetime.now() - analysis_start).total_seconds()

            final_result = {
                # Match info
                "match": match,

                # All block results
                "api_context": api_context,
                "ensemble": ensemble_result,  # NEW: Ensemble results
                "calibrated": calibrated_result,
                "confidence": confidence_result,
                "value": value_result,
                "kelly": kelly_result,
                "risk_decision": risk_decision,
                "timing": timing_result,

                # Final decision (easy access)
                "final_decision": {
                    "action": risk_decision["decision"],
                    "stake": risk_decision["final_stake"],
                    "timing": timing_result["timing_recommendation"],
                    "priority": risk_decision["priority"],
                    "urgency": timing_result["urgency"]
                },

                # Summary
                "summary": {
                    "probability": calibrated_result["prob_calibrated"],
                    "confidence": confidence_result["confidence_score"],
                    "value_score": value_result["value_score"],
                    "expected_value": value_result["expected_value"],
                    "stake": risk_decision["final_stake"],
                    "odds": odds_data.get("odds_current", 2.0),
                    "potential_profit": risk_decision["final_stake"] * (odds_data.get("odds_current", 2.0) - 1),
                },

                # Metadata
                "metadata": {
                    "analysis_time_seconds": analysis_time,
                    "timestamp": datetime.now().isoformat(),
                    "api_calls_used": api_context["metadata"]["api_calls_used"],
                    "models_used": self._get_models_status(),
                    "ensemble_enabled": self.ensemble is not None  # NEW
                }
            }

            # Save to history
            self.last_analysis = final_result
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "match": f"{match.get('home')} vs {match.get('away')}",
                "decision": risk_decision["decision"],
                "stake": risk_decision["final_stake"]
            })

            # Print final summary
            self._print_summary(final_result)

            logger.info(f"\n{'='*70}")
            logger.info(f"✅ Analysis completed in {analysis_time:.2f}s")
            logger.info(f"{'='*70}\n")

            return final_result

        except Exception as e:
            logger.error(f"❌ Error in pipeline analysis: {e}", exc_info=True)
            raise

    def _get_league_quality(self, league: str) -> float:
        """Map league to quality score"""
        league_lower = league.lower()
        scores = {
            "serie a": 0.90,
            "premier league": 0.95,
            "la liga": 0.92,
            "bundesliga": 0.90,
            "ligue 1": 0.85,
            "champions league": 1.00,
            "serie b": 0.70,
        }
        for league_name, score in scores.items():
            if league_name in league_lower:
                return score
        return 0.75  # Default

    def _get_models_status(self) -> Dict[str, str]:
        """Get status of all models"""
        return {
            "calibrator": "trained" if self.calibrator.is_trained else "rule_based",
            "confidence": "trained" if self.confidence_scorer.is_trained else "rule_based",
            "value_detector": "trained" if self.value_detector.is_trained else "rule_based",
            "odds_tracker": "trained" if self.odds_tracker.is_trained else "rule_based",
        }

    def _print_summary(self, result: Dict):
        """Print colorful summary"""
        summary = result["summary"]
        decision = result["final_decision"]

        logger.info(f"\n{'═'*70}")
        logger.info(f"📊 FINAL ANALYSIS SUMMARY")
        logger.info(f"{'═'*70}")
        logger.info(f"Match: {result['match']['home']} vs {result['match']['away']}")
        logger.info(f"League: {result['match'].get('league', 'N/A')}")
        logger.info(f"")
        logger.info(f"Probability: {summary['probability']:.1%}")
        logger.info(f"Confidence: {summary['confidence']:.0f}/100")
        logger.info(f"Value Score: {summary['value_score']:.0f}/100")
        logger.info(f"Expected Value: {summary['expected_value']:+.1%}")

        # Show ensemble breakdown if available
        if result.get("ensemble"):
            ensemble = result["ensemble"]
            logger.info(f"")
            logger.info(f"🤖 ENSEMBLE BREAKDOWN:")
            for model in ['dixon_coles', 'xgboost', 'lstm']:
                if model in ensemble['model_predictions']:
                    pred = ensemble['model_predictions'][model]
                    weight = ensemble['model_weights'][model]
                    logger.info(f"   {model:12s}: {pred:.1%} (weight: {weight:.1%})")

        logger.info(f"")
        logger.info(f"{'─'*70}")
        logger.info(f"DECISION: {decision['action']}")
        logger.info(f"STAKE: €{decision['stake']:.2f}")
        logger.info(f"TIMING: {decision['timing']}")
        logger.info(f"PRIORITY: {decision['priority']}")
        logger.info(f"{'─'*70}")

        if decision["action"] == "BET":
            logger.info(f"💰 Potential profit: €{summary['potential_profit']:.2f}")
        logger.info(f"{'═'*70}\n")

    def load_models(self, models_dir: Optional[str] = None):
        """Load all trained models"""
        models_dir = Path(models_dir or self.config.models_dir)

        logger.info(f"📦 Loading models from {models_dir}...")

        # Load calibrator
        calibrator_path = models_dir / "calibrator.pth"
        if calibrator_path.exists():
            self.calibrator.load(calibrator_path)
            logger.info("   ✓ Calibrator loaded")
        else:
            logger.warning("   ⚠️  Calibrator model not found")

        # Load confidence scorer
        confidence_path = models_dir / "confidence_scorer.pkl"
        if confidence_path.exists():
            self.confidence_scorer.load(confidence_path)
            logger.info("   ✓ Confidence Scorer loaded")
        else:
            logger.warning("   ⚠️  Confidence Scorer model not found")

        # Load value detector
        value_path = models_dir / "value_detector.pkl"
        if value_path.exists():
            self.value_detector.load(value_path)
            logger.info("   ✓ Value Detector loaded")
        else:
            logger.warning("   ⚠️  Value Detector model not found")

        logger.info("✅ Model loading completed")

    def save_analysis(self, filepath: str):
        """Save last analysis to file"""
        if self.last_analysis is None:
            logger.warning("⚠️  No analysis to save")
            return

        with open(filepath, 'w') as f:
            json.dump(self.last_analysis, f, indent=2, default=str)

        logger.info(f"💾 Analysis saved to {filepath}")

    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "total_analyses": len(self.analysis_history),
            "api_stats": self.api_engine.get_statistics(),
            "recent_analyses": self.analysis_history[-10:],  # Last 10
            "models_status": self._get_models_status()
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_analyze(
    home_team: str,
    away_team: str,
    league: str,
    prob_dixon_coles: float,
    odds: float,
    bankroll: float = 1000.0,
    **kwargs
) -> Dict:
    """
    Quick analysis con interfaccia semplificata.

    Args:
        home_team: Squadra casa
        away_team: Squadra trasferta
        league: Lega
        prob_dixon_coles: Probabilità Dixon-Coles
        odds: Quote attuali
        bankroll: Bankroll (default 1000)
        **kwargs: Altri parametri opzionali

    Returns:
        Risultato completo analisi
    """
    pipeline = AIPipeline()

    match = {
        "home": home_team,
        "away": away_team,
        "league": league,
        "date": kwargs.get("date", datetime.now().strftime("%Y-%m-%d"))
    }

    odds_data = {
        "odds_current": odds,
        "odds_history": kwargs.get("odds_history") or [],  # Ensure never None
        "market": kwargs.get("market", "1x2"),
        "time_to_kickoff_hours": kwargs.get("time_to_kickoff", 24.0)
    }

    return pipeline.analyze(match, prob_dixon_coles, odds_data, bankroll)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test pipeline
    print("Testing AI Pipeline...")

    result = quick_analyze(
        home_team="Inter",
        away_team="Napoli",
        league="Serie A",
        prob_dixon_coles=0.65,
        odds=1.85,
        bankroll=1000.0,
        odds_history=[
            {"odds": 1.90, "time": "10:00"},
            {"odds": 1.88, "time": "11:00"},
            {"odds": 1.85, "time": "12:00"}
        ]
    )

    print("\n" + "="*70)
    print("✅ Pipeline test completed")
    print(f"Decision: {result['final_decision']['action']}")
    print(f"Stake: €{result['final_decision']['stake']:.2f}")
    print("="*70)
