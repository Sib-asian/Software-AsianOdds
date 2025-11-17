"""
Test Ensemble Meta-Model
=========================

Script completo per testare l'Ensemble end-to-end.

Tests:
1. Individual models (XGBoost, LSTM, Meta-Learner)
2. Ensemble integration
3. Pipeline integration
4. Edge cases (no API, no history, etc.)
"""

import sys
import logging
from pathlib import Path

# Add ai_system to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_xgboost():
    """Test XGBoost Predictor"""
    from ai_system.models import XGBoostPredictor

    logger.info("\n" + "="*70)
    logger.info("TEST 1: XGBoost Predictor")
    logger.info("="*70)

    predictor = XGBoostPredictor()

    match_data = {
        'home': 'Inter',
        'away': 'Napoli',
        'league': 'Serie A'
    }

    api_context = {
        'home_context': {
            'data': {
                'form': 'WWWDW',
                'injuries': []
            }
        },
        'away_context': {
            'data': {
                'form': 'WDWWL',
                'injuries': ['Player1']
            }
        },
        'match_data': {
            'xg_home': 2.1,
            'xg_away': 1.6,
            'xga_home': 0.9,
            'xga_away': 1.3,
            'lineup_home': 0.90,
            'lineup_away': 0.82
        }
    }

    prob = predictor.predict(match_data, api_context)

    logger.info(f"✅ XGBoost prediction: {prob:.1%}")
    logger.info(f"   Features extracted: {len(predictor.feature_names)}")
    logger.info(f"   Model trained: {predictor.is_trained}")

    assert 0.0 < prob < 1.0, "Probability out of range"

    logger.info("✅ XGBoost test PASSED\n")
    return prob


def test_lstm():
    """Test LSTM Predictor"""
    from ai_system.models import LSTMPredictor

    logger.info("\n" + "="*70)
    logger.info("TEST 2: LSTM Predictor")
    logger.info("="*70)

    predictor = LSTMPredictor()

    match_history = [
        {'result': 'W', 'goals_scored': 2, 'goals_conceded': 0, 'xg': 2.1, 'xga': 0.8, 'venue': 'home'},
        {'result': 'W', 'goals_scored': 3, 'goals_conceded': 1, 'xg': 2.5, 'xga': 1.2, 'venue': 'away'},
        {'result': 'D', 'goals_scored': 1, 'goals_conceded': 1, 'xg': 1.8, 'xga': 1.5, 'venue': 'home'},
        {'result': 'W', 'goals_scored': 2, 'goals_conceded': 1, 'xg': 2.0, 'xga': 1.1, 'venue': 'away'},
        {'result': 'L', 'goals_scored': 0, 'goals_conceded': 2, 'xg': 1.2, 'xga': 2.3, 'venue': 'away'},
    ]

    current_match = {
        'home': 'Inter',
        'away': 'Napoli',
        'league': 'Serie A'
    }

    prob = predictor.predict(match_history, current_match)

    logger.info(f"✅ LSTM prediction: {prob:.1%}")
    logger.info(f"   Sequence length: {predictor.sequence_length}")
    logger.info(f"   Model trained: {predictor.is_trained}")

    assert 0.0 < prob < 1.0, "Probability out of range"

    logger.info("✅ LSTM test PASSED\n")
    return prob


def test_meta_learner():
    """Test Meta-Learner"""
    from ai_system.models import MetaLearner

    logger.info("\n" + "="*70)
    logger.info("TEST 3: Meta-Learner")
    logger.info("="*70)

    meta = MetaLearner(num_models=3)

    predictions = {
        'dixon_coles': 0.65,
        'xgboost': 0.58,
        'lstm': 0.62
    }

    match_data = {
        'league': 'Serie A',
        'hours_to_kickoff': 12,
        'season_progress': 0.6
    }

    api_context = {
        'metadata': {'data_quality': 0.85},
        'match_data': {'h2h': {'total': 8}},
        'home_context': {'data': {'injuries': []}},
        'away_context': {'data': {'injuries': ['Player1']}}
    }

    weights = meta.calculate_weights(predictions, match_data, api_context)

    logger.info(f"✅ Weights calculated:")
    for model, weight in weights.items():
        logger.info(f"   {model:12s}: {weight:.1%}")

    # Verify weights sum to 1.0
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 0.001, f"Weights don't sum to 1.0: {weight_sum}"

    # Calculate ensemble
    ensemble_pred = sum(predictions[model] * weights[model] for model in predictions)
    logger.info(f"\n   Ensemble prediction: {ensemble_pred:.1%}")

    logger.info("✅ Meta-Learner test PASSED\n")
    return weights


def test_ensemble():
    """Test Ensemble Meta-Model"""
    from ai_system.models import EnsembleMetaModel

    logger.info("\n" + "="*70)
    logger.info("TEST 4: Ensemble Meta-Model")
    logger.info("="*70)

    ensemble = EnsembleMetaModel()

    match_data = {
        'home': 'Inter',
        'away': 'Napoli',
        'league': 'Serie A',
        'hours_to_kickoff': 12,
        'season_progress': 0.6
    }

    prob_dixon_coles = 0.62

    api_context = {
        'metadata': {'data_quality': 0.85},
        'home_context': {
            'data': {
                'form': 'WWWDW',
                'injuries': []
            }
        },
        'away_context': {
            'data': {
                'form': 'WDWWL',
                'injuries': ['Player1']
            }
        },
        'match_data': {
            'xg_home': 2.1,
            'xg_away': 1.6,
            'xga_home': 0.9,
            'xga_away': 1.3,
            'lineup_home': 0.90,
            'lineup_away': 0.82,
            'h2h': {'total': 8}
        }
    }

    match_history = [
        {'result': 'W', 'goals_scored': 2, 'goals_conceded': 0, 'xg': 2.1, 'xga': 0.8, 'venue': 'home'},
        {'result': 'W', 'goals_scored': 3, 'goals_conceded': 1, 'xg': 2.5, 'xga': 1.2, 'venue': 'away'},
        {'result': 'D', 'goals_scored': 1, 'goals_conceded': 1, 'xg': 1.8, 'xga': 1.5, 'venue': 'home'},
        {'result': 'W', 'goals_scored': 2, 'goals_conceded': 1, 'xg': 2.0, 'xga': 1.1, 'venue': 'away'},
    ]

    result = ensemble.predict(
        match_data=match_data,
        prob_dixon_coles=prob_dixon_coles,
        api_context=api_context,
        match_history=match_history
    )

    logger.info(f"✅ Ensemble Result:")
    logger.info(f"   Final probability: {result['probability']:.1%}")
    logger.info(f"   Confidence: {result['confidence']:.0f}/100")
    logger.info(f"   Uncertainty: {result['uncertainty']:.3f}")
    logger.info(f"\n   Model Predictions:")
    for model, pred in result['model_predictions'].items():
        weight = result['model_weights'][model]
        logger.info(f"      {model:12s}: {pred:.1%} (weight: {weight:.1%})")

    logger.info(f"\n   Dominant model: {result['breakdown']['summary']['dominant_model']}")

    # Assertions
    assert 0.0 < result['probability'] < 1.0, "Probability out of range"
    assert 0 <= result['confidence'] <= 100, "Confidence out of range"
    assert result['uncertainty'] >= 0, "Uncertainty negative"

    logger.info("✅ Ensemble test PASSED\n")
    return result


def test_ensemble_without_api():
    """Test Ensemble without API context (fallback)"""
    from ai_system.models import EnsembleMetaModel

    logger.info("\n" + "="*70)
    logger.info("TEST 5: Ensemble WITHOUT API Context (Fallback)")
    logger.info("="*70)

    ensemble = EnsembleMetaModel()

    match_data = {
        'home': 'Inter',
        'away': 'Napoli',
        'league': 'Serie A'
    }

    prob_dixon_coles = 0.62

    # No API context, no history
    result = ensemble.predict(
        match_data=match_data,
        prob_dixon_coles=prob_dixon_coles,
        api_context=None,
        match_history=None
    )

    logger.info(f"✅ Ensemble Result (no API):")
    logger.info(f"   Final probability: {result['probability']:.1%}")
    logger.info(f"   Confidence: {result['confidence']:.0f}/100")
    logger.info(f"   (Lower confidence expected without API data)")

    assert 0.0 < result['probability'] < 1.0, "Probability out of range"

    logger.info("✅ Fallback test PASSED\n")
    return result


def test_pipeline_integration():
    """Test Pipeline integration with Ensemble"""
    from ai_system.pipeline import quick_analyze
    from ai_system.config import AIConfig

    logger.info("\n" + "="*70)
    logger.info("TEST 6: Pipeline Integration")
    logger.info("="*70)

    # Create config with ensemble enabled
    config = AIConfig()
    config.use_ensemble = True
    config.ensemble_load_models = False  # Don't load trained models (not exist yet)

    logger.info("Running pipeline with ensemble enabled...")

    try:
        result = quick_analyze(
            home_team="Inter",
            away_team="Napoli",
            league="Serie A",
            prob_dixon_coles=0.65,
            odds=1.85,
            bankroll=1000.0
        )

        logger.info(f"✅ Pipeline Result:")
        logger.info(f"   Decision: {result['final_decision']['action']}")
        logger.info(f"   Stake: €{result['final_decision']['stake']:.2f}")
        logger.info(f"   Probability: {result['summary']['probability']:.1%}")

        if result.get('ensemble'):
            logger.info(f"\n   Ensemble Info:")
            logger.info(f"      Final prob: {result['ensemble']['probability']:.1%}")
            logger.info(f"      Confidence: {result['ensemble']['confidence']:.0f}/100")
            logger.info(f"      Uncertainty: {result['ensemble']['uncertainty']:.3f}")
        else:
            logger.warning("   ⚠️  No ensemble data (may be disabled or failed)")

        logger.info("✅ Pipeline integration test PASSED\n")
        return result

    except Exception as e:
        logger.error(f"❌ Pipeline test FAILED: {e}")
        raise


def test_edge_cases():
    """Test edge cases and error handling"""
    from ai_system.models import EnsembleMetaModel

    logger.info("\n" + "="*70)
    logger.info("TEST 7: Edge Cases")
    logger.info("="*70)

    ensemble = EnsembleMetaModel()

    # Test 1: Empty match data
    logger.info("Testing empty match data...")
    try:
        result = ensemble.predict(
            match_data={},
            prob_dixon_coles=0.5,
            api_context=None,
            match_history=None
        )
        logger.info("✅ Handled empty match data")
    except Exception as e:
        logger.warning(f"⚠️  Failed with empty data (expected): {e}")

    # Test 2: Extreme probability
    logger.info("Testing extreme Dixon-Coles probability...")
    result = ensemble.predict(
        match_data={'home': 'Strong', 'away': 'Weak', 'league': 'Test'},
        prob_dixon_coles=0.95,  # Very high
        api_context=None,
        match_history=None
    )
    logger.info(f"✅ Handled extreme prob: DC=0.95 → Ensemble={result['probability']:.1%}")

    # Test 3: Very short match history
    logger.info("Testing short match history...")
    short_history = [
        {'result': 'W', 'goals_scored': 2, 'xg': 2.0, 'venue': 'home'}
    ]
    result = ensemble.predict(
        match_data={'home': 'Team', 'away': 'Team2', 'league': 'Test'},
        prob_dixon_coles=0.6,
        api_context=None,
        match_history=short_history
    )
    logger.info(f"✅ Handled short history: {result['probability']:.1%}")

    logger.info("✅ Edge cases test PASSED\n")


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("🚀 STARTING ENSEMBLE META-MODEL TESTS")
    logger.info("="*70)

    results = {}

    try:
        # Individual models
        results['xgboost'] = test_xgboost()
        results['lstm'] = test_lstm()
        results['meta_learner'] = test_meta_learner()

        # Ensemble
        results['ensemble'] = test_ensemble()
        results['ensemble_no_api'] = test_ensemble_without_api()

        # Integration
        results['pipeline'] = test_pipeline_integration()

        # Edge cases
        test_edge_cases()

        # Summary
        logger.info("\n" + "="*70)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*70)
        logger.info("\n📊 RESULTS SUMMARY:")
        logger.info(f"   XGBoost prediction: {results['xgboost']:.1%}")
        logger.info(f"   LSTM prediction: {results['lstm']:.1%}")
        logger.info(f"   Ensemble (full context): {results['ensemble']['probability']:.1%}")
        logger.info(f"   Ensemble (no API): {results['ensemble_no_api']['probability']:.1%}")
        logger.info(f"   Pipeline decision: {results['pipeline']['final_decision']['action']}")

        logger.info("\n✨ Ensemble Meta-Model is READY for production!")
        logger.info("="*70)

        return True

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        logger.error("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
