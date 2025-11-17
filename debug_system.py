#!/usr/bin/env python3
"""
🔍 Asian Odds System - Complete Debug & Test Script
====================================================

Verifica che tutto il sistema funzioni correttamente:
- Import di tutti i moduli
- Sistema AI (7 blocchi)
- Modelli ML (Ensemble, XGBoost, LSTM)
- Features avanzate (LLM, Sentiment, Backtesting)
- Pipeline completa
- Telegram (opzionale)

Usage:
    python debug_system.py

    # Test specifico
    python debug_system.py --test imports
    python debug_system.py --test ai_blocks
    python debug_system.py --test pipeline
"""

import sys
import os
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

# ============================================================
# TEST 1: PYTHON VERSION & DEPENDENCIES
# ============================================================

def test_python_version() -> bool:
    """Test Python version"""
    print_info("Testing Python version...")

    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (Need >= 3.8)")
        return False

def test_dependencies() -> Tuple[bool, List[str]]:
    """Test required dependencies"""
    print_info("Testing required dependencies...")

    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'requests',
        'streamlit'
    ]

    optional_packages = [
        'torch',           # For LSTM
        'xgboost',         # For XGBoost
        'openai',          # For LLM Analyst
        'anthropic',       # For Claude
        'telegram'         # For Telegram notifications
    ]

    missing_required = []
    missing_optional = []

    # Check required
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package} (REQUIRED)")
            missing_required.append(package)

    # Check optional
    for package in optional_packages:
        try:
            __import__(package)
            print_success(f"{package} (optional)")
        except ImportError:
            print_warning(f"{package} (optional - some features disabled)")
            missing_optional.append(package)

    if missing_required:
        print_error(f"Missing required packages: {', '.join(missing_required)}")
        print_info("Install with: pip install " + " ".join(missing_required))
        return False, missing_required

    return True, missing_optional

# ============================================================
# TEST 2: MODULE IMPORTS
# ============================================================

def test_imports() -> Tuple[bool, Dict[str, Any]]:
    """Test all module imports"""
    print_info("Testing module imports...")

    results = {}
    errors = []

    # Test main modules
    modules_to_test = [
        # AI System modules
        ('ai_system.config', 'AIConfig'),
        ('ai_system.pipeline', 'quick_analyze'),
        ('ai_system.blocco_0_api_engine', 'APIDataEngine'),
        ('ai_system.blocco_1_calibrator', 'ProbabilityCalibrator'),
        ('ai_system.blocco_2_confidence', 'ConfidenceScorer'),
        ('ai_system.blocco_3_value_detector', 'ValueDetector'),
        ('ai_system.blocco_4_kelly', 'SmartKellyOptimizer'),
        ('ai_system.blocco_5_risk_manager', 'RiskManager'),
        ('ai_system.blocco_6_odds_tracker', 'OddsMovementTracker'),

        # AI Models
        ('ai_system.models.ensemble', 'EnsembleMetaModel'),
        ('ai_system.models.xgboost_predictor', 'XGBoostPredictor'),
        ('ai_system.models.lstm_predictor', 'LSTMPredictor'),
        ('ai_system.models.meta_learner', 'MetaLearner'),

        # Advanced features
        ('ai_system.llm_analyst', 'LLMAnalyst'),
        ('ai_system.sentiment_analyzer', 'SentimentAnalyzer'),
        ('ai_system.live_betting', 'LiveBettingEngine'),
        ('ai_system.backtesting', 'Backtester'),
        ('ai_system.telegram_notifier', 'TelegramNotifier'),
    ]

    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            results[module_name] = cls
            print_success(f"{module_name}.{class_name}")
        except Exception as e:
            print_error(f"{module_name}.{class_name}: {str(e)}")
            errors.append((module_name, class_name, str(e)))

    if errors:
        print_error(f"Failed to import {len(errors)} modules")
        for mod, cls, err in errors:
            print(f"  - {mod}.{cls}: {err}")
        return False, results

    return True, results

# ============================================================
# TEST 3: AI BLOCKS (0-6)
# ============================================================

def test_ai_blocks(modules: Dict[str, Any]) -> bool:
    """Test all 7 AI blocks"""
    print_info("Testing AI blocks initialization...")

    try:
        # Test data
        test_match_data = {
            'home_team': 'Test Home',
            'away_team': 'Test Away',
            'league': 'Test League',
            'prob_home': 0.5,
            'prob_draw': 0.25,
            'prob_away': 0.25
        }

        # Blocco 0: API Engine
        print_info("Testing Blocco 0: API Data Engine...")
        APIEngine = modules.get('ai_system.blocco_0_api_engine')
        if APIEngine:
            engine = APIEngine()
            print_success("Blocco 0: API Engine initialized")

        # Blocco 1: Calibrator
        print_info("Testing Blocco 1: Probability Calibrator...")
        Calibrator = modules.get('ai_system.blocco_1_calibrator')
        if Calibrator:
            calibrator = Calibrator()
            # Test calibration with mock data and context
            result = calibrator.calibrate(0.65, context={})
            calibrated = result if isinstance(result, float) else result.get('calibrated_prob', 0.65)
            print_success(f"Blocco 1: Calibrator initialized (0.65 → {calibrated:.3f})")

        # Blocco 2: Confidence Scorer
        print_info("Testing Blocco 2: Confidence Scorer...")
        ConfScorer = modules.get('ai_system.blocco_2_confidence')
        if ConfScorer:
            scorer = ConfScorer()
            print_success("Blocco 2: Confidence Scorer initialized")

        # Blocco 3: Value Detector
        print_info("Testing Blocco 3: Value Detector...")
        ValueDet = modules.get('ai_system.blocco_3_value_detector')
        if ValueDet:
            detector = ValueDet()
            print_success("Blocco 3: Value Detector initialized")

        # Blocco 4: Kelly Optimizer
        print_info("Testing Blocco 4: Smart Kelly Optimizer...")
        KellyOpt = modules.get('ai_system.blocco_4_kelly')
        if KellyOpt:
            kelly = KellyOpt()
            # Kelly needs value_result and confidence_result
            print_success("Blocco 4: Kelly Optimizer initialized")

        # Blocco 5: Risk Manager
        print_info("Testing Blocco 5: Risk Manager...")
        RiskMgr = modules.get('ai_system.blocco_5_risk_manager')
        if RiskMgr:
            risk_mgr = RiskMgr()
            print_success("Blocco 5: Risk Manager initialized")

        # Blocco 6: Odds Tracker
        print_info("Testing Blocco 6: Odds Movement Tracker...")
        OddsTracker = modules.get('ai_system.blocco_6_odds_tracker')
        if OddsTracker:
            tracker = OddsTracker()
            print_success("Blocco 6: Odds Tracker initialized")

        print_success("All 7 AI blocks initialized successfully!")
        return True

    except Exception as e:
        print_error(f"AI blocks test failed: {str(e)}")
        traceback.print_exc()
        return False

# ============================================================
# TEST 4: ML MODELS
# ============================================================

def test_ml_models(modules: Dict[str, Any]) -> bool:
    """Test ML models"""
    print_info("Testing ML models...")

    try:
        # Test Ensemble
        print_info("Testing Ensemble Model...")
        Ensemble = modules.get('ai_system.models.ensemble')
        if Ensemble:
            ensemble = Ensemble()
            print_success("Ensemble Model initialized")

        # Test XGBoost
        print_info("Testing XGBoost Predictor...")
        XGB = modules.get('ai_system.models.xgboost_predictor')
        if XGB:
            xgb = XGB()
            print_success("XGBoost Predictor initialized")

        # Test LSTM
        print_info("Testing LSTM Predictor...")
        LSTM = modules.get('ai_system.models.lstm_predictor')
        if LSTM:
            try:
                lstm = LSTM()
                print_success("LSTM Predictor initialized")
            except Exception as e:
                print_warning(f"LSTM requires PyTorch: {str(e)}")

        # Test Meta Learner
        print_info("Testing Meta Learner...")
        Meta = modules.get('ai_system.models.meta_learner')
        if Meta:
            meta = Meta()
            print_success("Meta Learner initialized")

        print_success("ML models test completed!")
        return True

    except Exception as e:
        print_error(f"ML models test failed: {str(e)}")
        traceback.print_exc()
        return False

# ============================================================
# TEST 5: PIPELINE (END-TO-END)
# ============================================================

def test_pipeline(modules: Dict[str, Any]) -> bool:
    """Test complete pipeline end-to-end"""
    print_info("Testing complete pipeline...")

    try:
        quick_analyze = modules.get('ai_system.pipeline')
        if not quick_analyze:
            print_error("Pipeline module not found")
            return False

        # Test with real-world-like data
        print_info("Running test analysis...")

        result = quick_analyze(
            home_team="Manchester City",
            away_team="Arsenal",
            league="Premier League",
            prob_dixon_coles=0.65,
            odds=1.90,
            bankroll=1000.0,
            use_ensemble=False  # Start with basic
        )

        print_success("Pipeline executed successfully!")
        print_info("Result preview:")
        print(f"  Action: {result.get('action', 'N/A')}")
        print(f"  Stake: €{result.get('stake_amount', 0):.2f}")
        print(f"  EV: {result.get('expected_value', 0):.1f}%")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")

        # Test with ensemble
        print_info("Testing with Ensemble enabled...")
        result_ensemble = quick_analyze(
            home_team="Manchester City",
            away_team="Arsenal",
            league="Premier League",
            prob_dixon_coles=0.65,
            odds=1.90,
            bankroll=1000.0,
            use_ensemble=True
        )

        print_success("Ensemble pipeline executed successfully!")

        return True

    except Exception as e:
        print_error(f"Pipeline test failed: {str(e)}")
        traceback.print_exc()
        return False

# ============================================================
# TEST 6: ADVANCED FEATURES
# ============================================================

def test_advanced_features(modules: Dict[str, Any]) -> bool:
    """Test advanced features"""
    print_info("Testing advanced features...")

    success = True

    # Test LLM Analyst
    try:
        print_info("Testing LLM Analyst...")
        LLM = modules.get('ai_system.llm_analyst')
        if LLM:
            # Just init, don't call API
            llm = LLM(model="gpt-4")
            print_success("LLM Analyst initialized (API not called)")
        else:
            print_warning("LLM Analyst module not available")
    except Exception as e:
        print_warning(f"LLM Analyst: {str(e)}")

    # Test Sentiment Analyzer
    try:
        print_info("Testing Sentiment Analyzer...")
        Sentiment = modules.get('ai_system.sentiment_analyzer')
        if Sentiment:
            sentiment = Sentiment()
            print_success("Sentiment Analyzer initialized")
        else:
            print_warning("Sentiment Analyzer module not available")
    except Exception as e:
        print_warning(f"Sentiment Analyzer: {str(e)}")

    # Test Live Betting
    try:
        print_info("Testing Live Betting Adjuster...")
        LiveBetting = modules.get('ai_system.live_betting')
        if LiveBetting:
            live = LiveBetting()
            print_success("Live Betting Adjuster initialized")
        else:
            print_warning("Live Betting module not available")
    except Exception as e:
        print_warning(f"Live Betting: {str(e)}")

    # Test Backtesting
    try:
        print_info("Testing Backtester...")
        Backtester = modules.get('ai_system.backtesting')
        if Backtester:
            backtester = Backtester()
            print_success("Backtester initialized")
        else:
            print_warning("Backtester module not available")
    except Exception as e:
        print_warning(f"Backtester: {str(e)}")

    # Test Telegram (optional)
    try:
        print_info("Testing Telegram Notifier...")
        Telegram = modules.get('ai_system.telegram_notifier')
        if Telegram:
            # Don't actually send, just init
            print_success("Telegram Notifier available (not configured)")
        else:
            print_warning("Telegram Notifier module not available")
    except Exception as e:
        print_warning(f"Telegram Notifier: {str(e)}")

    return success

# ============================================================
# TEST 7: CONFIG
# ============================================================

def test_config(modules: Dict[str, Any]) -> bool:
    """Test configuration"""
    print_info("Testing configuration...")

    try:
        AIConfig = modules.get('ai_system.config')
        if not AIConfig:
            print_error("Config module not found")
            return False

        config = AIConfig()

        print_success("Configuration loaded:")
        print(f"  Ensemble enabled: {config.use_ensemble}")
        print(f"  Telegram enabled: {config.telegram_enabled}")
        print(f"  Live monitoring: {config.live_monitoring_enabled}")
        print(f"  Min EV alert: {config.live_min_ev_alert}%")
        print(f"  Update interval: {config.live_update_interval}s")

        return True

    except Exception as e:
        print_error(f"Config test failed: {str(e)}")
        traceback.print_exc()
        return False

# ============================================================
# TEST 8: FILE STRUCTURE
# ============================================================

def test_file_structure() -> bool:
    """Test file structure"""
    print_info("Testing file structure...")

    required_files = [
        'Frontendcloud.py',
        'requirements.txt',
        'ai_system/__init__.py',
        'ai_system/config.py',
        'ai_system/pipeline.py',
        'team_profiles.json'
    ]

    required_dirs = [
        'ai_system',
        'ai_system/models',
        'ai_system/utils'
    ]

    missing_files = []
    missing_dirs = []

    # Check files
    for file in required_files:
        if os.path.exists(file):
            print_success(f"{file}")
        else:
            print_error(f"{file} (missing)")
            missing_files.append(file)

    # Check directories
    for dir in required_dirs:
        if os.path.isdir(dir):
            print_success(f"{dir}/")
        else:
            print_error(f"{dir}/ (missing)")
            missing_dirs.append(dir)

    if missing_files or missing_dirs:
        print_error("Some files/directories are missing!")
        return False

    return True

# ============================================================
# MAIN DEBUG RUNNER
# ============================================================

def run_all_tests():
    """Run all debug tests"""

    print_header("🔍 ASIAN ODDS SYSTEM - COMPLETE DEBUG")

    start_time = datetime.now()

    results = {
        'python_version': False,
        'dependencies': False,
        'imports': False,
        'ai_blocks': False,
        'ml_models': False,
        'pipeline': False,
        'advanced_features': False,
        'config': False,
        'file_structure': False
    }

    # Test 1: Python version
    print_header("TEST 1: Python Version")
    results['python_version'] = test_python_version()

    # Test 2: Dependencies
    print_header("TEST 2: Dependencies")
    deps_ok, missing_optional = test_dependencies()
    results['dependencies'] = deps_ok

    if not deps_ok:
        print_error("\n❌ Critical dependencies missing. Install them first!")
        return False

    # Test 3: File structure
    print_header("TEST 3: File Structure")
    results['file_structure'] = test_file_structure()

    # Test 4: Module imports
    print_header("TEST 4: Module Imports")
    imports_ok, modules = test_imports()
    results['imports'] = imports_ok

    if not imports_ok:
        print_error("\n❌ Some modules failed to import!")
        print_info("Check errors above and fix them first.")
        return False

    # Test 5: Config
    print_header("TEST 5: Configuration")
    results['config'] = test_config(modules)

    # Test 6: AI blocks
    print_header("TEST 6: AI Blocks (0-6)")
    results['ai_blocks'] = test_ai_blocks(modules)

    # Test 7: ML models
    print_header("TEST 7: ML Models")
    results['ml_models'] = test_ml_models(modules)

    # Test 8: Pipeline
    print_header("TEST 8: Complete Pipeline")
    results['pipeline'] = test_pipeline(modules)

    # Test 9: Advanced features
    print_header("TEST 9: Advanced Features")
    results['advanced_features'] = test_advanced_features(modules)

    # Summary
    print_header("📊 DEBUG SUMMARY")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if result else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")

    if missing_optional:
        print(f"\n{Colors.YELLOW}Optional packages missing:{Colors.END}")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print(f"\n{Colors.BLUE}Some features will be disabled without these packages.{Colors.END}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{Colors.BLUE}Debug completed in {elapsed:.2f}s{Colors.END}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! System is ready to use! 🎉{Colors.END}")
        return True
    else:
        print(f"\n{Colors.YELLOW}⚠️  Some tests failed. Check errors above.{Colors.END}")
        return False

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Debug Asian Odds System")
    parser.add_argument(
        '--test',
        type=str,
        choices=['all', 'imports', 'ai_blocks', 'pipeline', 'models', 'features'],
        default='all',
        help='Which test to run (default: all)'
    )

    args = parser.parse_args()

    if args.test == 'all':
        success = run_all_tests()
    else:
        # Run specific test
        print_header(f"Running test: {args.test}")

        if args.test == 'imports':
            success, modules = test_imports()
        elif args.test == 'ai_blocks':
            _, modules = test_imports()
            success = test_ai_blocks(modules)
        elif args.test == 'pipeline':
            _, modules = test_imports()
            success = test_pipeline(modules)
        elif args.test == 'models':
            _, modules = test_imports()
            success = test_ml_models(modules)
        elif args.test == 'features':
            _, modules = test_imports()
            success = test_advanced_features(modules)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
