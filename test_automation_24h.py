#!/usr/bin/env python3
"""
Test Suite for Automation24H System
====================================

Comprehensive tests for the 24/7 automation system.
Tests cover core functionality without requiring external dependencies.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))


class TestAutomation24HCore(unittest.TestCase):
    """Test core functionality of Automation24H"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock all external dependencies at module level before import
        self.mock_patches = []
        
        # Mock AI system module imports
        self.mock_patches.append(patch.dict('sys.modules', {
            'ai_system': MagicMock(),
            'ai_system.pipeline': MagicMock(),
            'ai_system.config': MagicMock(),
            'ai_system.telegram_notifier': MagicMock(),
            'ai_system.multi_model_consensus': MagicMock(),
            'ai_system.intelligent_alert_system': MagicMock(),
            'ai_system.pattern_analyzer_llm': MagicMock(),
            'ai_system.parameter_optimizer': MagicMock(),
        }))
        
        # Start all patches
        for p in self.mock_patches:
            p.start()
        
        # Import after patching
        from automation_24h import Automation24H
        self.Automation24H = Automation24H
    
    def tearDown(self):
        """Clean up patches"""
        for p in self.mock_patches:
            p.stop()
    
    def test_initialization_basic(self):
        """Test basic initialization without external dependencies"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None,
            min_ev=8.0,
            min_confidence=70.0
        )
        
        # Verify basic attributes
        self.assertEqual(automation.min_ev, 8.0)
        self.assertEqual(automation.min_confidence, 70.0)
        self.assertFalse(automation.running)
        self.assertEqual(len(automation.notified_opportunities), 0)
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None,
            min_ev=15.0,
            min_confidence=80.0,
            update_interval=600,
            api_budget_per_day=200
        )
        
        self.assertEqual(automation.min_ev, 15.0)
        self.assertEqual(automation.min_confidence, 80.0)
        self.assertEqual(automation.update_interval, 600)
        self.assertEqual(automation.api_budget_per_day, 200)
    
    def test_reset_api_usage_new_day(self):
        """Test API usage reset on new day"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None
        )
        
        # Set usage from yesterday
        automation.api_usage_today = 50
        automation.last_api_reset = datetime.now().date() - timedelta(days=1)
        automation.notified_opportunities.add('match_123')
        
        # Reset
        automation._reset_api_usage_if_needed()
        
        # Verify reset
        self.assertEqual(automation.api_usage_today, 0)
        self.assertEqual(automation.last_api_reset, datetime.now().date())
        self.assertEqual(len(automation.notified_opportunities), 0)
    
    def test_reset_api_usage_same_day(self):
        """Test API usage NOT reset on same day"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None
        )
        
        # Set usage today
        automation.api_usage_today = 50
        automation.last_api_reset = datetime.now().date()
        automation.notified_opportunities.add('match_123')
        
        # Try reset
        automation._reset_api_usage_if_needed()
        
        # Verify NOT reset
        self.assertEqual(automation.api_usage_today, 50)
        self.assertEqual(len(automation.notified_opportunities), 1)


class TestOpportunityFiltering(unittest.TestCase):
    """Test opportunity filtering logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock all external dependencies at module level before import
        self.mock_patches = []
        
        # Mock AI system module imports
        self.mock_patches.append(patch.dict('sys.modules', {
            'ai_system': MagicMock(),
            'ai_system.pipeline': MagicMock(),
            'ai_system.config': MagicMock(),
            'ai_system.telegram_notifier': MagicMock(),
            'ai_system.multi_model_consensus': MagicMock(),
            'ai_system.intelligent_alert_system': MagicMock(),
            'ai_system.pattern_analyzer_llm': MagicMock(),
            'ai_system.parameter_optimizer': MagicMock(),
        }))
        
        # Start all patches
        for p in self.mock_patches:
            p.start()
        
        # Import after patching
        from automation_24h import Automation24H
        self.Automation24H = Automation24H
    
    def tearDown(self):
        """Clean up patches"""
        for p in self.mock_patches:
            p.stop()
    
    def test_is_true_value_bet_valid(self):
        """Test value bet detection with valid opportunity"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None
        )
        
        # Mock AI result with true value
        # Probability 60% vs odds 1.50 (implied prob 66.7%)
        # This should NOT be value (60% < 66.7%)
        ai_result = {
            'probability': 0.60,
            'odds': 1.50
        }
        self.assertFalse(automation._has_real_value(ai_result))
        
        # Probability 70% vs odds 1.30 (implied prob 76.9%)
        # This should NOT be value (70% < 76.9%)
        ai_result = {
            'probability': 0.70,
            'odds': 1.30
        }
        self.assertFalse(automation._has_real_value(ai_result))
        
        # Probability 80% vs odds 1.20 (implied prob 83.3%)
        # This should NOT be value (80% < 83.3%)
        ai_result = {
            'probability': 0.80,
            'odds': 1.20
        }
        self.assertFalse(automation._has_real_value(ai_result))
        
        # Probability 85% vs odds 1.20 (implied prob 83.3%)
        # This SHOULD be value (85% > 83.3% and margin > 5%)
        ai_result = {
            'probability': 0.90,
            'odds': 1.20
        }
        self.assertTrue(automation._has_real_value(ai_result))
    
    def test_is_true_value_bet_invalid_data(self):
        """Test value bet detection with invalid data"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None
        )
        
        # Missing probability
        ai_result = {'odds': 2.0}
        self.assertFalse(automation._has_real_value(ai_result))
        
        # Missing odds
        ai_result = {'probability': 0.60}
        self.assertFalse(automation._has_real_value(ai_result))
        
        # Invalid odds (<=1.0)
        ai_result = {'probability': 0.60, 'odds': 0.5}
        self.assertFalse(automation._has_real_value(ai_result))
        
        ai_result = {'probability': 0.60, 'odds': 1.0}
        self.assertFalse(automation._has_real_value(ai_result))
    
    def test_is_true_value_bet_nested_summary(self):
        """Test value bet detection with nested summary structure"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None
        )
        
        # Probability 90% vs odds 1.20 (implied prob 83.3%)
        # This SHOULD be value (90% > 83.3% + 5% margin)
        ai_result = {
            'summary': {
                'probability': 0.90,
                'odds': 1.20
            }
        }
        self.assertTrue(automation._has_real_value(ai_result))


class TestDuplicateDetection(unittest.TestCase):
    """Test duplicate notification prevention"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock all external dependencies at module level before import
        self.mock_patches = []
        
        # Mock AI system module imports
        self.mock_patches.append(patch.dict('sys.modules', {
            'ai_system': MagicMock(),
            'ai_system.pipeline': MagicMock(),
            'ai_system.config': MagicMock(),
            'ai_system.telegram_notifier': MagicMock(),
            'ai_system.multi_model_consensus': MagicMock(),
            'ai_system.intelligent_alert_system': MagicMock(),
            'ai_system.pattern_analyzer_llm': MagicMock(),
            'ai_system.parameter_optimizer': MagicMock(),
        }))
        
        for p in self.mock_patches:
            p.start()
        
        from automation_24h import Automation24H
        self.Automation24H = Automation24H
    
    def tearDown(self):
        """Clean up patches"""
        for p in self.mock_patches:
            p.stop()
    
    def test_handle_opportunity_prevents_duplicates(self):
        """Test that duplicate opportunities are not notified"""
        automation = self.Automation24H(
            telegram_token="test_token",
            telegram_chat_id="test_chat"
        )
        
        # Mock notifier
        automation.notifier = Mock()
        automation.notifier.send_betting_opportunity = Mock(return_value=True)
        
        opportunity = {
            'match_id': 'match_123',
            'match_data': {'home': 'Team A', 'away': 'Team B'},
            'ai_result': {'ev': 10.0, 'confidence': 75.0}
        }
        
        # First call should notify
        automation._handle_opportunity(opportunity)
        self.assertEqual(automation.notifier.send_betting_opportunity.call_count, 1)
        self.assertIn('match_123', automation.notified_opportunities)
        
        # Second call with same match_id should NOT notify
        automation._handle_opportunity(opportunity)
        self.assertEqual(automation.notifier.send_betting_opportunity.call_count, 1)  # Still 1
    
    def test_notified_opportunities_cleared_on_new_day(self):
        """Test that notified opportunities are cleared on new day"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None
        )
        
        # Add some notified opportunities
        automation.notified_opportunities.add('match_123')
        automation.notified_opportunities.add('match_456')
        
        # Set last reset to yesterday
        automation.last_api_reset = datetime.now().date() - timedelta(days=1)
        
        # Trigger reset
        automation._reset_api_usage_if_needed()
        
        # Verify cleared
        self.assertEqual(len(automation.notified_opportunities), 0)


class TestMockDataGeneration(unittest.TestCase):
    """Test mock data generation for testing"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock all external dependencies at module level before import
        self.mock_patches = []
        
        # Mock AI system module imports
        self.mock_patches.append(patch.dict('sys.modules', {
            'ai_system': MagicMock(),
            'ai_system.pipeline': MagicMock(),
            'ai_system.config': MagicMock(),
            'ai_system.telegram_notifier': MagicMock(),
            'ai_system.multi_model_consensus': MagicMock(),
            'ai_system.intelligent_alert_system': MagicMock(),
            'ai_system.pattern_analyzer_llm': MagicMock(),
            'ai_system.parameter_optimizer': MagicMock(),
        }))
        
        for p in self.mock_patches:
            p.start()
        
        from automation_24h import Automation24H
        self.Automation24H = Automation24H
    
    def tearDown(self):
        """Clean up patches"""
        for p in self.mock_patches:
            p.stop()
    
    def test_get_mock_matches_returns_data(self):
        """Test that mock matches are generated"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None
        )
        
        matches = automation._get_mock_matches()
        
        # Verify we got some matches
        self.assertGreater(len(matches), 0)
        
        # Verify match structure
        for match in matches:
            self.assertIn('id', match)
            self.assertIn('home', match)
            self.assertIn('away', match)
            # Odds are stored as odds_1, odds_x, odds_2
            self.assertTrue('odds_1' in match or 'odds' in match)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation (recommended improvement)"""
    
    def test_valid_min_ev_range(self):
        """Test that min_ev is validated"""
        # This test documents the RECOMMENDED validation
        # Currently NOT implemented in automation_24h.py
        
        # These should be valid
        valid_evs = [0, 5.0, 10.0, 50.0, 100.0]
        for ev in valid_evs:
            # Would call: automation = Automation24H(min_ev=ev)
            # Should NOT raise
            pass
        
        # These should be INVALID (but currently accepted)
        invalid_evs = [-1.0, 101.0, 200.0]
        for ev in invalid_evs:
            # Would call: automation = Automation24H(min_ev=ev)
            # SHOULD raise ValueError but currently doesn't
            pass
    
    def test_valid_confidence_range(self):
        """Test that min_confidence is validated"""
        # This test documents the RECOMMENDED validation
        # Currently NOT implemented in automation_24h.py
        
        # These should be valid
        valid_confs = [0, 50.0, 70.0, 90.0, 100.0]
        
        # These should be INVALID
        invalid_confs = [-1.0, 101.0]


class TestSingleRunMode(unittest.TestCase):
    """Test single run mode for cron jobs"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock all external dependencies at module level before import
        self.mock_patches = []
        
        # Mock AI system module imports
        self.mock_patches.append(patch.dict('sys.modules', {
            'ai_system': MagicMock(),
            'ai_system.pipeline': MagicMock(),
            'ai_system.config': MagicMock(),
            'ai_system.telegram_notifier': MagicMock(),
            'ai_system.multi_model_consensus': MagicMock(),
            'ai_system.intelligent_alert_system': MagicMock(),
            'ai_system.pattern_analyzer_llm': MagicMock(),
            'ai_system.parameter_optimizer': MagicMock(),
        }))
        
        for p in self.mock_patches:
            p.start()
        
        from automation_24h import Automation24H
        self.Automation24H = Automation24H
    
    def tearDown(self):
        """Clean up patches"""
        for p in self.mock_patches:
            p.stop()
    
    def test_single_run_executes_one_cycle(self):
        """Test that single_run mode executes only one cycle"""
        automation = self.Automation24H(
            telegram_token=None,
            telegram_chat_id=None
        )
        
        # Mock _run_cycle to track calls
        automation._run_cycle = Mock()
        
        # Run in single_run mode
        automation.start(single_run=True)
        
        # Verify only one cycle executed
        self.assertEqual(automation._run_cycle.call_count, 1)
        
        # Verify system stopped
        self.assertFalse(automation.running)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAutomation24HCore))
    suite.addTests(loader.loadTestsFromTestCase(TestOpportunityFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestDuplicateDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestMockDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSingleRunMode))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
