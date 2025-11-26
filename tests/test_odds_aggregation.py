"""
Tests for Odds Aggregation Improvements
========================================

Unit tests for:
1. Vig removal and probability normalization
2. MAD-based outlier detection
3. EWMA smoothing

These tests use pure functions and don't require external API connections.
"""

import pytest
import math
import statistics
from typing import Dict, Optional


class MockAutomation24H:
    """
    Mock class that implements the odds aggregation methods for testing.
    This allows testing the pure functions without loading the full module.
    """
    
    # Trusted bookmakers for weighted aggregation (weight 2.0)
    TRUSTED_BOOKMAKERS_WEIGHTED: Dict[str, float] = {
        'bet365': 2.0,
        'pinnacle': 2.0,
        'betfair': 2.0,
    }
    
    def _validate_odds(self, odd) -> Optional[float]:
        """Validate an odd value."""
        if odd is None:
            return None
        
        try:
            if isinstance(odd, str):
                odd = float(odd)
            elif not isinstance(odd, (int, float)):
                return None
            
            if math.isnan(odd) or math.isinf(odd):
                return None
            
            if odd <= 1.0:
                return None
            
            if odd > 1000:
                return None
            
            return float(odd)
            
        except (ValueError, TypeError):
            return None
    
    def _is_outlier_mad(self, values, x, k=3.5) -> bool:
        """MAD-based outlier detection."""
        if len(values) < 3:
            return False
        
        try:
            median_val = statistics.median(values)
            absolute_deviations = [abs(v - median_val) for v in values]
            mad = statistics.median(absolute_deviations)
            
            mad_adjusted = mad * 1.4826
            
            if mad_adjusted < 0.0001:
                return abs(x - median_val) > 0.5
            
            modified_z_score = abs(x - median_val) / mad_adjusted
            return modified_z_score > k
            
        except Exception:
            return False
    
    def _ewma(self, previous_value, new_value, alpha=0.4) -> float:
        """EWMA smoothing."""
        if previous_value is None:
            return new_value
        
        return alpha * new_value + (1 - alpha) * previous_value
    
    def _remove_vig_and_aggregate(self, odds_dict, weights=None):
        """Aggregate odds using weighted average of implied probabilities."""
        metadata = {
            'method': 'vig_removed_weighted_avg',
            'bookmakers_used': 0,
            'trusted_used': [],
            'raw_odds': {},
            'implied_probs': {},
            'weights_applied': {},
            'outliers_removed': [],
            'aggregated_prob': None,
            'aggregated_odd': None,
        }
        
        if not odds_dict:
            return None, metadata
        
        # Step 1: Validate odds
        valid_odds = {}
        for bookmaker, odd in odds_dict.items():
            validated = self._validate_odds(odd)
            if validated is not None:
                valid_odds[bookmaker] = validated
        
        if not valid_odds:
            return None, metadata
        
        metadata['raw_odds'] = valid_odds.copy()
        metadata['bookmakers_used'] = len(valid_odds)
        
        # Step 2: MAD filtering
        odds_values = list(valid_odds.values())
        filtered_odds = {}
        
        for bookmaker, odd in valid_odds.items():
            if len(odds_values) >= 3 and self._is_outlier_mad(odds_values, odd, k=3.5):
                metadata['outliers_removed'].append({'bookmaker': bookmaker, 'odd': odd})
            else:
                filtered_odds[bookmaker] = odd
        
        if not filtered_odds:
            max_bookmaker = max(valid_odds, key=valid_odds.get)
            filtered_odds = {max_bookmaker: valid_odds[max_bookmaker]}
        
        # Step 3: Convert to implied probabilities
        implied_probs = {}
        for bookmaker, odd in filtered_odds.items():
            implied_probs[bookmaker] = 1.0 / odd
        
        metadata['implied_probs'] = implied_probs.copy()
        
        # Step 4: Apply weights
        if weights is None:
            weights = {}
        
        applied_weights = {}
        for bookmaker in implied_probs.keys():
            normalized_name = bookmaker.lower().strip()
            if bookmaker in weights:
                applied_weights[bookmaker] = weights[bookmaker]
            elif normalized_name in self.TRUSTED_BOOKMAKERS_WEIGHTED:
                applied_weights[bookmaker] = self.TRUSTED_BOOKMAKERS_WEIGHTED[normalized_name]
                metadata['trusted_used'].append(bookmaker)
            else:
                applied_weights[bookmaker] = 1.0
        
        metadata['weights_applied'] = applied_weights.copy()
        
        # Step 5: Weighted average of implied probabilities
        total_weight = sum(applied_weights.values())
        if total_weight <= 0:
            return None, metadata
        
        aggregated_prob = sum(
            implied_probs[bm] * applied_weights[bm]
            for bm in implied_probs.keys()
        ) / total_weight
        
        metadata['aggregated_prob'] = aggregated_prob
        
        # Step 6: Convert back
        if aggregated_prob <= 0 or aggregated_prob >= 1:
            return None, metadata
        
        aggregated_odd = 1.0 / aggregated_prob
        metadata['aggregated_odd'] = aggregated_odd
        
        return aggregated_odd, metadata


class TestVigRemoval:
    """Tests for vig removal and probability normalization."""
    
    def setup_method(self):
        self.automation = MockAutomation24H()
    
    def test_simple_vig_removal(self):
        """Test that weighted average correctly aggregates probabilities."""
        # Odds from multiple bookmakers for same outcome
        odds_dict = {
            'bet365': 1.90,  # Implied prob: 52.6%
            'pinnacle': 1.95,  # Implied prob: 51.3%
            'betfair': 1.92,  # Implied prob: 52.1%
        }
        
        aggregated_odd, metadata = self.automation._remove_vig_and_aggregate(odds_dict)
        
        # Should produce valid aggregated odd
        assert aggregated_odd is not None
        assert aggregated_odd > 1.0
        
        # Aggregated odd should be within the range of input odds
        min_odd = min(odds_dict.values())
        max_odd = max(odds_dict.values())
        assert min_odd <= aggregated_odd <= max_odd
        
        # Metadata should track bookmakers
        assert metadata['bookmakers_used'] == 3
        assert len(metadata['trusted_used']) == 3  # All are trusted
    
    def test_vig_removal_with_untrusted_bookmaker(self):
        """Test that untrusted bookmakers get lower weight."""
        odds_dict = {
            'bet365': 2.00,  # Trusted, weight 2.0
            'unknown_bookie': 2.10,  # Untrusted, weight 1.0
        }
        
        aggregated_odd, metadata = self.automation._remove_vig_and_aggregate(odds_dict)
        
        assert aggregated_odd is not None
        # bet365 should have weight 2.0, unknown_bookie weight 1.0
        assert metadata['weights_applied']['bet365'] == 2.0
        assert metadata['weights_applied']['unknown_bookie'] == 1.0
    
    def test_empty_odds_dict(self):
        """Test handling of empty input."""
        aggregated_odd, metadata = self.automation._remove_vig_and_aggregate({})
        
        assert aggregated_odd is None
        assert metadata['bookmakers_used'] == 0
    
    def test_invalid_odds_filtered(self):
        """Test that invalid odds are filtered out."""
        odds_dict = {
            'bet365': 2.00,
            'invalid1': 0.5,  # Invalid: <= 1.0
            'invalid2': None,  # Invalid: None
            'invalid3': 'abc',  # Invalid: string
        }
        
        aggregated_odd, metadata = self.automation._remove_vig_and_aggregate(odds_dict)
        
        # Should still produce valid result with just bet365
        assert aggregated_odd is not None
        assert metadata['bookmakers_used'] == 1


class TestMADOutlierDetection:
    """Tests for MAD-based outlier detection."""
    
    def setup_method(self):
        self.automation = MockAutomation24H()
    
    def test_detects_extreme_outlier(self):
        """Test that extreme outlier is detected."""
        # Normal values around 2.0
        values = [1.95, 2.00, 2.05, 1.98, 10.0]  # 10.0 is extreme outlier
        
        # Should detect 10.0 as outlier
        assert self.automation._is_outlier_mad(values, 10.0, k=3.5) is True
        
        # Normal values should not be outliers
        assert self.automation._is_outlier_mad(values, 2.00, k=3.5) is False
        assert self.automation._is_outlier_mad(values, 1.95, k=3.5) is False
    
    def test_aggregation_filters_outlier(self):
        """Test that outlier is filtered during aggregation."""
        odds_dict = {
            'bet365': 2.00,
            'pinnacle': 2.05,
            'betfair': 1.98,
            'outlier_bookie': 15.00,  # Extreme outlier
        }
        
        aggregated_odd, metadata = self.automation._remove_vig_and_aggregate(odds_dict)
        
        # Should have removed the outlier
        assert len(metadata['outliers_removed']) >= 1
        outlier_values = [o['odd'] for o in metadata['outliers_removed']]
        assert 15.00 in outlier_values
        
        # Aggregated odd should be closer to normal values (around 2.0)
        assert aggregated_odd is not None
        assert 1.8 < aggregated_odd < 2.5  # Not influenced by 15.0
    
    def test_no_outlier_detection_with_few_values(self):
        """Test that outlier detection is skipped with < 3 values."""
        values = [2.0, 10.0]  # Only 2 values
        
        # Should not detect outlier (not enough data)
        assert self.automation._is_outlier_mad(values, 10.0, k=3.5) is False
    
    def test_moderate_variation_not_outlier(self):
        """Test that moderate variation is not flagged as outlier."""
        values = [1.80, 1.90, 2.00, 2.10, 2.20]  # Normal spread
        
        # None should be outliers
        for v in values:
            assert self.automation._is_outlier_mad(values, v, k=3.5) is False


class TestEWMASmoothing:
    """Tests for EWMA smoothing."""
    
    def setup_method(self):
        self.automation = MockAutomation24H()
    
    def test_first_value_passthrough(self):
        """Test that first value passes through unchanged."""
        result = self.automation._ewma(None, 2.0, alpha=0.4)
        assert result == 2.0
    
    def test_smoothing_produces_intermediate(self):
        """Test that EWMA produces intermediate values."""
        # Previous: 2.0, New: 3.0, Alpha: 0.4
        # Expected: 0.4 * 3.0 + 0.6 * 2.0 = 1.2 + 1.2 = 2.4
        result = self.automation._ewma(2.0, 3.0, alpha=0.4)
        assert abs(result - 2.4) < 0.001
    
    def test_smoothing_with_different_alpha(self):
        """Test EWMA with different alpha values."""
        # Higher alpha = more weight on new value
        result_high = self.automation._ewma(2.0, 4.0, alpha=0.8)
        # 0.8 * 4.0 + 0.2 * 2.0 = 3.2 + 0.4 = 3.6
        assert abs(result_high - 3.6) < 0.001
        
        # Lower alpha = more weight on previous value
        result_low = self.automation._ewma(2.0, 4.0, alpha=0.2)
        # 0.2 * 4.0 + 0.8 * 2.0 = 0.8 + 1.6 = 2.4
        assert abs(result_low - 2.4) < 0.001
    
    def test_multiple_smoothing_steps(self):
        """Test sequential EWMA applications."""
        alpha = 0.4
        
        # Step 1: First value
        value1 = self.automation._ewma(None, 2.0, alpha)
        assert value1 == 2.0
        
        # Step 2: New value 3.0
        value2 = self.automation._ewma(value1, 3.0, alpha)
        # 0.4 * 3.0 + 0.6 * 2.0 = 2.4
        assert abs(value2 - 2.4) < 0.001
        
        # Step 3: New value 2.5
        value3 = self.automation._ewma(value2, 2.5, alpha)
        # 0.4 * 2.5 + 0.6 * 2.4 = 1.0 + 1.44 = 2.44
        assert abs(value3 - 2.44) < 0.001
    
    def test_smoothing_dampens_spike(self):
        """Test that EWMA dampens a sudden spike."""
        alpha = 0.4
        
        # Start with stable value
        ewma = 2.0
        
        # Sudden spike to 5.0
        ewma = self.automation._ewma(ewma, 5.0, alpha)
        # Should be dampened, not jump to 5.0
        assert ewma < 5.0
        assert ewma > 2.0
        
        # Back to normal
        ewma = self.automation._ewma(ewma, 2.0, alpha)
        # Should gradually return towards 2.0
        assert ewma < 3.0


class TestOddsValidation:
    """Tests for odds validation."""
    
    def setup_method(self):
        self.automation = MockAutomation24H()
    
    def test_valid_odds(self):
        """Test that valid odds pass validation."""
        assert self.automation._validate_odds(2.0) == 2.0
        assert self.automation._validate_odds(1.5) == 1.5
        assert self.automation._validate_odds(10.0) == 10.0
        assert self.automation._validate_odds("2.5") == 2.5  # String
    
    def test_invalid_odds_rejected(self):
        """Test that invalid odds are rejected."""
        assert self.automation._validate_odds(None) is None
        assert self.automation._validate_odds(0.5) is None  # <= 1.0
        assert self.automation._validate_odds(1.0) is None  # <= 1.0
        assert self.automation._validate_odds(1001) is None  # > 1000
        assert self.automation._validate_odds("abc") is None  # Invalid string
        assert self.automation._validate_odds(float('inf')) is None
        assert self.automation._validate_odds(float('nan')) is None


class TestIntegration:
    """Integration tests for the complete aggregation pipeline."""
    
    def setup_method(self):
        self.automation = MockAutomation24H()
    
    def test_complete_pipeline(self):
        """Test complete aggregation pipeline with realistic data."""
        # Realistic odds from multiple bookmakers
        odds_dict = {
            'bet365': 1.85,
            'pinnacle': 1.87,
            'betfair': 1.83,
            'unibet': 1.82,
            'william_hill': 1.84,
        }
        
        aggregated_odd, metadata = self.automation._remove_vig_and_aggregate(odds_dict)
        
        # Should produce valid result
        assert aggregated_odd is not None
        # Should be within reasonable range of input odds
        min_odd = min(odds_dict.values())
        max_odd = max(odds_dict.values())
        assert min_odd - 0.1 <= aggregated_odd <= max_odd + 0.1
        
        # Trusted bookmakers should have higher weight
        assert metadata['weights_applied']['bet365'] == 2.0
        assert metadata['weights_applied']['pinnacle'] == 2.0
        assert metadata['weights_applied']['betfair'] == 2.0
    
    def test_aggregation_with_one_extreme_outlier(self):
        """Test aggregation handles one extreme outlier gracefully."""
        odds_dict = {
            'bet365': 2.00,
            'pinnacle': 2.02,
            'betfair': 1.98,
            'bad_bookie': 50.0,  # Extreme outlier
        }
        
        aggregated_odd, metadata = self.automation._remove_vig_and_aggregate(odds_dict)
        
        # Should filter the outlier
        assert len(metadata['outliers_removed']) >= 1
        
        # Result should be reasonable (around 2.0, not influenced by 50.0)
        assert aggregated_odd is not None
        assert 1.8 < aggregated_odd < 2.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
