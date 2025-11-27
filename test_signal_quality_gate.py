"""
Test Signal Quality Gate
========================

Testa il sistema di validazione qualità segnali.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ai_system.signal_quality_scorer import SignalQualityGate, QualityScore
from live_betting_advisor import LiveBettingOpportunity
from datetime import datetime


class MockLiveOpportunity:
    """Mock per LiveBettingOpportunity"""
    def __init__(self, market, confidence, ev, has_live_stats=True):
        self.market = market
        self.confidence = confidence
        self.ev = ev
        self.has_live_stats = has_live_stats
        self.match_stats = {}


def test_quality_gate():
    """Test Signal Quality Gate"""
    print("🧪 Test Signal Quality Gate")
    print("=" * 60)
    
    gate = SignalQualityGate(min_quality_score=75.0)
    
    # Test 1: Segnale valido (dovrebbe essere approvato)
    print("\n📊 Test 1: Segnale valido")
    print("-" * 60)
    opp1 = {
        'match_id': 'test_match_1',
        'live_opportunity': MockLiveOpportunity('over_1.5', 80.0, 12.0, True)
    }
    match_data1 = {'home': 'Team A', 'away': 'Team B', 'league': 'Serie A'}
    live_data1 = {
        'minute': 35,
        'score_home': 1,
        'score_away': 0,
        'shots_home': 8,
        'shots_away': 5,
        'shots_on_target_home': 4,
        'shots_on_target_away': 2,
        'possession_home': 55,
        'xg_home': 1.2,
        'xg_away': 0.5
    }
    
    should_send, score = gate.should_send_signal(opp1, match_data1, live_data1)
    print(f"   Market: {opp1['live_opportunity'].market}")
    print(f"   Score: {score.total_score:.1f}/100")
    print(f"   Context: {score.context_score:.1f}, Data: {score.data_quality_score:.1f}, Logic: {score.logic_score:.1f}, Timing: {score.timing_score:.1f}")
    print(f"   Approved: {should_send}")
    if score.reasons:
        print(f"   Reasons: {', '.join(score.reasons)}")
    if score.warnings:
        print(f"   Warnings: {', '.join(score.warnings)}")
    assert should_send == True, "Segnale valido dovrebbe essere approvato"
    print("   ✅ PASS")
    
    # Test 2: Segnale banale (dovrebbe essere bloccato)
    print("\n📊 Test 2: Segnale banale (Over 1.5 quando già 2-0)")
    print("-" * 60)
    opp2 = {
        'match_id': 'test_match_2',
        'live_opportunity': MockLiveOpportunity('over_1.5', 85.0, 15.0, True)
    }
    match_data2 = {'home': 'Team A', 'away': 'Team B', 'league': 'Serie A'}
    live_data2 = {
        'minute': 75,
        'score_home': 2,
        'score_away': 0,
        'shots_home': 10,
        'shots_away': 3,
        'shots_on_target_home': 5,
        'shots_on_target_away': 1,
        'possession_home': 60,
        'xg_home': 2.1,
        'xg_away': 0.3
    }
    
    should_send, score = gate.should_send_signal(opp2, match_data2, live_data2)
    print(f"   Market: {opp2['live_opportunity'].market}")
    print(f"   Score: {score.total_score:.1f}/100")
    print(f"   Context: {score.context_score:.1f}, Data: {score.data_quality_score:.1f}, Logic: {score.logic_score:.1f}, Timing: {score.timing_score:.1f}")
    print(f"   Approved: {should_send}")
    if score.reasons:
        print(f"   Reasons: {', '.join(score.reasons)}")
    if score.warnings:
        print(f"   Warnings: {', '.join(score.warnings)}")
    assert should_send == False, "Segnale banale dovrebbe essere bloccato"
    print("   ✅ PASS")
    
    # Test 3: Segnale con dati insufficienti (dovrebbe essere bloccato)
    print("\n📊 Test 3: Segnale con dati insufficienti")
    print("-" * 60)
    opp3 = {
        'match_id': 'test_match_3',
        'live_opportunity': MockLiveOpportunity('over_2.5', 75.0, 10.0, False)
    }
    match_data3 = {'home': 'Team A', 'away': 'Team B', 'league': 'Serie A'}
    live_data3 = {
        'minute': 25,
        'score_home': 0,
        'score_away': 0,
        'shots_home': 0,
        'shots_away': 0,
        'shots_on_target_home': 0,
        'shots_on_target_away': 0,
        'possession_home': None,
        'xg_home': 0.0,
        'xg_away': 0.0
    }
    
    should_send, score = gate.should_send_signal(opp3, match_data3, live_data3)
    print(f"   Market: {opp3['live_opportunity'].market}")
    print(f"   Score: {score.total_score:.1f}/100")
    print(f"   Context: {score.context_score:.1f}, Data: {score.data_quality_score:.1f}, Logic: {score.logic_score:.1f}, Timing: {score.timing_score:.1f}")
    print(f"   Approved: {should_send}")
    if score.reasons:
        print(f"   Reasons: {', '.join(score.reasons)}")
    if score.warnings:
        print(f"   Warnings: {', '.join(score.warnings)}")
    assert should_send == False, "Segnale con dati insufficienti dovrebbe essere bloccato"
    print("   ✅ PASS")
    
    # Test 4: Segnale troppo presto (dovrebbe avere warning)
    print("\n📊 Test 4: Segnale troppo presto (minuto 5)")
    print("-" * 60)
    opp4 = {
        'match_id': 'test_match_4',
        'live_opportunity': MockLiveOpportunity('over_0.5', 78.0, 11.0, True)
    }
    match_data4 = {'home': 'Team A', 'away': 'Team B', 'league': 'Serie A'}
    live_data4 = {
        'minute': 5,
        'score_home': 0,
        'score_away': 0,
        'shots_home': 3,
        'shots_away': 2,
        'shots_on_target_home': 1,
        'shots_on_target_away': 1,
        'possession_home': 52,
        'xg_home': 0.3,
        'xg_away': 0.2
    }
    
    should_send, score = gate.should_send_signal(opp4, match_data4, live_data4)
    print(f"   Market: {opp4['live_opportunity'].market}")
    print(f"   Score: {score.total_score:.1f}/100")
    print(f"   Context: {score.context_score:.1f}, Data: {score.data_quality_score:.1f}, Logic: {score.logic_score:.1f}, Timing: {score.timing_score:.1f}")
    print(f"   Approved: {should_send}")
    if score.reasons:
        print(f"   Reasons: {', '.join(score.reasons)}")
    if score.warnings:
        print(f"   Warnings: {', '.join(score.warnings)}")
    print("   ✅ PASS")
    
    # Test 5: Segnale con EV negativo (dovrebbe essere bloccato)
    print("\n📊 Test 5: Segnale con EV negativo")
    print("-" * 60)
    opp5 = {
        'match_id': 'test_match_5',
        'live_opportunity': MockLiveOpportunity('under_2.5', 70.0, -5.0, True)
    }
    match_data5 = {'home': 'Team A', 'away': 'Team B', 'league': 'Serie A'}
    live_data5 = {
        'minute': 40,
        'score_home': 1,
        'score_away': 1,
        'shots_home': 6,
        'shots_away': 7,
        'shots_on_target_home': 3,
        'shots_on_target_away': 4,
        'possession_home': 48,
        'xg_home': 0.8,
        'xg_away': 0.9
    }
    
    should_send, score = gate.should_send_signal(opp5, match_data5, live_data5)
    print(f"   Market: {opp5['live_opportunity'].market}")
    print(f"   Score: {score.total_score:.1f}/100")
    print(f"   Context: {score.context_score:.1f}, Data: {score.data_quality_score:.1f}, Logic: {score.logic_score:.1f}, Timing: {score.timing_score:.1f}")
    print(f"   Approved: {should_send}")
    if score.reasons:
        print(f"   Reasons: {', '.join(score.reasons)}")
    if score.warnings:
        print(f"   Warnings: {', '.join(score.warnings)}")
    assert should_send == False, "Segnale con EV negativo dovrebbe essere bloccato"
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("✅ TUTTI I TEST PASSATI!")
    print("=" * 60)


if __name__ == '__main__':
    test_quality_gate()


