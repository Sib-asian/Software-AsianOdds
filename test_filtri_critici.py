"""Test rapido filtri critici"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ai_system.signal_quality_scorer import SignalQualityGate

class MockOpp:
    def __init__(self, market, conf=80, ev=10, has_stats=True):
        self.market = market
        self.confidence = conf
        self.ev = ev
        self.has_live_stats = has_stats
        self.match_stats = {}

print("Test Filtri Critici")
print("=" * 60)

gate = SignalQualityGate(min_quality_score=75.0)

# Test 1: Over 0.5 quando c'è già 1 gol
opp1 = {'match_id': 'test1', 'live_opportunity': MockOpp('over_0.5')}
match1 = {'home': 'A', 'away': 'B', 'league': 'Serie A'}
live1 = {'minute': 30, 'score_home': 1, 'score_away': 0, 'shots_home': 5, 'shots_away': 3, 'possession_home': 55}
should, score = gate.should_send_signal(opp1, match1, live1)
print(f"Test 1 - Over 0.5 con 1 gol: {'BLOCCATO' if not should else 'APPROVATO'} (score: {score.total_score})")
assert not should, "Dovrebbe essere bloccato"

# Test 2: BTTS Yes quando entrambe hanno segnato
opp2 = {'match_id': 'test2', 'live_opportunity': MockOpp('btts_yes')}
live2 = {'minute': 50, 'score_home': 1, 'score_away': 1, 'shots_home': 8, 'shots_away': 7, 'possession_home': 50}
should, score = gate.should_send_signal(opp2, match1, live2)
print(f"Test 2 - BTTS Yes con 1-1: {'BLOCCATO' if not should else 'APPROVATO'} (score: {score.total_score})")
assert not should, "Dovrebbe essere bloccato"

# Test 3: Team to Score First quando NON è 0-0
opp3 = {'match_id': 'test3', 'live_opportunity': MockOpp('team_to_score_first_home')}
live3 = {'minute': 20, 'score_home': 1, 'score_away': 0, 'shots_home': 5, 'shots_away': 2, 'possession_home': 60}
should, score = gate.should_send_signal(opp3, match1, live3)
print(f"Test 3 - Team to Score First con 1-0: {'BLOCCATO' if not should else 'APPROVATO'} (score: {score.total_score})")
assert not should, "Dovrebbe essere bloccato"

# Test 4: Next Goal oltre 85'
opp4 = {'match_id': 'test4', 'live_opportunity': MockOpp('next_goal_home')}
live4 = {'minute': 87, 'score_home': 1, 'score_away': 1, 'shots_home': 10, 'shots_away': 9, 'possession_home': 52}
should, score = gate.should_send_signal(opp4, match1, live4)
print(f"Test 4 - Next Goal al 87': {'BLOCCATO' if not should else 'APPROVATO'} (score: {score.total_score})")
assert not should, "Dovrebbe essere bloccato"

# Test 5: Clean Sheet 2-0 al 75'
opp5 = {'match_id': 'test5', 'live_opportunity': MockOpp('clean_sheet_home')}
live5 = {'minute': 75, 'score_home': 2, 'score_away': 0, 'shots_home': 12, 'shots_away': 3, 'possession_home': 65}
should, score = gate.should_send_signal(opp5, match1, live5)
print(f"Test 5 - Clean Sheet 2-0 al 75': {'BLOCCATO' if not should else 'APPROVATO'} (score: {score.total_score})")
assert not should, "Dovrebbe essere bloccato"

print("\n" + "=" * 60)
print("TUTTI I TEST PASSATI!")


