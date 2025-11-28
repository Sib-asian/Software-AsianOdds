"""
Test per verificare che Champions League femminile ed Europa Cup Women siano accettate
"""
import sys
from live_betting_advisor import LiveBettingAdvisor

def test_women_tournaments():
    """Test: Verifica che Champions League femminile ed Europa Cup Women siano accettate"""
    print("=" * 70)
    print("🧪 TEST: TORNEI FEMMINILI IMPORTANTI")
    print("=" * 70)
    
    advisor = LiveBettingAdvisor()
    
    # Test cases: tornei femminili importanti (dovrebbero essere ACCETTATI)
    test_cases_accepted = [
        {
            'name': 'Champions League Women',
            'match_data': {
                'home': 'Barcelona Women',
                'away': 'Lyon Women',
                'league': 'UEFA Champions League Women'
            }
        },
        {
            'name': 'Champions League Women (variante)',
            'match_data': {
                'home': 'Chelsea Women',
                'away': 'Arsenal Women',
                'league': 'Champions League Women'
            }
        },
        {
            'name': 'Europa Cup Women',
            'match_data': {
                'home': 'Roma Women',
                'away': 'Juventus Women',
                'league': 'Europa Cup Women'
            }
        },
        {
            'name': 'Europa League Women',
            'match_data': {
                'home': 'Wolfsburg Women',
                'away': 'PSG Women',
                'league': 'UEFA Europa League Women'
            }
        },
        {
            'name': 'Women Champions League',
            'match_data': {
                'home': 'Real Madrid Women',
                'away': 'Bayern Women',
                'league': 'Women Champions League'
            }
        }
    ]
    
    # Test cases: altri campionati femminili (dovrebbero essere ESCLUSI)
    test_cases_rejected = [
        {
            'name': 'Serie A Femminile',
            'match_data': {
                'home': 'Juventus Women',
                'away': 'Inter Women',
                'league': 'Serie A Femminile'
            }
        },
        {
            'name': 'Women Super League',
            'match_data': {
                'home': 'Arsenal Women',
                'away': 'Chelsea Women',
                'league': 'Women Super League'
            }
        },
        {
            'name': 'Primera Division Femenina',
            'match_data': {
                'home': 'Barcelona Women',
                'away': 'Real Madrid Women',
                'league': 'Primera Division Femenina'
            }
        }
    ]
    
    print("\n✅ TEST: Tornei femminili importanti (dovrebbero essere ACCETTATI)")
    print("-" * 70)
    all_accepted_ok = True
    for test_case in test_cases_accepted:
        result = advisor._is_match_worth_analyzing(test_case['match_data'])
        if result:
            print(f"✅ {test_case['name']}: ACCETTATO (corretto)")
        else:
            print(f"❌ {test_case['name']}: RIFIUTATO (ERRORE - dovrebbe essere accettato)")
            all_accepted_ok = False
    
    print("\n❌ TEST: Altri campionati femminili (dovrebbero essere ESCLUSI)")
    print("-" * 70)
    all_rejected_ok = True
    for test_case in test_cases_rejected:
        result = advisor._is_match_worth_analyzing(test_case['match_data'])
        if not result:
            print(f"✅ {test_case['name']}: ESCLUSO (corretto)")
        else:
            print(f"❌ {test_case['name']}: ACCETTATO (ERRORE - dovrebbe essere escluso)")
            all_rejected_ok = False
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO")
    print("=" * 70)
    
    if all_accepted_ok and all_rejected_ok:
        print("✅ TUTTI I TEST PASSATI!")
        print("\n✅ Champions League femminile: ACCETTATA")
        print("✅ Europa Cup Women: ACCETTATA")
        print("✅ Altri campionati femminili: ESCLUSI (corretto)")
        return 0
    else:
        print("❌ ALCUNI TEST FALLITI")
        if not all_accepted_ok:
            print("   - Alcuni tornei importanti sono stati rifiutati")
        if not all_rejected_ok:
            print("   - Alcuni campionati minori sono stati accettati")
        return 1

if __name__ == '__main__':
    sys.exit(test_women_tournaments())

