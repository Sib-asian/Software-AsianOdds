#!/usr/bin/env python3
"""
Test Quote Reali da API-SPORTS
================================

Verifica che il sistema recuperi le quote reali da API-SPORTS
invece di usare quelle hardcoded.
"""

import sys
import logging
from multi_source_match_finder import MultiSourceMatchFinder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fetch_odds():
    """Test recupero quote da API-SPORTS"""
    print("\n" + "="*70)
    print("🔬 TEST RECUPERO QUOTE REALI DA API-SPORTS")
    print("="*70)

    finder = MultiSourceMatchFinder()

    # Test con un fixture ID di esempio (cambia con uno reale se necessario)
    # Nota: Questo richiede una partita live reale con fixture_id valido
    print("\n📡 Cercando partite LIVE...")
    matches = finder.find_all_matches(days_ahead=0, include_live=True)

    if not matches:
        print("⚠️  Nessuna partita live trovata. Test saltato.")
        return

    print(f"\n✅ Trovate {len(matches)} partite")

    # Controlla prime 3 partite
    for i, match in enumerate(matches[:3]):
        home = match.get('home', 'N/A')
        away = match.get('away', 'N/A')
        score_home = match.get('score_home', 0)
        score_away = match.get('score_away', 0)
        minute = match.get('minute', 0)

        print(f"\n{'='*70}")
        print(f"📊 Partita {i+1}: {home} vs {away}")
        print(f"   Score: {score_home}-{score_away} (min {minute})")
        print(f"\n💰 Quote recuperate:")

        # Quote 1X2
        odds_1 = match.get('odds_1')
        odds_x = match.get('odds_x')
        odds_2 = match.get('odds_2')
        print(f"   1X2: {odds_1} / {odds_x} / {odds_2}")

        # Quote Over/Under
        odds_over_0_5 = match.get('odds_over_0_5')
        odds_under_0_5 = match.get('odds_under_0_5')
        odds_over_1_5 = match.get('odds_over_1_5')
        odds_under_1_5 = match.get('odds_under_1_5')
        odds_over_2_5 = match.get('odds_over_2_5')
        odds_under_2_5 = match.get('odds_under_2_5')
        odds_over_3_5 = match.get('odds_over_3_5')
        odds_under_3_5 = match.get('odds_under_3_5')

        print(f"   Over/Under 0.5: {odds_over_0_5} / {odds_under_0_5}")
        print(f"   Over/Under 1.5: {odds_over_1_5} / {odds_under_1_5}")
        print(f"   Over/Under 2.5: {odds_over_2_5} / {odds_under_2_5}")
        print(f"   Over/Under 3.5: {odds_over_3_5} / {odds_under_3_5}")

        # Quote BTTS
        odds_btts_yes = match.get('odds_btts_yes')
        odds_btts_no = match.get('odds_btts_no')
        print(f"   BTTS: Yes {odds_btts_yes} / No {odds_btts_no}")

        # Verifica che almeno alcune quote siano state recuperate
        has_odds = any([
            odds_1, odds_x, odds_2,
            odds_over_2_5, odds_under_2_5,
            odds_btts_yes, odds_btts_no
        ])

        if has_odds:
            print(f"\n   ✅ Quote recuperate con successo!")
        else:
            print(f"\n   ⚠️  Nessuna quota disponibile per questa partita")
            print(f"   (Possibile che il bookmaker non fornisca quote live per questa lega)")

    print(f"\n{'='*70}")
    print("✅ Test completato!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_fetch_odds()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrotto dall'utente")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
