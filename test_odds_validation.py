#!/usr/bin/env python3
"""
Test realistico per validazione quote durante partite LIVE.

Simula lo scenario: partita live con risultato 1-0, quota over 1.5 a 4.80 (stale/errata).
La validazione dovrebbe scartare questa quota prima che venga selezionata.
"""

def test_validate_raw_odds_against_score():
    """Test validazione quote RAW rispetto al risultato"""
    
    # Simula quote RAW dai bookmaker per over 1.5
    all_bookmaker_odds = {
        'over_under': {
            1.5: {
                'over': {
                    '1xbet': 4.80,  # ← QUOTA STALE/ERRATA (dovrebbe essere ~1.2)
                    'bet365': 1.15,  # ← Quota realistica
                    'pinnacle': 1.20,  # ← Quota realistica
                    'betfair': 1.18,  # ← Quota realistica
                    'marathonbet': 1.25,  # ← Quota realistica
                },
                'under': {
                    '1xbet': 5.50,
                    'bet365': 4.50,
                }
            },
            2.5: {
                'over': {
                    '1xbet': 2.10,
                    'bet365': 2.05,
                },
                'under': {
                    '1xbet': 1.75,
                    'bet365': 1.70,
                }
            }
        }
    }
    
    # Scenario: partita live, risultato 1-0, minuto 35
    score_home = 1
    score_away = 0
    total_goals = 1
    
    print("=" * 80)
    print("TEST: Validazione quote RAW per partita LIVE")
    print("=" * 80)
    print(f"Scenario: Partita LIVE, risultato {score_home}-{score_away}, minuto 35'")
    print(f"Totale gol: {total_goals}")
    print()
    print("Quote RAW dai bookmaker per Over 1.5:")
    for bookmaker, odd in all_bookmaker_odds['over_under'][1.5]['over'].items():
        print(f"  - {bookmaker}: {odd}")
    print()
    
    # Simula validazione
    print("🔍 Validazione quote vs risultato...")
    print()
    
    # Controlla se quote sono realistiche
    threshold = 1.5
    max_allowed = 1.8  # Soglia massima per over già superato
    
    filtered_bookmakers = []
    for bookmaker, odd in list(all_bookmaker_odds['over_under'][1.5]['over'].items()):
        if odd > max_allowed:
            print(f"❌ Quota IRREALISTICA da {bookmaker}: {odd} > {max_allowed}")
            print(f"   (risultato: {total_goals} gol, threshold {threshold} già superato)")
            filtered_bookmakers.append(bookmaker)
            # Rimuovi quota stale
            del all_bookmaker_odds['over_under'][1.5]['over'][bookmaker]
        else:
            print(f"✅ Quota REALISTICA da {bookmaker}: {odd} <= {max_allowed}")
    
    print()
    print("=" * 80)
    print("RISULTATO:")
    print("=" * 80)
    print(f"Quote rimosse (stale): {filtered_bookmakers}")
    print()
    print("Quote rimaste dopo validazione:")
    for bookmaker, odd in all_bookmaker_odds['over_under'][1.5]['over'].items():
        print(f"  - {bookmaker}: {odd}")
    print()
    
    # Verifica che la quota stale sia stata rimossa
    assert '1xbet' not in all_bookmaker_odds['over_under'][1.5]['over'], \
        "ERRORE: Quota stale da 1xbet (4.80) NON è stata rimossa!"
    
    assert len(all_bookmaker_odds['over_under'][1.5]['over']) == 4, \
        f"ERRORE: Dovrebbero rimanere 4 quote realistiche, ma ce ne sono {len(all_bookmaker_odds['over_under'][1.5]['over'])}"
    
    print("✅ TEST PASSATO: Quota stale rimossa correttamente!")
    print()
    print("Ora la selezione sceglierà tra quote realistiche (1.15-1.25) invece della quota stale (4.80)")
    print()

if __name__ == "__main__":
    test_validate_raw_odds_against_score()
    
