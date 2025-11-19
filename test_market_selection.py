#!/usr/bin/env python3
"""
Test script per verificare la logica di selezione mercato e filtri
"""

def test_market_selection():
    """Test market selection logic"""
    print("=" * 70)
    print("TEST: Market Selection Logic")
    print("=" * 70)
    
    # Simula dati match
    match = {
        'id': 'test_match_1',
        'home': 'Inter',
        'away': 'Juventus',
        'league': 'Serie A',
        'odds_1': 2.10,  # Home
        'odds_x': 3.40,  # Draw
        'odds_2': 3.20   # Away
    }
    
    # Simula risultato AI che dice HOME ha 55% di probabilità
    ai_result = {
        'summary': {
            'probability': 0.55,  # 55% per HOME
            'confidence': 75.0,
            'expected_value': 0.12  # 12% EV
        },
        'final_decision': {
            'action': 'BET'
        },
        'ensemble': {
            'uncertainty': 0.15  # 15% incertezza
        }
    }
    
    # Test calcolo probabilità implicite
    print("\n1. Calcolo Probabilità Implicite:")
    print(f"   Quote: 1={match['odds_1']:.2f} X={match['odds_x']:.2f} 2={match['odds_2']:.2f}")
    
    impl_1 = 1.0 / match['odds_1']
    impl_x = 1.0 / match['odds_x']
    impl_2 = 1.0 / match['odds_2']
    total = impl_1 + impl_x + impl_2
    margin = (total - 1.0) * 100
    
    print(f"   Prob. Implicite: 1={impl_1:.3f} ({impl_1*100:.1f}%) X={impl_x:.3f} ({impl_x*100:.1f}%) 2={impl_2:.3f} ({impl_2*100:.1f}%)")
    print(f"   Totale: {total:.3f} ({total*100:.1f}%)")
    print(f"   Margine Bookmaker: {margin:.1f}%")
    
    # Normalizza
    impl_1_norm = impl_1 / total
    impl_x_norm = impl_x / total
    impl_2_norm = impl_2 / total
    
    print(f"   Normalizzate: 1={impl_1_norm:.3f} ({impl_1_norm*100:.1f}%) X={impl_x_norm:.3f} ({impl_x_norm*100:.1f}%) 2={impl_2_norm:.3f} ({impl_2_norm*100:.1f}%)")
    
    # Test selezione mercato
    print("\n2. Selezione Miglior Mercato:")
    prob_home = ai_result['summary']['probability']
    prob_draw = min(0.30, 1.0 - prob_home)
    prob_away = max(0.0, 1.0 - prob_home - prob_draw)
    
    print(f"   Prob. AI: HOME={prob_home:.3f} ({prob_home*100:.1f}%) DRAW={prob_draw:.3f} ({prob_draw*100:.1f}%) AWAY={prob_away:.3f} ({prob_away*100:.1f}%)")
    
    ev_home = (prob_home * match['odds_1'] - 1.0) * 100
    ev_draw = (prob_draw * match['odds_x'] - 1.0) * 100
    ev_away = (prob_away * match['odds_2'] - 1.0) * 100
    
    print(f"   EV: HOME={ev_home:+.1f}% DRAW={ev_draw:+.1f}% AWAY={ev_away:+.1f}%")
    
    # Determina miglior mercato
    markets = [
        ('HOME', ev_home, prob_home, match['odds_1']),
        ('DRAW', ev_draw, prob_draw, match['odds_x']),
        ('AWAY', ev_away, prob_away, match['odds_2'])
    ]
    markets.sort(key=lambda x: x[1], reverse=True)
    
    best_market, best_ev, best_prob, best_odds = markets[0]
    print(f"\n   ✓ Miglior Mercato: {best_market}")
    print(f"     EV: {best_ev:+.1f}%")
    print(f"     Probabilità: {best_prob*100:.1f}%")
    print(f"     Quota: {best_odds:.2f}")
    
    # Test filtri
    print("\n3. Verifica Filtri:")
    
    # Filtro 1: EV minimo
    min_ev_base = 8.0
    min_ev_effective = min_ev_base + 2.0
    ev_pass = best_ev >= min_ev_effective
    print(f"   ✓ EV >= {min_ev_effective:.1f}%? {ev_pass} (EV={best_ev:.1f}%)")
    
    # Filtro 2: Confidence
    confidence = ai_result['summary']['confidence']
    conf_pass = confidence >= 70.0
    print(f"   ✓ Confidence >= 70%? {conf_pass} (Conf={confidence:.1f}%)")
    
    # Filtro 3: Edge minimo
    implied_prob = 1.0 / best_odds
    edge = best_prob - implied_prob
    edge_pass = edge >= 0.08
    print(f"   ✓ Edge >= 8%? {edge_pass} (Edge={edge*100:.1f}%)")
    
    # Filtro 4: Uncertainty
    uncertainty = ai_result['ensemble']['uncertainty']
    unc_pass = uncertainty <= 0.20
    print(f"   ✓ Uncertainty <= 20%? {unc_pass} (Unc={uncertainty*100:.1f}%)")
    
    # Filtro 5: Odds range
    odds_pass = 1.30 <= best_odds <= 5.0
    print(f"   ✓ Quote 1.30-5.0? {odds_pass} (Quota={best_odds:.2f})")
    
    # Filtro 6: Probabilità range
    prob_pass = 0.30 <= best_prob <= 0.75
    print(f"   ✓ Prob 30%-75%? {prob_pass} (Prob={best_prob*100:.1f}%)")
    
    all_pass = ev_pass and conf_pass and edge_pass and unc_pass and odds_pass and prob_pass
    
    print(f"\n   {'✅ PASSA TUTTI I FILTRI' if all_pass else '❌ NON PASSA I FILTRI'}")
    
    # Test scenario troppi segnali
    print("\n" + "=" * 70)
    print("TEST: Scenario con Troppi Segnali (Weak Opportunity)")
    print("=" * 70)
    
    weak_ai_result = {
        'summary': {
            'probability': 0.40,  # 40% - al limite
            'confidence': 72.0,   # Appena sopra soglia
            'expected_value': 0.09  # 9% EV - vicino alla soglia
        },
        'final_decision': {
            'action': 'BET'
        },
        'ensemble': {
            'uncertainty': 0.22  # 22% incertezza - TROPPA
        }
    }
    
    print("\n   Probabilità: 40% (limite basso)")
    print("   Confidence: 72% (appena sopra 70%)")
    print("   EV: 9% (sotto soglia effettiva di 10%)")
    print("   Uncertainty: 22% (SOPRA limite 20%)")
    
    ev_weak = weak_ai_result['summary']['expected_value'] * 100
    conf_weak = weak_ai_result['summary']['confidence']
    unc_weak = weak_ai_result['ensemble']['uncertainty']
    
    weak_pass = (
        ev_weak >= min_ev_effective and
        conf_weak >= 70.0 and
        unc_weak <= 0.20
    )
    
    print(f"\n   {'✅ Passa filtri' if weak_pass else '❌ NON passa filtri (come previsto)'}")
    
    if not weak_pass:
        if ev_weak < min_ev_effective:
            print(f"   Motivo: EV troppo basso ({ev_weak:.1f}% < {min_ev_effective:.1f}%)")
        if unc_weak > 0.20:
            print(f"   Motivo: Incertezza troppo alta ({unc_weak*100:.1f}% > 20%)")
    
    print("\n" + "=" * 70)
    print("✅ Test completati!")
    print("=" * 70)

if __name__ == '__main__':
    test_market_selection()
