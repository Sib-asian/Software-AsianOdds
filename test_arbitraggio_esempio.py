#!/usr/bin/env python3
"""
Test Esempio Arbitraggio
========================

Mostra un esempio pratico di come funziona un arbitraggio.
"""

print("=" * 70)
print("💰 ESEMPIO PRATICO ARBITRAGGIO 1X2")
print("=" * 70)

print("\n📋 ESEMPIO: Juventus vs Inter")
print("-" * 70)

# Quote da bookmaker diversi
bookmaker_a = {
    'name': 'Bet365',
    'home': 2.30,  # Juventus
    'draw': 3.40,
    'away': 3.10   # Inter
}

bookmaker_b = {
    'name': 'Pinnacle',
    'home': 2.10,
    'draw': 3.70,  # MIGLIORE
    'away': 3.40   # MIGLIORE
}

bookmaker_c = {
    'name': 'William Hill',
    'home': 2.00,
    'draw': 3.50,
    'away': 3.20
}

print("\n📊 QUOTE DISPONIBILI:")
print(f"\n{bookmaker_a['name']}:")
print(f"  Juventus (1): {bookmaker_a['home']}")
print(f"  Pareggio (X): {bookmaker_a['draw']}")
print(f"  Inter (2):    {bookmaker_a['away']}")

print(f"\n{bookmaker_b['name']}:")
print(f"  Juventus (1): {bookmaker_b['home']}")
print(f"  Pareggio (X): {bookmaker_b['draw']} ✅ MIGLIORE")
print(f"  Inter (2):    {bookmaker_b['away']} ✅ MIGLIORE")

print(f"\n{bookmaker_c['name']}:")
print(f"  Juventus (1): {bookmaker_c['home']}")
print(f"  Pareggio (X): {bookmaker_c['draw']}")
print(f"  Inter (2):    {bookmaker_c['away']}")

# Trova migliori quote
best_home = max(bookmaker_a['home'], bookmaker_b['home'], bookmaker_c['home'])
best_draw = max(bookmaker_a['draw'], bookmaker_b['draw'], bookmaker_c['draw'])
best_away = max(bookmaker_a['away'], bookmaker_b['away'], bookmaker_c['away'])

print("\n✅ MIGLIORI QUOTE:")
print(f"  Juventus (1): {best_home} (da {bookmaker_a['name']})")
print(f"  Pareggio (X): {best_draw} (da {bookmaker_b['name']})")
print(f"  Inter (2):    {best_away} (da {bookmaker_b['name']})")

# Calcola somma probabilità
total_prob = (1/best_home) + (1/best_draw) + (1/best_away)
profit_pct = ((1/total_prob) - 1) * 100

print("\n🧮 CALCOLO ARBITRAGGIO:")
print(f"  1/{best_home} + 1/{best_draw} + 1/{best_away}")
print(f"  = {1/best_home:.4f} + {1/best_draw:.4f} + {1/best_away:.4f}")
print(f"  = {total_prob:.4f}")

if total_prob < 1.0:
    print(f"\n✅ ARBITRAGGIO TROVATO!")
    print(f"   Profitto garantito: {profit_pct:.2f}%")
    
    # Calcola stake ottimali per €100
    total_stake = 100.0
    stake_home = ((1/best_home) / total_prob) * total_stake
    stake_draw = ((1/best_draw) / total_prob) * total_stake
    stake_away = ((1/best_away) / total_prob) * total_stake
    
    print(f"\n💰 COSA PUNTARE (con €{total_stake:.2f} totali):")
    print(f"  Su {bookmaker_a['name']}:")
    print(f"    • Juventus (1): €{stake_home:.2f}")
    print(f"  Su {bookmaker_b['name']}:")
    print(f"    • Pareggio (X): €{stake_draw:.2f}")
    print(f"    • Inter (2):    €{stake_away:.2f}")
    print(f"  Totale investito: €{stake_home + stake_draw + stake_away:.2f}")
    
    print(f"\n💵 RISULTATO GARANTITO:")
    print(f"  Se Juventus vince:")
    print(f"    Vincita: €{stake_home:.2f} × {best_home} = €{stake_home * best_home:.2f}")
    print(f"    Investito: €{stake_home + stake_draw + stake_away:.2f}")
    print(f"    Profitto: €{(stake_home * best_home) - (stake_home + stake_draw + stake_away):.2f}")
    
    print(f"\n  Se Pareggio:")
    print(f"    Vincita: €{stake_draw:.2f} × {best_draw} = €{stake_draw * best_draw:.2f}")
    print(f"    Investito: €{stake_home + stake_draw + stake_away:.2f}")
    print(f"    Profitto: €{(stake_draw * best_draw) - (stake_home + stake_draw + stake_away):.2f}")
    
    print(f"\n  Se Inter vince:")
    print(f"    Vincita: €{stake_away:.2f} × {best_away} = €{stake_away * best_away:.2f}")
    print(f"    Investito: €{stake_home + stake_draw + stake_away:.2f}")
    print(f"    Profitto: €{(stake_away * best_away) - (stake_home + stake_draw + stake_away):.2f}")
    
    print(f"\n✅ QUALSIASI RISULTATO → PROFITTO GARANTITO DI €{total_stake * (profit_pct/100):.2f} ({profit_pct:.2f}%)!")
    
else:
    print(f"\n❌ NON C'È ARBITRAGGIO")
    print(f"   Somma: {total_prob:.4f} (deve essere < 1.0)")
    print(f"   Mancano {((total_prob - 1.0) * 100):.2f}% per avere arbitraggio")

print("\n" + "=" * 70)
print("💡 REGOLA SEMPLICE:")
print("=" * 70)
print("""
1. Il sistema trova le MIGLIORI quote da bookmaker diversi
2. Calcola: 1/quota_1 + 1/quota_X + 1/quota_2
3. Se < 1.0 → C'è arbitraggio!
4. Ti dice ESATTAMENTE cosa puntare e dove
5. Punti su TUTTI i risultati su bookmaker DIVERSI
6. GUADAGNI SEMPRE la percentuale indicata!

È come comprare un biglietto della lotteria dove VINCI SEMPRE! 🎯
""")

