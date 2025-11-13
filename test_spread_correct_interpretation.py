#!/usr/bin/env python3
"""
Test con INTERPRETAZIONE CORRETTA dello spread:
- Spread > 0 → Away favorita
- Spread < 0 → Home favorita
"""

import math
from scipy.stats import poisson

def calcola_lambda_CORRETTO(spread_utente, total):
    """
    Formula CORRETTA che rispetta:
    1. lambda_a - lambda_h = spread_utente (spread > 0 favorisce Away)
    2. lambda_a + lambda_h = total

    Risolvendo il sistema:
    lambda_a = (total + spread_utente) / 2
    lambda_h = (total - spread_utente) / 2
    """
    lambda_a = (total + spread_utente) / 2.0
    lambda_h = (total - spread_utente) / 2.0

    return lambda_h, lambda_a

def calc_prob_away(lambda_h, lambda_a, max_goals=10):
    """Calcola P(Away vince)."""
    prob = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h < a:
                prob += poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
    return prob

def calc_prob_away_over(lambda_h, lambda_a, soglia=2.5, max_goals=10):
    """Calcola P(Away vince & Over)."""
    prob = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h < a and (h + a) > soglia:
                prob += poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
    return prob

def calc_prob_away_multigol(lambda_h, lambda_a, gmin=2, gmax=4, max_goals=10):
    """Calcola P(Away vince & Multigol)."""
    prob = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            tot = h + a
            if h < a and gmin <= tot <= gmax:
                prob += poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
    return prob

print("="*80)
print("TEST CON INTERPRETAZIONE CORRETTA DELLO SPREAD")
print("="*80)
print("\nInterpretazione:")
print("  - Spread > 0 → Away favorita")
print("  - Spread < 0 → Home favorita")
print("  - Formula: spread = lambda_a - lambda_h")

# Test Case: spread aumenta da +0.25 a +0.50 (Away guadagna vantaggio)
spread_apertura = 0.25
spread_corrente = 0.50
total = 2.5

print(f"\n" + "="*80)
print("CASO TEST: Spread apertura +0.25 → corrente +0.50")
print("="*80)
print(f"Total fisso: {total}")

# Calcolo con APERTURA
lambda_h_ap, lambda_a_ap = calcola_lambda_CORRETTO(spread_apertura, total)

print(f"\n📊 APERTURA (spread = +{spread_apertura}):")
print(f"  Lambda_h: {lambda_h_ap:.4f}")
print(f"  Lambda_a: {lambda_a_ap:.4f}")
print(f"  Spread verificato: {lambda_a_ap - lambda_h_ap:+.4f} (dovrebbe essere +{spread_apertura})")
print(f"  Total verificato: {lambda_h_ap + lambda_a_ap:.4f} (dovrebbe essere {total})")

# Verifica
assert abs((lambda_a_ap - lambda_h_ap) - spread_apertura) < 1e-10, "Spread non rispettato!"
assert abs((lambda_h_ap + lambda_a_ap) - total) < 1e-10, "Total non rispettato!"
print(f"  ✅ Spread e total RISPETTATI!")

# Calcolo probabilità
prob_2_ap = calc_prob_away(lambda_h_ap, lambda_a_ap)
prob_2_over25_ap = calc_prob_away_over(lambda_h_ap, lambda_a_ap, 2.5)
prob_2_mg24_ap = calc_prob_away_multigol(lambda_h_ap, lambda_a_ap, 2, 4)

print(f"\n  Probabilità:")
print(f"    P(2):           {prob_2_ap:.4f} ({prob_2_ap*100:.2f}%)")
print(f"    P(2 & Over2.5): {prob_2_over25_ap:.4f} ({prob_2_over25_ap*100:.2f}%)")
print(f"    P(2 & MG 2-4):  {prob_2_mg24_ap:.4f} ({prob_2_mg24_ap*100:.2f}%)")

# Calcolo con CORRENTE
lambda_h_curr, lambda_a_curr = calcola_lambda_CORRETTO(spread_corrente, total)

print(f"\n📊 CORRENTE (spread = +{spread_corrente}):")
print(f"  Lambda_h: {lambda_h_curr:.4f}")
print(f"  Lambda_a: {lambda_a_curr:.4f}")
print(f"  Spread verificato: {lambda_a_curr - lambda_h_curr:+.4f} (dovrebbe essere +{spread_corrente})")
print(f"  Total verificato: {lambda_h_curr + lambda_a_curr:.4f} (dovrebbe essere {total})")

# Verifica
assert abs((lambda_a_curr - lambda_h_curr) - spread_corrente) < 1e-10, "Spread non rispettato!"
assert abs((lambda_h_curr + lambda_a_curr) - total) < 1e-10, "Total non rispettato!"
print(f"  ✅ Spread e total RISPETTATI!")

# Calcolo probabilità
prob_2_curr = calc_prob_away(lambda_h_curr, lambda_a_curr)
prob_2_over25_curr = calc_prob_away_over(lambda_h_curr, lambda_a_curr, 2.5)
prob_2_mg24_curr = calc_prob_away_multigol(lambda_h_curr, lambda_a_curr, 2, 4)

print(f"\n  Probabilità:")
print(f"    P(2):           {prob_2_curr:.4f} ({prob_2_curr*100:.2f}%)")
print(f"    P(2 & Over2.5): {prob_2_over25_curr:.4f} ({prob_2_over25_curr*100:.2f}%)")
print(f"    P(2 & MG 2-4):  {prob_2_mg24_curr:.4f} ({prob_2_mg24_curr*100:.2f}%)")

# Analisi variazioni
print(f"\n" + "="*80)
print("ANALISI VARIAZIONI (Apertura → Corrente)")
print("="*80)

print(f"\n🔍 Movimento spread: +{spread_apertura} → +{spread_corrente}")
print(f"  Interpretazione: Away guadagna vantaggio (spread aumenta)")

print(f"\n📈 Variazioni Lambda:")
print(f"  Lambda_h: {lambda_h_ap:.4f} → {lambda_h_curr:.4f} ({lambda_h_curr - lambda_h_ap:+.4f})")
print(f"  Lambda_a: {lambda_a_ap:.4f} → {lambda_a_curr:.4f} ({lambda_a_curr - lambda_a_ap:+.4f})")

print(f"\n📈 Variazioni Probabilità:")
delta_p2 = prob_2_curr - prob_2_ap
delta_p2_over = prob_2_over25_curr - prob_2_over25_ap
delta_p2_mg = prob_2_mg24_curr - prob_2_mg24_ap

print(f"  P(2):           {delta_p2:+.4f} ({(delta_p2/prob_2_ap)*100:+.2f}%)")
print(f"  P(2 & Over2.5): {delta_p2_over:+.4f} ({(delta_p2_over/prob_2_over25_ap)*100:+.2f}%)")
print(f"  P(2 & MG 2-4):  {delta_p2_mg:+.4f} ({(delta_p2_mg/prob_2_mg24_ap)*100:+.2f}%)")

# Verifica logica
print(f"\n" + "="*80)
print("VERIFICA LOGICA")
print("="*80)

if delta_p2 > 0 and delta_p2_over > 0 and delta_p2_mg > 0:
    print(f"\n✅ CORRETTO!")
    print(f"   Spread aumenta (+0.25 → +0.50) → Away guadagna vantaggio")
    print(f"   → P(2) aumenta ✅")
    print(f"   → P(2 & Over) aumenta ✅")
    print(f"   → P(2 & Multigol) aumenta ✅")
    print(f"\n   Comportamento atteso dall'utente: ✅ RISPETTATO")
else:
    print(f"\n❌ ERRORE! Le probabilità dovrebbero aumentare")

# Test con spread negativo (Home favorita)
print(f"\n\n" + "="*80)
print("TEST INVERSO: Spread negativo (Home favorita)")
print("="*80)

spread_neg1 = -0.25
spread_neg2 = -0.50

lambda_h_neg1, lambda_a_neg1 = calcola_lambda_CORRETTO(spread_neg1, total)
lambda_h_neg2, lambda_a_neg2 = calcola_lambda_CORRETTO(spread_neg2, total)

prob_1_neg1 = 1 - prob_2_neg1 - prob_x_neg1 if 'prob_x_neg1' in locals() else 0
# Calcolo P(Home) per spread negativo
def calc_prob_home(lambda_h, lambda_a, max_goals=10):
    prob = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h > a:
                prob += poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
    return prob

prob_1_neg1 = calc_prob_home(lambda_h_neg1, lambda_a_neg1)
prob_1_neg2 = calc_prob_home(lambda_h_neg2, lambda_a_neg2)

print(f"\nSpread = {spread_neg1} (Home leggermente favorita):")
print(f"  Lambda_h: {lambda_h_neg1:.4f}, Lambda_a: {lambda_a_neg1:.4f}")
print(f"  P(1): {prob_1_neg1:.4f} ({prob_1_neg1*100:.2f}%)")

print(f"\nSpread = {spread_neg2} (Home più favorita):")
print(f"  Lambda_h: {lambda_h_neg2:.4f}, Lambda_a: {lambda_a_neg2:.4f}")
print(f"  P(1): {prob_1_neg2:.4f} ({prob_1_neg2*100:.2f}%)")

print(f"\nMovimento: {spread_neg1} → {spread_neg2}")
print(f"  ΔP(1): {prob_1_neg2 - prob_1_neg1:+.4f} ({((prob_1_neg2/prob_1_neg1 - 1)*100):+.2f}%)")

if prob_1_neg2 > prob_1_neg1:
    print(f"  ✅ CORRETTO! Spread diminuisce (più negativo) → Home guadagna vantaggio → P(1) aumenta")
else:
    print(f"  ❌ ERRORE!")

print(f"\n" + "="*80)
print("CONCLUSIONE")
print("="*80)
print(f"\n✅ Formula CORRETTA implementata:")
print(f"   lambda_a = (total + spread) / 2")
print(f"   lambda_h = (total - spread) / 2")
print(f"\n✅ Interpretazione CORRETTA:")
print(f"   spread = lambda_a - lambda_h")
print(f"   spread > 0 → Away favorita")
print(f"   spread < 0 → Home favorita")
print(f"\n✅ Comportamento:")
print(f"   Spread aumenta → P(Away) aumenta ✅")
print(f"   Spread diminuisce → P(Home) aumenta ✅")
print(f"\n🎯 Questa è la formula che deve essere usata nel codice!")
