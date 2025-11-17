#!/usr/bin/env python3
"""
Test specifico per verificare movimento spread 0.75 → 1.25
e impatto su mercati "2 + multigol" e "2 + over"
"""

import math

# Simulazione calcolo lambda da spread
def calcola_lambda_da_spread(spread, total, home_advantage=1.30):
    """Calcola lambda_h e lambda_a da spread e total."""
    lambda_total = total / 2.0
    spread_factor = math.exp(spread * 0.5)
    sqrt_ha = math.sqrt(home_advantage)

    lambda_h = lambda_total * spread_factor * sqrt_ha
    lambda_a = lambda_total / spread_factor / sqrt_ha

    return lambda_h, lambda_a

# Caso 1: Spread apertura = 0.75
spread_apertura = 0.75
total_apertura = 2.5

lambda_h_ap, lambda_a_ap = calcola_lambda_da_spread(spread_apertura, total_apertura)

print("="*80)
print("TEST MOVIMENTO SPREAD: 0.75 → 1.25")
print("="*80)

print(f"\n📊 SITUAZIONE APERTURA:")
print(f"  Spread apertura: {spread_apertura}")
print(f"  Total apertura: {total_apertura}")
print(f"  Lambda_h apertura: {lambda_h_ap:.4f}")
print(f"  Lambda_a apertura: {lambda_a_ap:.4f}")
print(f"  Spread verificato: {lambda_h_ap - lambda_a_ap:.4f}")
print(f"  Total verificato: {lambda_h_ap + lambda_a_ap:.4f}")

# Caso 2: Spread corrente = 1.25
spread_corrente = 1.25
total_corrente = 2.5

lambda_h_curr, lambda_a_curr = calcola_lambda_da_spread(spread_corrente, total_corrente)

print(f"\n📊 SITUAZIONE CORRENTE:")
print(f"  Spread corrente: {spread_corrente}")
print(f"  Total corrente: {total_corrente}")
print(f"  Lambda_h corrente: {lambda_h_curr:.4f}")
print(f"  Lambda_a corrente: {lambda_a_curr:.4f}")
print(f"  Spread verificato: {lambda_h_curr - lambda_a_curr:.4f}")
print(f"  Total verificato: {lambda_h_curr + lambda_a_curr:.4f}")

# Calcola movimento
movement_spread = abs(spread_corrente - spread_apertura)
print(f"\n📈 MOVIMENTO:")
print(f"  Movement spread: {movement_spread}")
print(f"  Lambda_h variazione: {lambda_h_curr - lambda_h_ap:+.4f} ({((lambda_h_curr/lambda_h_ap - 1) * 100):+.1f}%)")
print(f"  Lambda_a variazione: {lambda_a_curr - lambda_a_ap:+.4f} ({((lambda_a_curr/lambda_a_ap - 1) * 100):+.1f}%)")

# Determina pesi basati su movimento
if movement_spread < 0.2:
    weight_apertura = 0.70
    weight_corrente = 0.30
    movement_type = "STABLE"
elif movement_spread < 0.4:
    weight_apertura = 0.50
    weight_corrente = 0.50
    movement_type = "MODERATE"
else:
    weight_apertura = 0.30
    weight_corrente = 0.70
    movement_type = "HIGH_SMART_MONEY"

print(f"\n⚖️ MOVEMENT FACTOR:")
print(f"  Type: {movement_type}")
print(f"  Weight apertura: {weight_apertura:.0%}")
print(f"  Weight corrente: {weight_corrente:.0%}")

# Blend finale
lambda_h_blend = weight_apertura * lambda_h_ap + weight_corrente * lambda_h_curr
lambda_a_blend = weight_apertura * lambda_a_ap + weight_corrente * lambda_a_curr

print(f"\n🎯 LAMBDA BLENDED:")
print(f"  Lambda_h blend: {lambda_h_blend:.4f}")
print(f"  Lambda_a blend: {lambda_a_blend:.4f}")
print(f"  Spread blend: {lambda_h_blend - lambda_a_blend:.4f}")
print(f"  Total blend: {lambda_h_blend + lambda_a_blend:.4f}")

# Calcola probabilità approssimate P(2) usando Poisson
from scipy.stats import poisson

def calc_prob_away_from_lambda(lambda_h, lambda_a, max_goals=10):
    """Calcola P(Away vince) = P(h < a) dalla distribuzione Poisson."""
    prob_away = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h < a:
                p_h = poisson.pmf(h, lambda_h)
                p_a = poisson.pmf(a, lambda_a)
                prob_away += p_h * p_a
    return prob_away

def calc_prob_away_over(lambda_h, lambda_a, soglia=2.5, max_goals=10):
    """Calcola P(Away vince & Over soglia)."""
    prob = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h < a and (h + a) > soglia:
                p_h = poisson.pmf(h, lambda_h)
                p_a = poisson.pmf(a, lambda_a)
                prob += p_h * p_a
    return prob

def calc_prob_away_multigol(lambda_h, lambda_a, gmin=2, gmax=4, max_goals=10):
    """Calcola P(Away vince & Multigol)."""
    prob = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            tot = h + a
            if h < a and gmin <= tot <= gmax:
                p_h = poisson.pmf(h, lambda_h)
                p_a = poisson.pmf(a, lambda_a)
                prob += p_h * p_a
    return prob

# Calcola probabilità con lambda APERTURA
prob_2_ap = calc_prob_away_from_lambda(lambda_h_ap, lambda_a_ap)
prob_2_over25_ap = calc_prob_away_over(lambda_h_ap, lambda_a_ap, soglia=2.5)
prob_2_mg24_ap = calc_prob_away_multigol(lambda_h_ap, lambda_a_ap, gmin=2, gmax=4)

# Calcola probabilità con lambda CORRENTE
prob_2_curr = calc_prob_away_from_lambda(lambda_h_curr, lambda_a_curr)
prob_2_over25_curr = calc_prob_away_over(lambda_h_curr, lambda_a_curr, soglia=2.5)
prob_2_mg24_curr = calc_prob_away_multigol(lambda_h_curr, lambda_a_curr, gmin=2, gmax=4)

# Calcola probabilità con lambda BLEND
prob_2_blend = calc_prob_away_from_lambda(lambda_h_blend, lambda_a_blend)
prob_2_over25_blend = calc_prob_away_over(lambda_h_blend, lambda_a_blend, soglia=2.5)
prob_2_mg24_blend = calc_prob_away_multigol(lambda_h_blend, lambda_a_blend, gmin=2, gmax=4)

print(f"\n" + "="*80)
print("PROBABILITÀ MERCATI AWAY (2)")
print("="*80)

print(f"\n📊 CON LAMBDA APERTURA (spread={spread_apertura}):")
print(f"  P(2):             {prob_2_ap:.4f} ({prob_2_ap*100:.2f}%)")
print(f"  P(2 & Over2.5):   {prob_2_over25_ap:.4f} ({prob_2_over25_ap*100:.2f}%)")
print(f"  P(2 & MG 2-4):    {prob_2_mg24_ap:.4f} ({prob_2_mg24_ap*100:.2f}%)")

print(f"\n📊 CON LAMBDA CORRENTE (spread={spread_corrente}):")
print(f"  P(2):             {prob_2_curr:.4f} ({prob_2_curr*100:.2f}%)")
print(f"  P(2 & Over2.5):   {prob_2_over25_curr:.4f} ({prob_2_over25_curr*100:.2f}%)")
print(f"  P(2 & MG 2-4):    {prob_2_mg24_curr:.4f} ({prob_2_mg24_curr*100:.2f}%)")

print(f"\n📊 CON LAMBDA BLEND ({weight_corrente:.0%} corrente + {weight_apertura:.0%} apertura):")
print(f"  P(2):             {prob_2_blend:.4f} ({prob_2_blend*100:.2f}%)")
print(f"  P(2 & Over2.5):   {prob_2_over25_blend:.4f} ({prob_2_over25_blend*100:.2f}%)")
print(f"  P(2 & MG 2-4):    {prob_2_mg24_blend:.4f} ({prob_2_mg24_blend*100:.2f}%)")

# Analisi variazioni
print(f"\n" + "="*80)
print("ANALISI VARIAZIONI")
print("="*80)

print(f"\n🔍 Variazione APERTURA → CORRENTE:")
print(f"  ΔP(2):           {prob_2_curr - prob_2_ap:+.4f} ({((prob_2_curr/prob_2_ap - 1) * 100):+.2f}%)")
print(f"  ΔP(2 & Over2.5): {prob_2_over25_curr - prob_2_over25_ap:+.4f} ({((prob_2_over25_curr/prob_2_over25_ap - 1) * 100):+.2f}%)")
print(f"  ΔP(2 & MG 2-4):  {prob_2_mg24_curr - prob_2_mg24_ap:+.4f} ({((prob_2_mg24_curr/prob_2_mg24_ap - 1) * 100):+.2f}%)")

print(f"\n🔍 Variazione APERTURA → BLEND:")
print(f"  ΔP(2):           {prob_2_blend - prob_2_ap:+.4f} ({((prob_2_blend/prob_2_ap - 1) * 100):+.2f}%)")
print(f"  ΔP(2 & Over2.5): {prob_2_over25_blend - prob_2_over25_ap:+.4f} ({((prob_2_over25_blend/prob_2_over25_ap - 1) * 100):+.2f}%)")
print(f"  ΔP(2 & MG 2-4):  {prob_2_mg24_blend - prob_2_mg24_ap:+.4f} ({((prob_2_mg24_blend/prob_2_mg24_ap - 1) * 100):+.2f}%)")

# Conclusione
print(f"\n" + "="*80)
print("CONCLUSIONE")
print("="*80)

if prob_2_curr < prob_2_ap:
    print(f"\n❌ PROBLEMA CONFERMATO!")
    print(f"   Quando spread aumenta (0.75 → 1.25), Home diventa più favorita")
    print(f"   Quindi P(2) DIMINUISCE (da {prob_2_ap*100:.2f}% a {prob_2_curr*100:.2f}%)")
    print(f"   E anche P(2 + Over) e P(2 + Multigol) diminuiscono")
    print(f"\n   Questo è CORRETTO matematicamente: spread più alto = Home più forte = Away più debole")
    print(f"\n   Se ti aspetti che P(2) aumenti, c'è una interpretazione errata:")
    print(f"   - Spread > 0 significa Home favorita")
    print(f"   - Spread aumenta → Home ancora più favorita → Away più debole")
else:
    print(f"\n✅ Comportamento coerente")

print(f"\n" + "="*80)
