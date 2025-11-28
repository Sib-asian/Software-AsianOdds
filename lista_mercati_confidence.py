#!/usr/bin/env python3
"""
Script per visualizzare tutti i mercati che il sistema può inviare
con le relative confidence minime
"""

import sys
from pathlib import Path

# Aggiungi path per import
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("📊 LISTA MERCATI INVIATI SU TELEGRAM CON CONFIDENCE MINIME")
print("=" * 80)
print()

# Confidence minima generale
print("🎯 CONFIDENCE MINIMA GENERALE: 72.0%")
print("   (Tutti i segnali devono avere almeno questa confidence per essere inviati)")
print()

# Confidence minime per mercato (dal codice)
market_confidence = {
    'over_0.5': 72.0,
    'over_0.5_ht': 75.0,
    'over_1.5': 75.0,
    'over_1.5_ht': 75.0,
    'over_2.5': 78.0,
    'over_3.5': 80.0,
    'under_0.5': 75.0,
    'under_0.5_ht': 78.0,
    'under_1.5': 78.0,
    'under_1.5_ht': 78.0,
    'under_2.5': 80.0,
    'under_3.5': 82.0,
    'exact_score': 85.0,
    'goal_range_': 75.0,
    'dnb_': 78.0,
    'clean_sheet': 82.0,
    'match_winner': 78.0,
    'home_win': 78.0,
    'away_win': 78.0,
    '1x': 75.0,
    'x2': 75.0,
    '12': 75.0,
    'btts_yes': 75.0,
    'btts_no': 75.0,
    'win_to_nil': 80.0,
}

# Raggruppa per categoria
categories = {
    'OVER/UNDER': [
        ('over_0.5', 72.0),
        ('over_0.5_ht', 73.0),  # 🆕 Abbassato da 75%
        ('over_1.5', 73.0),  # 🆕 Abbassato da 75%
        ('over_1.5_ht', 73.0),  # 🆕 Abbassato da 75%
        ('over_2.5', 75.0),  # 🆕 Abbassato da 78%
        ('over_3.5', 76.0),  # 🆕 Abbassato da 80%
        ('under_0.5', 73.0),  # 🆕 Abbassato da 75%
        ('under_0.5_ht', 75.0),  # 🆕 Abbassato da 78%
        ('under_1.5', 75.0),  # 🆕 Abbassato da 78%
        ('under_1.5_ht', 75.0),  # 🆕 Abbassato da 78%
        ('under_2.5', 76.0),  # 🆕 Abbassato da 80%
        ('under_3.5', 78.0),  # 🆕 Abbassato da 82%
    ],
    'RISULTATO FINALE (1X2)': [
        ('home_win', 75.0),  # 🆕 Abbassato da 78%
        ('away_win', 75.0),  # 🆕 Abbassato da 78%
        ('match_winner', 75.0),  # 🆕 Abbassato da 78%
        ('1x', 75.0),
        ('x2', 75.0),
        ('12', 75.0),
    ],
    'DOUBLE CHANCE': [
        ('1x', 75.0),
        ('x2', 75.0),
        ('12', 75.0),
    ],
    'DRAW NO BET': [
        ('dnb_home', 75.0),  # 🆕 Abbassato da 78%
        ('dnb_away', 75.0),  # 🆕 Abbassato da 78%
    ],
    'BTTS (Both Teams To Score)': [
        ('btts_yes', 73.0),  # 🆕 Abbassato da 75%
        ('btts_no', 73.0),  # 🆕 Abbassato da 75%
    ],
    'CLEAN SHEET': [
        ('clean_sheet_home', 78.0),  # 🆕 Abbassato da 82%
        ('clean_sheet_away', 78.0),  # 🆕 Abbassato da 82%
    ],
    'ALTRI MERCATI': [
        ('exact_score', 80.0),  # 🆕 Abbassato da 85%
        ('goal_range_0_1', 73.0),  # 🆕 Abbassato da 75%
        ('goal_range_2_3', 73.0),  # 🆕 Abbassato da 75%
        ('goal_range_4_plus', 73.0),  # 🆕 Abbassato da 75%
        ('win_to_nil', 76.0),  # 🆕 Abbassato da 80%
        ('team_to_score_next', 73.0),  # 🆕 Abbassato da 75%
        ('next_goal', 75.0),  # 🆕 Abbassato da 78%
        ('total_goals_odd', 75.0),  # 🆕 Abbassato da 78%
        ('total_goals_even', 75.0),  # 🆕 Abbassato da 78%
        ('ht_ft', 75.0),  # 🆕 Abbassato da 78%
        ('corner', 73.0),  # 🆕 Abbassato da 75%
    ],
    '🆕 NUOVI MERCATI': [
        ('team_to_score_first_home', 73.0),
        ('team_to_score_first_away', 73.0),
        ('team_to_score_last_home', 73.0),
        ('team_to_score_last_away', 73.0),
        ('highest_scoring_half_1h', 75.0),
        ('highest_scoring_half_2h', 75.0),
        ('win_either_half_home', 73.0),
        ('win_either_half_away', 73.0),
        ('btts_first_half', 73.0),
        ('half_time_result_home', 73.0),
        ('half_time_result_away', 73.0),
    ],
}

print("=" * 80)
print("📋 MERCATI PER CATEGORIA")
print("=" * 80)
print()

for category, markets in categories.items():
    print(f"\n🏷️  {category}")
    print("-" * 80)
    
    for market, conf in markets:
        # Determina se usa confidence generale o specifica
        if conf >= 72.0:
            status = "✅"
        elif conf >= 70.0:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"  {status} {market:25s} → Confidence minima: {conf:.1f}%")
    
    print()

print("=" * 80)
print("📊 RIEPILOGO CONFIDENCE")
print("=" * 80)
print()

# Statistiche
all_confidences = [conf for _, conf in sum(categories.values(), [])]
if all_confidences:
    print(f"  Confidence minima assoluta: {min(all_confidences):.1f}%")
    print(f"  Confidence massima: {max(all_confidences):.1f}%")
    print(f"  Confidence media: {sum(all_confidences)/len(all_confidences):.1f}%")
    print()
    
    # Distribuzione
    high = sum(1 for c in all_confidences if c >= 80)
    medium = sum(1 for c in all_confidences if 75 <= c < 80)
    low = sum(1 for c in all_confidences if c < 75)
    
    print(f"  Distribuzione:")
    print(f"    Alta (≥80%): {high} mercati")
    print(f"    Media (75-79%): {medium} mercati")
    print(f"    Bassa (<75%): {low} mercati")

print()
print("=" * 80)
print("⚠️  NOTE IMPORTANTI")
print("=" * 80)
print()
print("1. CONFIDENCE MINIMA GENERALE: 72.0%")
print("   → Tutti i segnali devono avere almeno 72% per essere inviati")
print()
print("2. CONFIDENCE PER MERCATO:")
print("   → Alcuni mercati richiedono confidence più alta (es. exact_score: 85%)")
print("   → I mercati più rischiosi hanno confidence minima più alta")
print()
print("3. FILTRI ANTI-OVVIETÀ:")
print("   → Il sistema blocca segnali banali anche se hanno confidence alta")
print("   → Esempi: Over 0.5 quando c'è già 1 gol, 1X quando è già 1-0, ecc.")
print()
print("4. MERCATI BLOCCATI AL 90'+:")
print("   → Tutti i mercati su risultato finale sono bloccati al 88'+")
print("   → Mercati su risultato finale bloccati al 85'+ se pareggio")
print()
print("=" * 80)

