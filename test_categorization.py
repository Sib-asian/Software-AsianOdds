#!/usr/bin/env python3
"""
Test della funzione di categorizzazione dei mercati.
"""

def categorize_market(market_name: str) -> str:
    """
    Categorizza un mercato in base al nome.

    Categorie:
    - 📊 Multigol Combo: 1X/X2/12 + Multigol, 1/2 + Multigol
    - 🎯 Combo Over/Under: 1X/X2/12 + Over/Under, 1/2 + Over/Under
    - ⚽ Combo GG: 1/2 + GG, 1X/X2 + GG
    - 📈 Over/Under 2.5/3.5: Over/Under semplici
    - ⏰ Over HT: Over HT + FT, Over HT solo
    """
    market_lower = market_name.lower()

    # Multigol Combo (priorità massima per multigol combo)
    if 'multigol' in market_lower and ('&' in market_lower or '+' in market_lower):
        return "📊 Multigol Combo"

    # GG + Over (categoria speciale - deve essere prima di "Combo Over/Under")
    if ('gg' in market_lower or 'btts' in market_lower) and ('&' in market_lower or '+' in market_lower):
        return "⚽ Combo GG"

    # Combo Over/Under (dopo GG)
    if ('over' in market_lower or 'under' in market_lower) and ('&' in market_lower or '+' in market_lower):
        if 'ht' not in market_lower:
            return "🎯 Combo Over/Under"

    # Over HT
    if 'ht' in market_lower and ('over' in market_lower or 'under' in market_lower):
        return "⏰ Over HT"

    # Over/Under 2.5/3.5 semplici
    if ('over 2.5' in market_lower or 'under 2.5' in market_lower or
        'over 3.5' in market_lower or 'under 3.5' in market_lower or
        'over 1.5' in market_lower or 'under 1.5' in market_lower):
        if '&' not in market_lower and '+' not in market_lower:
            return "📈 Over/Under"

    # Multigol semplice (senza combo)
    if 'multigol' in market_lower:
        return "📊 Multigol"

    # Categorie base
    if market_name.startswith("1X (") or market_name.startswith("X2 (") or market_name.startswith("12 ("):
        return "🔄 Double Chance"
    elif market_name.startswith("1 (") or market_name.startswith("X (") or market_name.startswith("2 ("):
        return "🏠 1X2"
    elif 'btts' in market_lower or market_name.startswith("BTTS"):
        return "⚽ BTTS/GG"
    elif 'dnb' in market_lower:
        return "🎲 DNB"
    elif 'risultato esatto' in market_lower or 'correct score' in market_lower:
        return "🎯 Correct Score"
    else:
        return "🔀 Altri"


# Test cases
test_cases = [
    # Multigol Combo
    ("1 & Multigol 1-3", "📊 Multigol Combo"),
    ("2 & Multigol 2-4", "📊 Multigol Combo"),
    ("1X & Multigol 1-3", "📊 Multigol Combo"),
    ("X2 & Multigol 2-5", "📊 Multigol Combo"),
    ("12 & Multigol 3-5", "📊 Multigol Combo"),

    # Combo Over/Under
    ("1X & Over 1.5", "🎯 Combo Over/Under"),
    ("X2 & Over 2.5", "🎯 Combo Over/Under"),
    ("1 & Over 2.5", "🎯 Combo Over/Under"),
    ("2 & Over 1.5", "🎯 Combo Over/Under"),
    ("1X & Under 2.5", "🎯 Combo Over/Under"),
    ("X2 & Under 3.5", "🎯 Combo Over/Under"),

    # Combo GG
    ("1 & BTTS", "⚽ Combo GG"),
    ("2 & BTTS", "⚽ Combo GG"),
    ("1X & GG", "⚽ Combo GG"),
    ("X2 & GG", "⚽ Combo GG"),
    ("GG & Over 2.5", "⚽ Combo GG"),

    # Over/Under semplici
    ("Over 1.5", "📈 Over/Under"),
    ("Under 1.5", "📈 Over/Under"),
    ("Over 2.5", "📈 Over/Under"),
    ("Under 2.5", "📈 Over/Under"),
    ("Over 3.5", "📈 Over/Under"),
    ("Under 3.5", "📈 Over/Under"),

    # Over HT
    ("Over 0.5 HT", "⏰ Over HT"),
    ("Over 1.5 HT", "⏰ Over HT"),
    ("Over 0.5 HT + Over 0.5 FT", "⏰ Over HT"),
    ("Over 0.5 HT + Over 1.5 FT", "⏰ Over HT"),
    ("Over 0.5 HT + Over 2.5 FT", "⏰ Over HT"),
    ("Over 1.5 HT + Over 2.5 FT", "⏰ Over HT"),

    # Categorie base
    ("1 (Casa)", "🏠 1X2"),
    ("X (Pareggio)", "🏠 1X2"),
    ("2 (Trasferta)", "🏠 1X2"),
    ("1X (DC Casa o Pareggio)", "🔄 Double Chance"),
    ("X2 (DC Trasferta o Pareggio)", "🔄 Double Chance"),
    ("12 (DC Casa o Trasferta)", "🔄 Double Chance"),
    ("BTTS Sì (GG)", "⚽ BTTS/GG"),
    ("DNB Casa", "🎲 DNB"),
    ("DNB Trasferta", "🎲 DNB"),
    ("Risultato Esatto 1-0", "🎯 Correct Score"),

    # Multigol semplice
    ("Multigol: 1-2", "📊 Multigol"),
    ("Multigol: 2-3", "📊 Multigol"),
]

print("=" * 80)
print("TEST CATEGORIZZAZIONE MERCATI")
print("=" * 80)
print()

passed = 0
failed = 0

for market_name, expected_category in test_cases:
    result = categorize_market(market_name)
    status = "✅" if result == expected_category else "❌"

    if result == expected_category:
        passed += 1
    else:
        failed += 1

    print(f"{status} '{market_name}'")
    print(f"   Atteso: {expected_category}")
    print(f"   Ottenuto: {result}")
    if result != expected_category:
        print("   ⚠️ FALLITO!")
    print()

print("=" * 80)
print(f"RISULTATO: {passed} test passati, {failed} test falliti")
print(f"Percentuale successo: {(passed / len(test_cases) * 100):.1f}%")
print("=" * 80)

# Test raggruppamento
print("\nTEST RAGGRUPPAMENTO PER CATEGORIA:")
print("=" * 80)

from collections import defaultdict

markets_by_category = defaultdict(list)
for market_name, _ in test_cases:
    category = categorize_market(market_name)
    markets_by_category[category].append(market_name)

for category, markets in sorted(markets_by_category.items()):
    print(f"\n{category} ({len(markets)} mercati):")
    for market in markets:
        print(f"  - {market}")

print("\n" + "=" * 80)
print("Test completato!")
print("=" * 80)
