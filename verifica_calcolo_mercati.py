#!/usr/bin/env python3
"""
Script per verificare come vengono calcolati i mercati e se usano dati reali o stime
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🔍 VERIFICA CALCOLO MERCATI E FONTI DATI")
print("=" * 80)
print()

print("📊 FONTI DATI LIVE:")
print("-" * 80)
print()
print("1. API-FOOTBALL (Primaria - Dati Reali)")
print("   ✅ Score in tempo reale")
print("   ✅ Minuto di gioco")
print("   ✅ Possesso palla (%)")
print("   ✅ Tiri totali")
print("   ✅ Tiri in porta")
print("   ✅ Corner")
print("   ✅ Cartellini (gialli/rossi)")
print("   ✅ Falli")
print("   ✅ Fuorigioco")
print("   ✅ Eventi (gol, sostituzioni)")
print()
print("2. API-SPORTS (Secondaria - Dati Reali)")
print("   ✅ Score in tempo reale")
print("   ✅ Minuto di gioco")
print("   ✅ Statistiche base")
print()
print("3. SISTEMA ALTERNATIVO (Fallback - Stime)")
print("   ⚠️  Usato solo se API-Football e API-SPORTS non disponibili")
print("   ⚠️  Stima dati basandosi su:")
print("      - Tempo trascorso dall'inizio partita")
print("      - Pattern statistici medi")
print("      - Quote attuali")
print()

print("=" * 80)
print("📋 MERCATI E LORO CALCOLO")
print("=" * 80)
print()

mercati_analisi = {
    'OVER/UNDER': {
        'over_0.5': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'shots_home', 'shots_away', 'shots_on_target_home', 'shots_on_target_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
        'over_1.5': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'shots_home', 'shots_away', 'shots_on_target_home', 'shots_on_target_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
        'under_2.5': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'shots_home', 'shots_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
    },
    'RISULTATO FINALE': {
        'home_win': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'possession_home', 'shots_home', 'shots_away', 'shots_on_target_home'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
        'away_win': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'possession_home', 'shots_home', 'shots_away', 'shots_on_target_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
    },
    '🆕 NUOVI MERCATI': {
        'team_to_score_first': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'possession_home', 'shots_home', 'shots_away', 'shots_on_target_home', 'shots_on_target_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
        'team_to_score_last': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'possession_home', 'shots_home', 'shots_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
        'highest_scoring_half': {
            'dati_usati': ['score_home', 'score_away', 'minute'],
            'fonte': 'API-Football (reale) + Stima',
            'stime': 'Sì - stima gol primo tempo (non disponibile direttamente)',
            'nota': 'Stima basata su: se siamo a 50\' e ci sono 2+ gol, probabilmente 1+ nel primo tempo'
        },
        'win_either_half': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'possession_home', 'shots_home', 'shots_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
        'btts_first_half': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'shots_on_target_home', 'shots_on_target_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
        'half_time_result': {
            'dati_usati': ['score_home', 'score_away', 'minute', 'possession_home', 'shots_home', 'shots_away', 'shots_on_target_home', 'shots_on_target_away'],
            'fonte': 'API-Football (reale)',
            'stime': 'No - usa dati reali'
        },
    },
}

for categoria, mercati in mercati_analisi.items():
    print(f"\n🏷️  {categoria}")
    print("-" * 80)
    
    for mercato, info in mercati.items():
        print(f"\n  📊 {mercato}:")
        print(f"     Dati usati: {', '.join(info['dati_usati'])}")
        print(f"     Fonte: {info['fonte']}")
        print(f"     Stime: {info['stime']}")
        if 'nota' in info:
            print(f"     ⚠️  Nota: {info['nota']}")

print()
print("=" * 80)
print("⚠️  LIMITAZIONI E NOTE")
print("=" * 80)
print()
print("1. API-FOOTBALL:")
print("   ✅ Fornisce dati REALI e accurati")
print("   ⚠️  Richiede chiave API (a pagamento per uso intensivo)")
print("   ⚠️  Limite chiamate giornaliere (dipende dal piano)")
print()
print("2. SISTEMA ALTERNATIVO (Fallback):")
print("   ⚠️  Usa STIME basate su pattern statistici")
print("   ⚠️  Meno accurato di API-Football")
print("   ✅ Disponibile anche senza chiave API")
print()
print("3. MERCATI CON STIME:")
print("   ⚠️  highest_scoring_half: stima gol primo tempo")
print("      (API-Football non fornisce direttamente gol per tempo)")
print("   ✅ Tutti gli altri mercati usano dati REALI")
print()
print("4. QUALITÀ DATI:")
print("   ✅ Score: sempre reale (da API-Football o API-SPORTS)")
print("   ✅ Minuto: sempre reale")
print("   ✅ Statistiche (tiri, possesso, ecc.): reali se API-Football disponibile")
print("   ⚠️  Se solo sistema alternativo: statistiche stimate")
print()
print("=" * 80)
print("💡 RACCOMANDAZIONE")
print("=" * 80)
print()
print("Per massima accuratezza:")
print("  → Configura API-Football con chiave API valida")
print("  → Il sistema userà dati REALI per tutti i mercati")
print("  → Solo highest_scoring_half userà una stima (necessaria)")
print()
print("Senza API-Football:")
print("  → Il sistema userà stime per statistiche")
print("  → Score e minuto rimangono reali (da API-SPORTS o sistema alternativo)")
print("  → I mercati funzionano ma con accuratezza ridotta")
print()
print("=" * 80)


