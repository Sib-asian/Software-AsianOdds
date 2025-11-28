#!/usr/bin/env python3
"""
Test OPZIONE B - Threshold più bilanciati
==========================================

Confronta OPZIONE A (attuale) vs OPZIONE B (proposta)

OPZIONE A (CONSERVATIVA):
- MAX_PROB_DEVIATION = 15%
- Penalizzazione = -20%

OPZIONE B (BILANCIATA):
- MAX_PROB_DEVIATION = 20%
- Penalizzazione = -10%
"""

def test_opzioni(market, confidence_initial, odds):
    """Confronta opzioni A vs B"""
    print(f"\n{'='*70}")
    print(f"🎯 Market: {market}")
    print(f"   Confidence: {confidence_initial:.1f}% | Quota: {odds:.2f}")

    prob_ai = confidence_initial / 100.0
    prob_implied = 1.0 / odds
    prob_deviation = abs(prob_ai - prob_implied)

    ev_initial = (prob_ai * odds - 1) * 100

    print(f"   Prob AI: {prob_ai*100:.1f}% | Prob Quote: {prob_implied*100:.1f}%")
    print(f"   Deviazione: {prob_deviation*100:.1f}% | EV iniziale: {ev_initial:+.1f}%")

    # OPZIONE A (attuale)
    MAX_EV_A = 15.0
    MAX_CONF_A = 80.0
    MAX_DEV_A = 0.15
    PENALTY_A = 0.20

    conf_a = min(confidence_initial, MAX_CONF_A)
    ev_a = (conf_a/100 * odds - 1) * 100
    ev_a = min(ev_a, MAX_EV_A)

    if prob_deviation > MAX_DEV_A:
        conf_a *= (1 - PENALTY_A)
        ev_a = (conf_a/100 * odds - 1) * 100
        ev_a = min(ev_a, MAX_EV_A)
        result_a = "PENALIZZATO"
    else:
        result_a = "OK"

    # OPZIONE B (proposta)
    MAX_EV_B = 15.0
    MAX_CONF_B = 80.0
    MAX_DEV_B = 0.20  # Aumentato da 15% a 20%
    PENALTY_B = 0.10  # Ridotto da 20% a 10%

    conf_b = min(confidence_initial, MAX_CONF_B)
    ev_b = (conf_b/100 * odds - 1) * 100
    ev_b = min(ev_b, MAX_EV_B)

    if prob_deviation > MAX_DEV_B:
        conf_b *= (1 - PENALTY_B)
        ev_b = (conf_b/100 * odds - 1) * 100
        ev_b = min(ev_b, MAX_EV_B)
        result_b = "PENALIZZATO"
    else:
        result_b = "OK"

    MIN_CONF = 75.0

    print(f"\n   📊 OPZIONE A (Conservativa):")
    print(f"      Confidence finale: {conf_a:.1f}% | EV: {ev_a:+.1f}% | {result_a}")
    if conf_a >= MIN_CONF:
        print(f"      ✅ NOTIFICA INVIATA")
    else:
        print(f"      ❌ NOTIFICA BLOCCATA (< {MIN_CONF:.0f}%)")

    print(f"\n   📊 OPZIONE B (Bilanciata):")
    print(f"      Confidence finale: {conf_b:.1f}% | EV: {ev_b:+.1f}% | {result_b}")
    if conf_b >= MIN_CONF:
        print(f"      ✅ NOTIFICA INVIATA")
    else:
        print(f"      ❌ NOTIFICA BLOCCATA (< {MIN_CONF:.0f}%)")

    return conf_a >= MIN_CONF, conf_b >= MIN_CONF


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 CONFRONTO OPZIONE A vs OPZIONE B")
    print("="*70)

    casi = [
        ("next_goal_before_75 (TUA NOTIFICA)", 75.0, 1.80),
        ("over_2_5 (Deviazione media)", 70.0, 2.00),
        ("home_win (Deviazione alta)", 85.0, 2.00),
        ("under_2_5 (Confidence normale)", 78.0, 1.90),
    ]

    results_a = []
    results_b = []

    for market, conf, odds in casi:
        pass_a, pass_b = test_opzioni(market, conf, odds)
        results_a.append(pass_a)
        results_b.append(pass_b)

    print(f"\n{'='*70}")
    print("📊 RIEPILOGO")
    print("="*70)

    count_a = sum(results_a)
    count_b = sum(results_b)

    print(f"\nOPZIONE A: {count_a}/{len(casi)} notifiche inviate")
    print(f"OPZIONE B: {count_b}/{len(casi)} notifiche inviate")

    print(f"\n{'='*70}")
    print("💡 RACCOMANDAZIONE:")
    print("="*70)
    print("""
OPZIONE A (attuale):
  ✅ Più sicura, evita molti falsi positivi
  ❌ Troppo restrittiva, blocca opportunità valide

OPZIONE B (proposta):
  ✅ Più bilanciata, permette value betting legittimo
  ✅ Limita comunque EV a 15% (realistico)
  ⚠️ Potrebbe permettere qualche segnale ottimistico

🎯 CONSIGLIO: Usa OPZIONE B se vuoi ricevere notifiche per
   value betting dove AI trova valore che il mercato sottostima.
   Comunque EV rimane limitato a max 15%.
""")
    print("="*70 + "\n")
