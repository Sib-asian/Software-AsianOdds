#!/usr/bin/env python3
"""
Test SANITY CHECK - Simula notifica ricevuta dall'utente
=========================================================

Notifica ricevuta:
- Market: Prossimo gol prima del 75'
- Probabilità: 75%
- Quota: 1.80
- EV calcolato: +35.0%

Verifica che i SANITY CHECK limitino correttamente EV e confidence.
"""

# Costanti SANITY CHECK - OPZIONE B (uguali a quelle in live_betting_advisor.py)
MAX_EV_ALLOWED = 15.0
MAX_CONFIDENCE_ALLOWED = 80.0
MAX_PROB_DEVIATION = 0.20  # OPZIONE B: 20% invece di 15%
CONFIDENCE_PENALTY = 0.10  # OPZIONE B: -10% invece di -20%

def calculate_ev(confidence_pct, odds):
    """Calcola EV% da confidence% e quota"""
    prob = confidence_pct / 100.0
    ev_decimal = (prob * odds) - 1.0
    return ev_decimal * 100.0

def calculate_adjusted_confidence(ev_pct, odds):
    """Calcola confidence aggiustata per coerenza con EV cappato"""
    ev_decimal = ev_pct / 100.0
    confidence = ((ev_decimal + 1.0) / odds) * 100.0
    return confidence

def simulate_sanity_check(market, confidence_initial, odds):
    """Simula l'applicazione dei SANITY CHECK"""
    print(f"\n{'='*70}")
    print(f"🎯 TEST: {market}")
    print(f"{'='*70}")
    print(f"\n📊 INPUT:")
    print(f"   Confidence iniziale: {confidence_initial:.1f}%")
    print(f"   Quota: {odds:.2f}")

    # Calcola EV iniziale
    ev_initial = calculate_ev(confidence_initial, odds)
    print(f"   EV iniziale: {ev_initial:+.1f}%")

    # Applica SANITY CHECK (stessa logica di live_betting_advisor.py)
    confidence = confidence_initial
    ev = ev_initial
    ev_was_capped = False

    print(f"\n🛡️ SANITY CHECK:")

    # CHECK 1: Limita EV massimo
    if ev > MAX_EV_ALLOWED:
        print(f"   ⚠️ CHECK 1: EV {ev:.1f}% > {MAX_EV_ALLOWED:.1f}% → LIMITATO a {MAX_EV_ALLOWED:.1f}%")
        ev = MAX_EV_ALLOWED
        ev_was_capped = True
    else:
        print(f"   ✅ CHECK 1: EV {ev:.1f}% ≤ {MAX_EV_ALLOWED:.1f}% → OK")

    # CHECK 2: Limita confidence massima
    if confidence > MAX_CONFIDENCE_ALLOWED:
        print(f"   ⚠️ CHECK 2: Confidence {confidence:.1f}% > {MAX_CONFIDENCE_ALLOWED:.1f}% → LIMITATA a {MAX_CONFIDENCE_ALLOWED:.1f}%")
        confidence = MAX_CONFIDENCE_ALLOWED
        # Ricalcola EV
        ev = calculate_ev(confidence, odds)
        if ev > MAX_EV_ALLOWED:
            ev = MAX_EV_ALLOWED
            ev_was_capped = True
        print(f"   ↳ EV ricalcolato: {ev:.1f}%")
    else:
        print(f"   ✅ CHECK 2: Confidence {confidence:.1f}% ≤ {MAX_CONFIDENCE_ALLOWED:.1f}% → OK")

    # CHECK 3: Verifica deviazione probabilità
    prob_ai = confidence / 100.0
    prob_implied = 1.0 / odds if odds > 1.0 else 0.5
    prob_deviation = abs(prob_ai - prob_implied)

    print(f"   📊 CHECK 3: Deviazione probabilità")
    print(f"      - Prob AI: {prob_ai*100:.1f}%")
    print(f"      - Prob Quote (implicita): {prob_implied*100:.1f}%")
    print(f"      - Deviazione: {prob_deviation*100:.1f}%")

    if prob_deviation > MAX_PROB_DEVIATION:
        print(f"   ⚠️ CHECK 3: Deviazione {prob_deviation*100:.1f}% > {MAX_PROB_DEVIATION*100:.1f}% → PENALIZZO confidence -{CONFIDENCE_PENALTY*100:.0f}%")
        confidence *= (1 - CONFIDENCE_PENALTY)
        print(f"   ↳ Confidence penalizzata: {confidence:.1f}%")
        # Ricalcola EV
        ev = calculate_ev(confidence, odds)
        if ev > MAX_EV_ALLOWED:
            ev = MAX_EV_ALLOWED
            ev_was_capped = True
        print(f"   ↳ EV ricalcolato: {ev:.1f}%")
    else:
        print(f"   ✅ CHECK 3: Deviazione {prob_deviation*100:.1f}% ≤ {MAX_PROB_DEVIATION*100:.1f}% → OK")

    # CHECK 4: Ricalcola confidence per coerenza matematica se EV è stato cappato
    if ev_was_capped and odds > 1.0:
        confidence_adjusted = calculate_adjusted_confidence(ev, odds)
        if abs(confidence_adjusted - confidence) > 1.0:  # Solo se differenza significativa
            print(f"   🔧 CHECK 4: COERENZA - Confidence aggiustata per far tornare i conti")
            print(f"      - Confidence prima: {confidence:.1f}%")
            print(f"      - Confidence dopo: {confidence_adjusted:.1f}%")
            print(f"      - Motivo: EV cappato a {ev:.1f}% → confidence deve essere ({ev:.1f}/100 + 1) / {odds:.2f} × 100")
            confidence = confidence_adjusted
            # Verifica finale
            ev_verify = calculate_ev(confidence, odds)
            print(f"      - ✅ Verifica: ({confidence/100:.4f} × {odds:.2f}) - 1 = {ev_verify:+.1f}%")
        else:
            print(f"   ✅ CHECK 4: COERENZA - Nessun aggiustamento necessario (differenza < 1%)")
    else:
        print(f"   ✅ CHECK 4: COERENZA - Nessun aggiustamento necessario (EV non cappato)")

    # OUTPUT FINALE
    print(f"\n📤 OUTPUT FINALE:")
    print(f"   Confidence finale: {confidence:.1f}%")
    print(f"   EV finale: {ev:+.1f}%")
    print(f"   Quota: {odds:.2f}")

    # Verifica se passerebbe il filtro min_confidence (75%)
    MIN_CONFIDENCE = 75.0
    if confidence >= MIN_CONFIDENCE:
        print(f"\n✅ RISULTATO: Notifica INVIATA (confidence {confidence:.1f}% ≥ {MIN_CONFIDENCE:.1f}%)")
    else:
        print(f"\n❌ RISULTATO: Notifica BLOCCATA (confidence {confidence:.1f}% < {MIN_CONFIDENCE:.1f}%)")

    return confidence, ev


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 TEST SANITY CHECK - Caso Reale Notifica Ricevuta")
    print("="*70)

    # CASO 1: Notifica ricevuta dall'utente
    print("\n" + "🔴 CASO 1: Notifica che hai ricevuto".upper())
    c1, ev1 = simulate_sanity_check(
        market="next_goal_before_75",
        confidence_initial=75.0,
        odds=1.80
    )

    # CASO 2: Scenario con confidence alta
    print("\n" + "🔴 CASO 2: Confidence molto alta (95%)".upper())
    c2, ev2 = simulate_sanity_check(
        market="next_goal_before_75",
        confidence_initial=95.0,
        odds=1.80
    )

    # CASO 3: Scenario con EV altissimo ma confidence normale
    print("\n" + "🔴 CASO 3: EV altissimo (50%) con confidence 70%".upper())
    c3, ev3 = simulate_sanity_check(
        market="over_2_5",
        confidence_initial=70.0,
        odds=2.50  # 70% * 2.50 = 175% - 100% = +75% EV!
    )

    # CASO 4: Scenario con deviazione eccessiva
    print("\n" + "🔴 CASO 4: Deviazione eccessiva (AI 85% vs Quote 50%)".upper())
    c4, ev4 = simulate_sanity_check(
        market="home_win",
        confidence_initial=85.0,
        odds=2.00  # Prob implicita = 50%, AI = 85% → deviazione 35%!
    )

    # RIEPILOGO
    print("\n" + "="*70)
    print("📊 RIEPILOGO TEST")
    print("="*70)
    print(f"\nCASO 1 (Notifica ricevuta): Confidence {c1:.1f}%, EV {ev1:+.1f}%")
    print(f"CASO 2 (Confidence alta):   Confidence {c2:.1f}%, EV {ev2:+.1f}%")
    print(f"CASO 3 (EV altissimo):      Confidence {c3:.1f}%, EV {ev3:+.1f}%")
    print(f"CASO 4 (Deviazione alta):   Confidence {c4:.1f}%, EV {ev4:+.1f}%")

    print("\n" + "="*70)
    print("✅ TEST COMPLETATO")
    print("="*70 + "\n")
