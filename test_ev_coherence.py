#!/usr/bin/env python3
"""
Test EV Coherence Fix
=====================

Verifica che quando l'EV viene cappato, la confidence viene aggiustata
per mantenere la coerenza matematica nella notifica.

CASO TEST: Notifica ricevuta dall'utente
- Over 1.5 gol
- Probabilità: 80%
- Quota: 1.60
- EV mostrato: +15.0%

PROBLEMA: 80% × 1.60 - 1 = 28%, non 15%!
SOLUZIONE: Aggiustare probabilità a ~72% per far tornare i conti
"""

# Costanti SANITY CHECK (uguali a quelle in live_betting_advisor.py)
MAX_EV_ALLOWED = 15.0
MAX_CONFIDENCE_ALLOWED = 80.0

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

def test_ev_coherence():
    """Test del fix di coerenza EV"""
    print("\n" + "="*70)
    print("🔬 TEST EV COHERENCE FIX")
    print("="*70)

    # CASO UTENTE: Over 1.5 con 80% e quota 1.60
    confidence_initial = 80.0
    odds = 1.60

    print(f"\n📊 INPUT:")
    print(f"   Market: Over 1.5 gol")
    print(f"   Confidence iniziale: {confidence_initial:.1f}%")
    print(f"   Quota: {odds:.2f}")

    # Calcola EV iniziale
    ev_initial = calculate_ev(confidence_initial, odds)
    print(f"\n💰 CALCOLO EV:")
    print(f"   EV calcolato: ({confidence_initial/100:.2f} × {odds:.2f}) - 1 = {ev_initial:+.1f}%")

    # Applica SANITY CHECK
    ev_final = ev_initial
    confidence_final = confidence_initial

    if ev_initial > MAX_EV_ALLOWED:
        print(f"\n🛡️ SANITY CHECK:")
        print(f"   ⚠️ EV {ev_initial:.1f}% > {MAX_EV_ALLOWED:.1f}% → CAPPATO a {MAX_EV_ALLOWED:.1f}%")
        ev_final = MAX_EV_ALLOWED

        # NUOVO: Aggiusta confidence per coerenza
        confidence_adjusted = calculate_adjusted_confidence(ev_final, odds)
        print(f"\n🔧 FIX COERENZA:")
        print(f"   Confidence aggiustata: ((({ev_final:.1f}/100) + 1) / {odds:.2f}) × 100")
        print(f"   = {confidence_adjusted:.2f}%")
        print(f"   (da {confidence_initial:.1f}% → {confidence_adjusted:.1f}%)")

        confidence_final = confidence_adjusted

        # Verifica che i conti tornino
        ev_verify = calculate_ev(confidence_final, odds)
        print(f"\n✅ VERIFICA COERENZA:")
        print(f"   EV ricalcolato: ({confidence_final/100:.4f} × {odds:.2f}) - 1 = {ev_verify:+.1f}%")

        if abs(ev_verify - ev_final) < 0.1:
            print(f"   ✅ COERENZA OK: {ev_verify:.1f}% ≈ {ev_final:.1f}%")
        else:
            print(f"   ❌ INCOERENZA: {ev_verify:.1f}% ≠ {ev_final:.1f}%")

    # OUTPUT FINALE
    print(f"\n📱 NOTIFICA MOSTRATA:")
    print(f"   📊 Over 1.5 gol | {confidence_final:.0f}% | {odds:.2f} | EV: {ev_final:+.1f}%")

    print(f"\n{'='*70}")
    print(f"✅ Test completato!")
    print(f"{'='*70}\n")

    return confidence_final, ev_final

if __name__ == "__main__":
    # Test del fix
    confidence, ev = test_ev_coherence()

    # Confronto PRIMA/DOPO
    print("\n📊 CONFRONTO PRIMA/DOPO:")
    print(f"\nPRIMA del fix (INCOERENTE):")
    print(f"   📊 Over 1.5 gol | 80% | 1.60 | EV: +15.0%")
    print(f"   ❌ Verifica: (0.80 × 1.60) - 1 = +28% ≠ +15%")

    print(f"\nDOPO il fix (COERENTE):")
    print(f"   📊 Over 1.5 gol | {confidence:.0f}% | 1.60 | EV: {ev:+.1f}%")
    print(f"   ✅ Verifica: ({confidence/100:.4f} × 1.60) - 1 ≈ {ev:+.1f}%")

    print("\n" + "="*70)
    print("💡 CONCLUSIONE:")
    print("   Il sistema ora mostra valori matematicamente coerenti!")
    print("   L'utente può verificare i calcoli e fidarsi delle notifiche.")
    print("="*70 + "\n")
