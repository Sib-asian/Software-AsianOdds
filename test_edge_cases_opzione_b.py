#!/usr/bin/env python3
"""
Test Edge Cases OPZIONE B
==========================

Testa scenari limite per verificare che non ci siano bug.
"""

# Costanti OPZIONE B
MAX_EV_ALLOWED = 15.0
MAX_CONFIDENCE_ALLOWED = 80.0
MAX_PROB_DEVIATION = 0.20
CONFIDENCE_PENALTY = 0.10

def calculate_ev(confidence_pct, odds):
    prob = confidence_pct / 100.0
    return (prob * odds - 1) * 100.0

def test_edge_case(name, confidence_initial, odds):
    """Testa un edge case"""
    print(f"\n{'='*70}")
    print(f"🧪 {name}")
    print(f"{'='*70}")
    print(f"Input: Confidence {confidence_initial:.1f}%, Quota {odds:.2f}")

    # Simula SANITY CHECK
    confidence = min(confidence_initial, MAX_CONFIDENCE_ALLOWED)
    ev = calculate_ev(confidence, odds)
    ev = min(ev, MAX_EV_ALLOWED)

    prob_ai = confidence / 100.0
    prob_implied = 1.0 / odds if odds > 1.0 else 0.5
    prob_deviation = abs(prob_ai - prob_implied)

    if prob_deviation > MAX_PROB_DEVIATION:
        confidence *= (1 - CONFIDENCE_PENALTY)
        ev = calculate_ev(confidence, odds)
        ev = min(ev, MAX_EV_ALLOWED)

    print(f"Output: Confidence {confidence:.1f}%, EV {ev:+.1f}%")

    # Verifica edge cases
    issues = []

    # Check 1: Confidence negativa o > 100%
    if confidence < 0 or confidence > 100:
        issues.append(f"⚠️ Confidence fuori range: {confidence:.1f}%")

    # Check 2: EV estremo
    if ev > MAX_EV_ALLOWED * 1.1:  # Tolleranza 10%
        issues.append(f"⚠️ EV sopra limite: {ev:.1f}% > {MAX_EV_ALLOWED}%")

    # Check 3: Divisione per zero
    if odds <= 0:
        issues.append(f"⚠️ Quota invalida: {odds}")

    # Check 4: Confidence/EV NaN o Inf
    if str(confidence) in ['nan', 'inf', '-inf'] or str(ev) in ['nan', 'inf', '-inf']:
        issues.append(f"⚠️ Valore NaN/Inf rilevato")

    if issues:
        print("❌ PROBLEMI RILEVATI:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ OK - Nessun problema")
        return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 TEST EDGE CASES - OPZIONE B")
    print("="*70)

    results = []

    # Edge Case 1: Quota molto bassa
    results.append(test_edge_case(
        "Quota molto bassa (1.01)",
        confidence_initial=75.0,
        odds=1.01
    ))

    # Edge Case 2: Quota molto alta
    results.append(test_edge_case(
        "Quota molto alta (50.00)",
        confidence_initial=75.0,
        odds=50.00
    ))

    # Edge Case 3: Confidence esattamente al limite
    results.append(test_edge_case(
        "Confidence esattamente 80%",
        confidence_initial=80.0,
        odds=1.80
    ))

    # Edge Case 4: Deviazione esattamente al limite
    results.append(test_edge_case(
        "Deviazione esattamente 20%",
        confidence_initial=70.0,  # 70% AI vs 50% quote (odds 2.0) = 20% deviazione
        odds=2.00
    ))

    # Edge Case 5: Confidence molto bassa
    results.append(test_edge_case(
        "Confidence molto bassa (50%)",
        confidence_initial=50.0,
        odds=1.80
    ))

    # Edge Case 6: Confidence altissima con penalizzazione
    results.append(test_edge_case(
        "Confidence 100% con deviazione alta",
        confidence_initial=100.0,
        odds=2.00
    ))

    # Edge Case 7: EV negativissimo
    results.append(test_edge_case(
        "EV molto negativo",
        confidence_initial=20.0,
        odds=1.50
    ))

    print("\n" + "="*70)
    print("📊 RIEPILOGO EDGE CASES")
    print("="*70)

    passed = sum(results)
    total = len(results)

    print(f"\n✅ Passati: {passed}/{total}")

    if passed == total:
        print("\n🎉 TUTTI I TEST EDGE CASES PASSATI!")
    else:
        print(f"\n⚠️ {total - passed} test falliti - verificare codice")

    print("="*70 + "\n")
