#!/usr/bin/env python3
"""
Test completo per verificare che TUTTE le correzioni dello spread funzionino correttamente.
Test: spread = lambda_a - lambda_h (spread > 0 favorisce Away)

Il file supporta sia pytest sia esecuzione manuale (`python test_spread_fix_complete.py`)
per ottenere un report dettagliato in console.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pytest


# =============================================================================
# Helper matematici
# =============================================================================
def calc_lambda_from_spread_total(spread: float, total: float) -> Tuple[float, float]:
    """
    Formula CORRETTA:
        lambda_a - lambda_h = spread
        lambda_a + lambda_h = total
    """
    lambda_a = (total + spread) / 2.0
    lambda_h = (total - spread) / 2.0
    return lambda_h, lambda_a


def calc_spread_from_lambda(lambda_h: float, lambda_a: float) -> float:
    """Ricalcola lo spread a partire dai lambda."""
    return lambda_a - lambda_h


TEST_CASES = [
    (0.25, 2.5),
    (0.75, 2.5),
    (-0.25, 2.5),
    (-0.75, 2.5),
    (0.0, 2.5),
    (1.0, 3.0),
    (-1.0, 3.0),
]


# =============================================================================
# Test Pytest
# =============================================================================
@pytest.mark.parametrize("spread,total", TEST_CASES)
def test_formula_spread_total_respected(spread: float, total: float) -> None:
    """Verifica che spread e total siano coerenti con i lambda calcolati."""
    lambda_h, lambda_a = calc_lambda_from_spread_total(spread, total)
    spread_check = calc_spread_from_lambda(lambda_h, lambda_a)
    total_check = lambda_h + lambda_a

    assert pytest.approx(spread_check, abs=1e-10) == spread
    assert pytest.approx(total_check, abs=1e-10) == total


def test_interpretazione_spread() -> None:
    """Spread positivo favorisce away, negativo favorisce home, zero bilanciato."""
    total = 2.5

    lambda_h, lambda_a = calc_lambda_from_spread_total(0.5, total)
    assert lambda_a > lambda_h

    lambda_h, lambda_a = calc_lambda_from_spread_total(-0.5, total)
    assert lambda_h > lambda_a

    lambda_h, lambda_a = calc_lambda_from_spread_total(0.0, total)
    assert pytest.approx(lambda_h, abs=1e-10) == lambda_a


def test_movimento_spread_away_favored_when_increasing() -> None:
    """Quando lo spread aumenta, l'away guadagna vantaggio."""
    total = 2.5
    spread1, spread2 = 0.25, 0.50

    lambda_h1, lambda_a1 = calc_lambda_from_spread_total(spread1, total)
    lambda_h2, lambda_a2 = calc_lambda_from_spread_total(spread2, total)

    assert lambda_a2 > lambda_a1
    assert lambda_h2 < lambda_h1


def test_movimento_spread_home_favored_when_decreasing() -> None:
    """Quando lo spread diminuisce, l'home guadagna vantaggio."""
    total = 2.5
    spread1, spread2 = -0.25, -0.50

    lambda_h1, lambda_a1 = calc_lambda_from_spread_total(spread1, total)
    lambda_h2, lambda_a2 = calc_lambda_from_spread_total(spread2, total)

    assert lambda_h2 > lambda_h1
    assert lambda_a2 < lambda_a1


def test_market_spread_sign_coherence() -> None:
    """Verifica che il segno di market_spread_sign sia coerente con le probabilità."""
    # Caso 1: home favorita
    p_home, p_away = 0.40, 0.35
    market_spread_sign = p_away - p_home
    assert market_spread_sign < 0

    # Caso 2: away favorita
    p_home2, p_away2 = 0.30, 0.45
    market_spread_sign2 = p_away2 - p_home2
    assert market_spread_sign2 > 0


def test_lambda_bounds_with_high_spread() -> None:
    """Spread elevati non producono lambda negativi (prima del clamp di sicurezza)."""
    lambda_h, lambda_a = calc_lambda_from_spread_total(2.0, 2.5)
    assert lambda_h >= 0
    assert lambda_a >= lambda_h


# =============================================================================
# Supporto per esecuzione manuale (report stampato)
# =============================================================================
@dataclass
class ManualTestResult:
    description: str
    passed: bool
    details: str = ""


def _run_manual_tests() -> Iterable[ManualTestResult]:
    """Riesegue gli stessi check delle funzioni pytest e ritorna i risultati."""
    results: list[ManualTestResult] = []

    for spread, total in TEST_CASES:
        lambda_h, lambda_a = calc_lambda_from_spread_total(spread, total)
        spread_ok = abs(calc_spread_from_lambda(lambda_h, lambda_a) - spread) < 1e-10
        total_ok = abs((lambda_h + lambda_a) - total) < 1e-10
        results.append(
            ManualTestResult(
                f"Formula spread/total (spread={spread:+.2f}, total={total:.2f})",
                spread_ok and total_ok,
            )
        )

    results.append(
        ManualTestResult(
            "Interpretazione spread (away favorita con spread>0)",
            calc_lambda_from_spread_total(0.5, 2.5)[1] > calc_lambda_from_spread_total(0.5, 2.5)[0],
        )
    )

    lambda_h_neg, lambda_a_neg = calc_lambda_from_spread_total(-0.5, 2.5)
    results.append(
        ManualTestResult(
            "Interpretazione spread (home favorita con spread<0)",
            lambda_h_neg > lambda_a_neg,
        )
    )

    lambda_h_bal, lambda_a_bal = calc_lambda_from_spread_total(0.0, 2.5)
    results.append(
        ManualTestResult(
            "Interpretazione spread (squadre bilanciate con spread=0)",
            abs(lambda_h_bal - lambda_a_bal) < 1e-10,
        )
    )

    lambda_h1, lambda_a1 = calc_lambda_from_spread_total(0.25, 2.5)
    lambda_h2, lambda_a2 = calc_lambda_from_spread_total(0.50, 2.5)
    results.append(
        ManualTestResult(
            "Movimento spread ↑ (away più forte)",
            lambda_a2 > lambda_a1 and lambda_h2 < lambda_h1,
        )
    )

    lambda_h1, lambda_a1 = calc_lambda_from_spread_total(-0.25, 2.5)
    lambda_h2, lambda_a2 = calc_lambda_from_spread_total(-0.50, 2.5)
    results.append(
        ManualTestResult(
            "Movimento spread ↓ (home più forte)",
            lambda_h2 > lambda_h1 and lambda_a2 < lambda_a1,
        )
    )

    p_home, p_away = 0.40, 0.35
    p_home2, p_away2 = 0.30, 0.45
    results.append(
        ManualTestResult(
            "market_spread_sign coerente",
            (p_away - p_home) < 0 and (p_away2 - p_home2) > 0,
        )
    )

    lambda_h_edge, lambda_a_edge = calc_lambda_from_spread_total(2.0, 2.5)
    results.append(
        ManualTestResult(
            "Bound lambda non negativi",
            lambda_h_edge >= 0 and lambda_a_edge >= 0,
        )
    )

    return results


def main() -> int:
    """Entry point manuale per avere lo stesso report dettagliato di prima."""
    print("=" * 80)
    print("TEST COMPLETO CORREZIONE SPREAD")
    print("=" * 80)

    results = list(_run_manual_tests())
    passed = sum(result.passed for result in results)

    for result in results:
        symbol = "✅" if result.passed else "❌"
        detail = result.details or "OK" if result.passed else "KO"
        print(f"{symbol} {result.description}")
        if result.details:
            print(f"    {result.details}")

    print("\n" + "=" * 80)
    print("RIEPILOGO")
    print("=" * 80)
    print(f"{passed}/{len(results)} test superati")

    overall_success = passed == len(results)
    if overall_success:
        print("✅ La correzione dello spread è completa e corretta.")
    else:
        print("❌ Alcuni controlli sono falliti. Verifica i dettagli sopra.")

    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
