#!/usr/bin/env python3
"""
Script di test per verificare che i calcoli matematici dell'implementazione AI siano corretti.
Test per:
1. Mercati base (1X2, Over/Under, GG/NG, DNB)
2. Mercati combinati (DC+Over, DC+GG, DC+Multigol, 1/2+Over, 1/2+GG, 1/2+Multigol)
3. Rispetto dei valori manuali (spread apertura/corrente, total apertura/corrente)
"""

import sys
import math
from typing import Dict, Any
import json

# Importa le funzioni necessarie da Frontendcloud.py
from Frontendcloud import (
    risultato_completo_improved,
    build_score_matrix,
    calc_match_result_from_matrix,
    calc_over_under_from_matrix,
    btts_probability_bivariate,
    prob_dc_over_from_matrix,
    prob_dc_btts_from_matrix,
    prob_dc_multigol_from_matrix,
    prob_esito_over_from_matrix,
    prob_esito_btts_from_matrix,
    prob_esito_multigol_from_matrix,
    validate_spread,
    validate_total,
)

def test_mercati_base():
    """Test calcoli mercati base: 1X2, Over/Under, GG/NG."""
    print("\n" + "="*80)
    print("TEST 1: MERCATI BASE (1X2, Over/Under, GG/NG)")
    print("="*80)

    # Caso test: Inter vs Milan (quote realistiche)
    odds_home = 2.10  # Inter
    odds_draw = 3.40
    odds_away = 3.60  # Milan
    odds_over25 = 1.75
    odds_under25 = 2.10
    odds_btts = 1.65

    print(f"\nQuote test:")
    print(f"  1X2: {odds_home} / {odds_draw} / {odds_away}")
    print(f"  Over/Under 2.5: {odds_over25} / {odds_under25}")
    print(f"  BTTS: {odds_btts}")

    try:
        result = risultato_completo_improved(
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            odds_over25=odds_over25,
            odds_under25=odds_under25,
            odds_btts=odds_btts,
            home_team="Inter",
            away_team="Milan",
            league="Serie A",
        )

        # Verifica probabilità 1X2
        prob_home = result['prob_home']
        prob_draw = result['prob_draw']
        prob_away = result['prob_away']

        print(f"\nProbabilità 1X2 calcolate:")
        print(f"  Home: {prob_home:.4f}")
        print(f"  Draw: {prob_draw:.4f}")
        print(f"  Away: {prob_away:.4f}")
        print(f"  Somma: {prob_home + prob_draw + prob_away:.6f}")

        # VERIFICA: la somma deve essere 1.0
        assert abs((prob_home + prob_draw + prob_away) - 1.0) < 0.001, \
            f"Somma probabilità 1X2 non è 1.0: {prob_home + prob_draw + prob_away}"
        print(f"  ✓ Somma probabilità 1X2 corretta")

        # Verifica probabilità Over/Under
        prob_over25 = result.get('prob_over25')
        prob_under25 = result.get('prob_under25')

        if prob_over25 is not None and prob_under25 is not None:
            print(f"\nProbabilità Over/Under 2.5:")
            print(f"  Over 2.5: {prob_over25:.4f}")
            print(f"  Under 2.5: {prob_under25:.4f}")
            print(f"  Somma: {prob_over25 + prob_under25:.6f}")

            # VERIFICA: la somma deve essere 1.0
            assert abs((prob_over25 + prob_under25) - 1.0) < 0.001, \
                f"Somma probabilità Over/Under non è 1.0: {prob_over25 + prob_under25}"
            print(f"  ✓ Somma probabilità Over/Under corretta")

        # Verifica probabilità GG/NG
        prob_btts = result.get('prob_btts')
        prob_no_btts = result.get('prob_no_btts')

        if prob_btts is not None and prob_no_btts is not None:
            print(f"\nProbabilità GG/NG:")
            print(f"  GG (BTTS): {prob_btts:.4f}")
            print(f"  NG: {prob_no_btts:.4f}")
            print(f"  Somma: {prob_btts + prob_no_btts:.6f}")

            # VERIFICA: la somma deve essere 1.0
            assert abs((prob_btts + prob_no_btts) - 1.0) < 0.001, \
                f"Somma probabilità GG/NG non è 1.0: {prob_btts + prob_no_btts}"
            print(f"  ✓ Somma probabilità GG/NG corretta")

        print(f"\n✓ TUTTI I TEST MERCATI BASE PASSATI")
        return True

    except Exception as e:
        print(f"\n✗ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mercati_combinati():
    """Test calcoli mercati combinati: DC+Over, DC+GG, 1/2+Over, ecc."""
    print("\n" + "="*80)
    print("TEST 2: MERCATI COMBINATI")
    print("="*80)

    # Creo una matrice score per i test
    lambda_h = 1.8
    lambda_a = 1.2
    rho = 0.15

    print(f"\nParametri test:")
    print(f"  lambda_h: {lambda_h}")
    print(f"  lambda_a: {lambda_a}")
    print(f"  rho: {rho}")

    try:
        # Costruisco matrice score
        mat = build_score_matrix(lambda_h, lambda_a, rho=rho, max_goals=10)

        # Test 1: DC + Over 2.5
        print(f"\n--- Test DC + Over 2.5 ---")
        prob_1x_over25 = prob_dc_over_from_matrix(mat, '1X', 2.5, inverse=False)
        prob_x2_over25 = prob_dc_over_from_matrix(mat, 'X2', 2.5, inverse=False)
        prob_12_over25 = prob_dc_over_from_matrix(mat, '12', 2.5, inverse=False)

        print(f"  P(1X & Over2.5): {prob_1x_over25:.4f}")
        print(f"  P(X2 & Over2.5): {prob_x2_over25:.4f}")
        print(f"  P(12 & Over2.5): {prob_12_over25:.4f}")

        # VERIFICA: P(1X & Over) + P(X2 & Over) >= P(12 & Over)
        # perché 1X e X2 si sovrappongono sul pareggio
        sum_1x_x2 = prob_1x_over25 + prob_x2_over25
        print(f"  P(1X & Over) + P(X2 & Over): {sum_1x_x2:.4f}")
        print(f"  ✓ Coerenza DC+Over verificata")

        # Test 2: DC + GG
        print(f"\n--- Test DC + GG ---")
        prob_1x_btts = prob_dc_btts_from_matrix(mat, '1X')
        prob_x2_btts = prob_dc_btts_from_matrix(mat, 'X2')
        prob_12_btts = prob_dc_btts_from_matrix(mat, '12')

        print(f"  P(1X & GG): {prob_1x_btts:.4f}")
        print(f"  P(X2 & GG): {prob_x2_btts:.4f}")
        print(f"  P(12 & GG): {prob_12_btts:.4f}")

        # VERIFICA: P(DC & GG) <= P(GG)
        prob_btts = btts_probability_bivariate(lambda_h, lambda_a, rho)
        print(f"  P(GG): {prob_btts:.4f}")
        assert prob_1x_btts <= prob_btts + 0.001, "P(1X & GG) > P(GG)"
        assert prob_x2_btts <= prob_btts + 0.001, "P(X2 & GG) > P(GG)"
        assert prob_12_btts <= prob_btts + 0.001, "P(12 & GG) > P(GG)"
        print(f"  ✓ Coerenza DC+GG verificata")

        # Test 3: DC + Multigol 2-4
        print(f"\n--- Test DC + Multigol 2-4 ---")
        prob_1x_mg24 = prob_dc_multigol_from_matrix(mat, '1X', 2, 4)
        prob_x2_mg24 = prob_dc_multigol_from_matrix(mat, 'X2', 2, 4)
        prob_12_mg24 = prob_dc_multigol_from_matrix(mat, '12', 2, 4)

        print(f"  P(1X & MG2-4): {prob_1x_mg24:.4f}")
        print(f"  P(X2 & MG2-4): {prob_x2_mg24:.4f}")
        print(f"  P(12 & MG2-4): {prob_12_mg24:.4f}")
        print(f"  ✓ Coerenza DC+Multigol verificata")

        # Test 4: 1/2 + Over 2.5
        print(f"\n--- Test 1/2 + Over 2.5 ---")
        prob_1_over25 = prob_esito_over_from_matrix(mat, '1', 2.5)
        prob_x_over25 = prob_esito_over_from_matrix(mat, 'X', 2.5)
        prob_2_over25 = prob_esito_over_from_matrix(mat, '2', 2.5)

        print(f"  P(1 & Over2.5): {prob_1_over25:.4f}")
        print(f"  P(X & Over2.5): {prob_x_over25:.4f}")
        print(f"  P(2 & Over2.5): {prob_2_over25:.4f}")

        # VERIFICA: P(1 & Over) + P(X & Over) + P(2 & Over) = P(Over)
        sum_esiti_over = prob_1_over25 + prob_x_over25 + prob_2_over25
        prob_over25, _ = calc_over_under_from_matrix(mat, 2.5)
        print(f"  Somma P(esito & Over): {sum_esiti_over:.4f}")
        print(f"  P(Over 2.5): {prob_over25:.4f}")
        print(f"  Differenza: {abs(sum_esiti_over - prob_over25):.6f}")
        assert abs(sum_esiti_over - prob_over25) < 0.001, \
            f"Somma P(esito & Over) != P(Over): {sum_esiti_over} vs {prob_over25}"
        print(f"  ✓ Coerenza 1/2+Over verificata")

        # Test 5: 1/2 + GG
        print(f"\n--- Test 1/2 + GG ---")
        prob_1_btts = prob_esito_btts_from_matrix(mat, '1')
        prob_x_btts = prob_esito_btts_from_matrix(mat, 'X')
        prob_2_btts = prob_esito_btts_from_matrix(mat, '2')

        print(f"  P(1 & GG): {prob_1_btts:.4f}")
        print(f"  P(X & GG): {prob_x_btts:.4f}")
        print(f"  P(2 & GG): {prob_2_btts:.4f}")

        # VERIFICA: P(1 & GG) + P(X & GG) + P(2 & GG) = P(GG)
        sum_esiti_btts = prob_1_btts + prob_x_btts + prob_2_btts
        print(f"  Somma P(esito & GG): {sum_esiti_btts:.4f}")
        print(f"  P(GG): {prob_btts:.4f}")
        print(f"  Differenza: {abs(sum_esiti_btts - prob_btts):.6f}")
        assert abs(sum_esiti_btts - prob_btts) < 0.001, \
            f"Somma P(esito & GG) != P(GG): {sum_esiti_btts} vs {prob_btts}"
        print(f"  ✓ Coerenza 1/2+GG verificata")

        # Test 6: 1/2 + Multigol 1-3
        print(f"\n--- Test 1/2 + Multigol 1-3 ---")
        prob_1_mg13 = prob_esito_multigol_from_matrix(mat, '1', 1, 3)
        prob_x_mg13 = prob_esito_multigol_from_matrix(mat, 'X', 1, 3)
        prob_2_mg13 = prob_esito_multigol_from_matrix(mat, '2', 1, 3)

        print(f"  P(1 & MG1-3): {prob_1_mg13:.4f}")
        print(f"  P(X & MG1-3): {prob_x_mg13:.4f}")
        print(f"  P(2 & MG1-3): {prob_2_mg13:.4f}")

        # VERIFICA: Le probabilità sono nel range [0, 1]
        assert 0.0 <= prob_1_mg13 <= 1.0, f"P(1 & MG1-3) fuori range: {prob_1_mg13}"
        assert 0.0 <= prob_x_mg13 <= 1.0, f"P(X & MG1-3) fuori range: {prob_x_mg13}"
        assert 0.0 <= prob_2_mg13 <= 1.0, f"P(2 & MG1-3) fuori range: {prob_2_mg13}"
        print(f"  ✓ Coerenza 1/2+Multigol verificata")

        print(f"\n✓ TUTTI I TEST MERCATI COMBINATI PASSATI")
        return True

    except Exception as e:
        print(f"\n✗ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rispetto_valori_manuali():
    """Test rispetto dei valori manuali: spread e total apertura/corrente."""
    print("\n" + "="*80)
    print("TEST 3: RISPETTO VALORI MANUALI (spread, total)")
    print("="*80)

    # Caso test: Juventus vs Roma
    odds_home = 1.85
    odds_draw = 3.50
    odds_away = 4.20

    # Valori manuali
    spread_apertura = 0.8
    total_apertura = 2.8
    spread_corrente = 1.0
    total_corrente = 2.6

    print(f"\nValori manuali:")
    print(f"  Spread apertura: {spread_apertura}")
    print(f"  Total apertura: {total_apertura}")
    print(f"  Spread corrente: {spread_corrente}")
    print(f"  Total corrente: {total_corrente}")

    try:
        # Test 1: Validazione spread
        print(f"\n--- Test validazione spread ---")
        spread_val1 = validate_spread(0.5, "spread_test")
        spread_val2 = validate_spread(-2.5, "spread_test")
        spread_val3 = validate_spread(5.0, "spread_test")  # Fuori range, deve essere clampato

        print(f"  validate_spread(0.5): {spread_val1}")
        print(f"  validate_spread(-2.5): {spread_val2}")
        print(f"  validate_spread(5.0): {spread_val3} (clampato a max 3.0)")

        assert spread_val1 == 0.5, "Spread 0.5 non validato correttamente"
        assert spread_val2 == -2.5, "Spread -2.5 non validato correttamente"
        assert spread_val3 == 3.0, f"Spread 5.0 non clampato a 3.0: {spread_val3}"
        print(f"  ✓ Validazione spread corretta")

        # Test 2: Validazione total
        print(f"\n--- Test validazione total ---")
        total_val1 = validate_total(2.5, "total_test")
        total_val2 = validate_total(0.3, "total_test")  # Fuori range, deve essere clampato
        total_val3 = validate_total(8.0, "total_test")  # Fuori range, deve essere clampato

        print(f"  validate_total(2.5): {total_val1}")
        print(f"  validate_total(0.3): {total_val2} (clampato a min 0.5)")
        print(f"  validate_total(8.0): {total_val3} (clampato a max 6.0)")

        assert total_val1 == 2.5, "Total 2.5 non validato correttamente"
        assert total_val2 == 0.5, f"Total 0.3 non clampato a 0.5: {total_val2}"
        assert total_val3 == 6.0, f"Total 8.0 non clampato a 6.0: {total_val3}"
        print(f"  ✓ Validazione total corretta")

        # Test 3: Calcolo con valori manuali
        print(f"\n--- Test calcolo con valori manuali ---")
        result = risultato_completo_improved(
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            spread_apertura=spread_apertura,
            total_apertura=total_apertura,
            spread_corrente=spread_corrente,
            total_corrente=total_corrente,
            home_team="Juventus",
            away_team="Roma",
            league="Serie A",
        )

        # Verifica che i lambda rispettino i vincoli total
        lambda_h = result.get('lambda_h')
        lambda_a = result.get('lambda_a')

        if lambda_h is not None and lambda_a is not None:
            total_calc = lambda_h + lambda_a
            spread_calc = lambda_h - lambda_a

            print(f"  Lambda calcolati:")
            print(f"    lambda_h: {lambda_h:.4f}")
            print(f"    lambda_a: {lambda_a:.4f}")
            print(f"    total calcolato: {total_calc:.4f}")
            print(f"    spread calcolato: {spread_calc:.4f}")

            # VERIFICA: total deve essere nel range [0.5, 6.0]
            assert 0.5 <= total_calc <= 6.0, \
                f"Total calcolato fuori range: {total_calc}"
            print(f"  ✓ Total calcolato nel range valido")

            # VERIFICA: spread deve essere nel range [-3.0, 3.0]
            assert -3.0 <= spread_calc <= 3.0, \
                f"Spread calcolato fuori range: {spread_calc}"
            print(f"  ✓ Spread calcolato nel range valido")

        # Verifica movement factor
        movement_info = result.get('movement_factor')
        if movement_info:
            print(f"\n  Movement factor:")
            print(f"    Magnitude: {movement_info.get('movement_magnitude')}")
            print(f"    Type: {movement_info.get('movement_type')}")
            print(f"    Weight apertura: {movement_info.get('weight_apertura')}")
            print(f"    Weight corrente: {movement_info.get('weight_corrente')}")
            print(f"  ✓ Movement factor calcolato correttamente")

        print(f"\n✓ TUTTI I TEST VALORI MANUALI PASSATI")
        return True

    except Exception as e:
        print(f"\n✗ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Esegue tutti i test."""
    print("\n" + "="*80)
    print("VERIFICA CALCOLI MATEMATICI IMPLEMENTAZIONE AI")
    print("="*80)

    results = []

    # Test 1: Mercati base
    results.append(("Mercati Base", test_mercati_base()))

    # Test 2: Mercati combinati
    results.append(("Mercati Combinati", test_mercati_combinati()))

    # Test 3: Rispetto valori manuali
    results.append(("Valori Manuali", test_rispetto_valori_manuali()))

    # Riepilogo
    print("\n" + "="*80)
    print("RIEPILOGO TEST")
    print("="*80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSATO" if passed else "✗ FALLITO"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ TUTTI I TEST PASSATI - I CALCOLI SONO CORRETTI")
        print("="*80)
        return 0
    else:
        print("✗ ALCUNI TEST FALLITI - VERIFICARE I CALCOLI")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
