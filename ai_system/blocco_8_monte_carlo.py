"""
BLOCCO 8: Monte Carlo Simulation Engine

Sistema avanzato per simulazione Monte Carlo che valuta la robustezza
delle previsioni attraverso migliaia di scenari simulati.

Features:
- Simulazione multi-scenario con distribuzione probabilistica
- Value at Risk (VaR) e Conditional VaR (CVaR)
- Stress testing sotto condizioni estreme
- Sensitivity analysis per parametri chiave
- Scenario generation con correlazioni
- Robustness scoring
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MonteCarloResult:
    """Risultato della simulazione Monte Carlo"""
    mean_outcome: float
    median_outcome: float
    std_outcome: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    var_95: float  # Value at Risk al 95%
    cvar_95: float  # Conditional VaR al 95%
    worst_case: float
    best_case: float
    probability_positive_ev: float
    expected_value: float
    robustness_score: float  # 0-100
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    confidence_interval: Tuple[float, float]
    simulation_samples: np.ndarray


@dataclass
class StressTestResult:
    """Risultato dello stress test"""
    scenario_name: str
    mean_impact: float
    worst_impact: float
    probability_failure: float
    resilience_score: float  # 0-100
    recommendations: List[str]


class MonteCarloSimulator:
    """
    Engine per simulazioni Monte Carlo avanzate.

    Simula migliaia di scenari per valutare robustezza delle previsioni
    e quantificare il rischio sotto diverse condizioni.
    """

    def __init__(self, n_simulations: int = 10000, random_state: int = 42):
        """
        Args:
            n_simulations: Numero di simulazioni da eseguire
            random_state: Seed per riproducibilità
        """
        self.n_simulations = n_simulations
        self.random_state = random_state
        np.random.seed(random_state)

    def simulate_match_outcome(
        self,
        lambda_home: float,
        lambda_away: float,
        rho: float = 0.0,
        uncertainty_home: float = 0.1,
        uncertainty_away: float = 0.1
    ) -> MonteCarloResult:
        """
        Simula outcome di un match con uncertainty nei parametri.

        Args:
            lambda_home: Expected goals squadra casa
            lambda_away: Expected goals squadra trasferta
            rho: Correlazione Dixon-Coles
            uncertainty_home: Incertezza su lambda_home (std relativa)
            uncertainty_away: Incertezza su lambda_away (std relativa)

        Returns:
            MonteCarloResult con statistiche della simulazione
        """
        outcomes = []

        for _ in range(self.n_simulations):
            # Campiona lambda con incertezza (assumiamo distribuzione gamma)
            # Parametri gamma per mantenere media e std desiderati
            shape_home = (lambda_home / (uncertainty_home * lambda_home)) ** 2
            scale_home = (uncertainty_home * lambda_home) ** 2 / lambda_home

            shape_away = (lambda_away / (uncertainty_away * lambda_away)) ** 2
            scale_away = (uncertainty_away * lambda_away) ** 2 / lambda_away

            lambda_h_sim = np.random.gamma(shape_home, scale_home)
            lambda_a_sim = np.random.gamma(shape_away, scale_away)

            # Simula goals (Poisson)
            goals_home = np.random.poisson(lambda_h_sim)
            goals_away = np.random.poisson(lambda_a_sim)

            # Outcome: 1 = Home win, 0 = Draw, -1 = Away win
            if goals_home > goals_away:
                outcome = 1.0
            elif goals_home < goals_away:
                outcome = -1.0
            else:
                outcome = 0.0

            outcomes.append(outcome)

        outcomes = np.array(outcomes)

        # Calcoliamo statistiche
        mean_outcome = np.mean(outcomes)
        median_outcome = np.median(outcomes)
        std_outcome = np.std(outcomes)

        # Percentili
        p5 = np.percentile(outcomes, 5)
        p25 = np.percentile(outcomes, 25)
        p75 = np.percentile(outcomes, 75)
        p95 = np.percentile(outcomes, 95)

        # VaR e CVaR (dal punto di vista del betting)
        # Assumiamo stake = 1, odds = 2.0 per esempio
        # VaR: peggiore perdita al 95% di confidenza
        var_95 = np.percentile(outcomes, 5)

        # CVaR: media delle perdite peggiori del 5%
        worst_5_percent = outcomes[outcomes <= np.percentile(outcomes, 5)]
        cvar_95 = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else var_95

        worst_case = np.min(outcomes)
        best_case = np.max(outcomes)

        # Probabilità di outcome positivo
        prob_positive = np.sum(outcomes > 0) / len(outcomes)

        # Expected value
        expected_value = mean_outcome

        # Robustness score basato su coerenza dei risultati
        # Alta robustezza = bassa varianza e risultati consistenti
        cv = std_outcome / abs(mean_outcome) if mean_outcome != 0 else float('inf')
        robustness_score = max(0, min(100, 100 * (1 - cv)))

        # Risk level
        if cv < 0.2:
            risk_level = "LOW"
        elif cv < 0.5:
            risk_level = "MEDIUM"
        elif cv < 1.0:
            risk_level = "HIGH"
        else:
            risk_level = "EXTREME"

        # Confidence interval
        ci = (np.percentile(outcomes, 2.5), np.percentile(outcomes, 97.5))

        return MonteCarloResult(
            mean_outcome=mean_outcome,
            median_outcome=median_outcome,
            std_outcome=std_outcome,
            percentile_5=p5,
            percentile_25=p25,
            percentile_75=p75,
            percentile_95=p95,
            var_95=var_95,
            cvar_95=cvar_95,
            worst_case=worst_case,
            best_case=best_case,
            probability_positive_ev=prob_positive,
            expected_value=expected_value,
            robustness_score=robustness_score,
            risk_level=risk_level,
            confidence_interval=ci,
            simulation_samples=outcomes
        )

    def simulate_betting_roi(
        self,
        probability: float,
        odds: float,
        stake: float = 1.0,
        probability_uncertainty: float = 0.05,
        n_bets: int = 100
    ) -> MonteCarloResult:
        """
        Simula ROI di una strategia di betting nel tempo.

        Args:
            probability: Probabilità stimata di vincita
            odds: Quote offerte
            stake: Stake per bet
            probability_uncertainty: Incertezza sulla probabilità
            n_bets: Numero di bet da simulare

        Returns:
            MonteCarloResult con distribuzione ROI
        """
        roi_results = []

        for _ in range(self.n_simulations):
            # Campiona la "vera" probabilità con incertezza
            true_prob = np.clip(
                np.random.normal(probability, probability_uncertainty),
                0.01, 0.99
            )

            # Simula n_bets
            wins = np.random.binomial(n_bets, true_prob)
            losses = n_bets - wins

            # Calcola ROI
            profit = wins * (odds - 1) * stake - losses * stake
            total_staked = n_bets * stake
            roi = (profit / total_staked) * 100 if total_staked > 0 else 0

            roi_results.append(roi)

        roi_results = np.array(roi_results)

        # Statistiche
        mean_roi = np.mean(roi_results)
        median_roi = np.median(roi_results)
        std_roi = np.std(roi_results)

        p5 = np.percentile(roi_results, 5)
        p25 = np.percentile(roi_results, 25)
        p75 = np.percentile(roi_results, 75)
        p95 = np.percentile(roi_results, 95)

        var_95 = np.percentile(roi_results, 5)
        worst_5_percent = roi_results[roi_results <= var_95]
        cvar_95 = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else var_95

        worst_case = np.min(roi_results)
        best_case = np.max(roi_results)

        prob_positive = np.sum(roi_results > 0) / len(roi_results)
        expected_value = mean_roi

        # Robustness score
        sharpe_ratio = mean_roi / std_roi if std_roi > 0 else 0
        robustness_score = max(0, min(100, 50 + sharpe_ratio * 10))

        # Risk level basato su Sharpe ratio
        if sharpe_ratio > 2.0:
            risk_level = "LOW"
        elif sharpe_ratio > 1.0:
            risk_level = "MEDIUM"
        elif sharpe_ratio > 0:
            risk_level = "HIGH"
        else:
            risk_level = "EXTREME"

        ci = (np.percentile(roi_results, 2.5), np.percentile(roi_results, 97.5))

        return MonteCarloResult(
            mean_outcome=mean_roi,
            median_outcome=median_roi,
            std_outcome=std_roi,
            percentile_5=p5,
            percentile_25=p25,
            percentile_75=p75,
            percentile_95=p95,
            var_95=var_95,
            cvar_95=cvar_95,
            worst_case=worst_case,
            best_case=best_case,
            probability_positive_ev=prob_positive,
            expected_value=expected_value,
            robustness_score=robustness_score,
            risk_level=risk_level,
            confidence_interval=ci,
            simulation_samples=roi_results
        )

    def stress_test(
        self,
        base_probability: float,
        base_odds: float,
        scenarios: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, StressTestResult]:
        """
        Esegue stress test sotto scenari estremi.

        Args:
            base_probability: Probabilità di base
            base_odds: Odds di base
            scenarios: Dict di scenari custom {name: {params}}

        Returns:
            Dict {scenario_name: StressTestResult}
        """
        if scenarios is None:
            # Scenari di default
            scenarios = {
                "high_uncertainty": {
                    "probability_shift": 0.0,
                    "uncertainty_multiplier": 3.0
                },
                "overestimation": {
                    "probability_shift": -0.10,
                    "uncertainty_multiplier": 1.5
                },
                "underestimation": {
                    "probability_shift": +0.10,
                    "uncertainty_multiplier": 1.5
                },
                "extreme_variance": {
                    "probability_shift": 0.0,
                    "uncertainty_multiplier": 5.0
                },
                "market_inefficiency": {
                    "probability_shift": -0.05,
                    "uncertainty_multiplier": 2.0
                }
            }

        results = {}

        for scenario_name, params in scenarios.items():
            prob_shift = params.get("probability_shift", 0.0)
            uncertainty_mult = params.get("uncertainty_multiplier", 1.0)

            # Simula con parametri modificati
            adjusted_prob = np.clip(base_probability + prob_shift, 0.01, 0.99)
            adjusted_uncertainty = 0.05 * uncertainty_mult

            mc_result = self.simulate_betting_roi(
                probability=adjusted_prob,
                odds=base_odds,
                probability_uncertainty=adjusted_uncertainty,
                n_bets=100
            )

            # Analizza impatto
            mean_impact = mc_result.mean_outcome
            worst_impact = mc_result.worst_case
            prob_failure = 1.0 - mc_result.probability_positive_ev

            # Resilience score: quanto bene il sistema resiste allo stress
            if mc_result.mean_outcome > 0 and mc_result.percentile_5 > -20:
                resilience_score = 90.0
                recommendations = ["Sistema resiliente sotto questo scenario"]
            elif mc_result.mean_outcome > 0:
                resilience_score = 70.0
                recommendations = [
                    "Sistema positivo ma con alta varianza",
                    "Considerare riduzione stake sotto questo scenario"
                ]
            elif mc_result.percentile_25 > 0:
                resilience_score = 50.0
                recommendations = [
                    "Rischio medio-alto sotto questo scenario",
                    "Raccomandato skip o stake molto ridotto"
                ]
            else:
                resilience_score = 20.0
                recommendations = [
                    "Sistema a rischio sotto questo scenario",
                    "SKIP consigliato",
                    "Rivedere model assumptions"
                ]

            results[scenario_name] = StressTestResult(
                scenario_name=scenario_name,
                mean_impact=mean_impact,
                worst_impact=worst_impact,
                probability_failure=prob_failure,
                resilience_score=resilience_score,
                recommendations=recommendations
            )

        return results

    def sensitivity_analysis(
        self,
        base_params: Dict,
        param_ranges: Dict[str, Tuple[float, float]],
        outcome_function: Callable
    ) -> Dict[str, np.ndarray]:
        """
        Analisi di sensitività: come cambiano gli outcome al variare dei parametri.

        Args:
            base_params: Parametri di base
            param_ranges: Range per ciascun parametro {param: (min, max)}
            outcome_function: Funzione che calcola outcome da parametri

        Returns:
            Dict {param_name: array of outcomes}
        """
        sensitivity_results = {}

        for param_name, (min_val, max_val) in param_ranges.items():
            # Varia questo parametro tenendo gli altri fissi
            param_values = np.linspace(min_val, max_val, 50)
            outcomes = []

            for val in param_values:
                # Crea params con questo valore
                test_params = base_params.copy()
                test_params[param_name] = val

                # Calcola outcome
                outcome = outcome_function(test_params)
                outcomes.append(outcome)

            sensitivity_results[param_name] = np.array(outcomes)

        return sensitivity_results

    def portfolio_simulation(
        self,
        bets: List[Dict],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> MonteCarloResult:
        """
        Simula performance di un portfolio di bets con correlazioni.

        Args:
            bets: Lista di dict {probability, odds, stake}
            correlation_matrix: Matrice di correlazione tra bets

        Returns:
            MonteCarloResult per il portfolio
        """
        n_bets = len(bets)

        if correlation_matrix is None:
            # Assume indipendenza
            correlation_matrix = np.eye(n_bets)

        portfolio_outcomes = []

        for _ in range(self.n_simulations):
            # Genera outcomes correlati
            # Usiamo copula Gaussiana
            z = np.random.multivariate_normal(np.zeros(n_bets), correlation_matrix)
            uniform_samples = stats.norm.cdf(z)

            total_profit = 0

            for i, bet in enumerate(bets):
                prob = bet['probability']
                odds = bet['odds']
                stake = bet['stake']

                # Outcome del bet basato sul campione uniforme
                win = uniform_samples[i] < prob

                if win:
                    total_profit += (odds - 1) * stake
                else:
                    total_profit -= stake

            portfolio_outcomes.append(total_profit)

        portfolio_outcomes = np.array(portfolio_outcomes)

        # Statistiche
        mean_profit = np.mean(portfolio_outcomes)
        median_profit = np.median(portfolio_outcomes)
        std_profit = np.std(portfolio_outcomes)

        p5 = np.percentile(portfolio_outcomes, 5)
        p25 = np.percentile(portfolio_outcomes, 25)
        p75 = np.percentile(portfolio_outcomes, 75)
        p95 = np.percentile(portfolio_outcomes, 95)

        var_95 = np.percentile(portfolio_outcomes, 5)
        worst_5_percent = portfolio_outcomes[portfolio_outcomes <= var_95]
        cvar_95 = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else var_95

        worst_case = np.min(portfolio_outcomes)
        best_case = np.max(portfolio_outcomes)

        prob_positive = np.sum(portfolio_outcomes > 0) / len(portfolio_outcomes)
        expected_value = mean_profit

        # Robustness
        sharpe = mean_profit / std_profit if std_profit > 0 else 0
        robustness_score = max(0, min(100, 50 + sharpe * 15))

        if sharpe > 1.5:
            risk_level = "LOW"
        elif sharpe > 0.75:
            risk_level = "MEDIUM"
        elif sharpe > 0:
            risk_level = "HIGH"
        else:
            risk_level = "EXTREME"

        ci = (np.percentile(portfolio_outcomes, 2.5), np.percentile(portfolio_outcomes, 97.5))

        return MonteCarloResult(
            mean_outcome=mean_profit,
            median_outcome=median_profit,
            std_outcome=std_profit,
            percentile_5=p5,
            percentile_25=p25,
            percentile_75=p75,
            percentile_95=p95,
            var_95=var_95,
            cvar_95=cvar_95,
            worst_case=worst_case,
            best_case=best_case,
            probability_positive_ev=prob_positive,
            expected_value=expected_value,
            robustness_score=robustness_score,
            risk_level=risk_level,
            confidence_interval=ci,
            simulation_samples=portfolio_outcomes
        )


if __name__ == "__main__":
    # Test del sistema
    simulator = MonteCarloSimulator(n_simulations=10000)

    # Test 1: Simulazione match
    print("=== TEST 1: Simulazione Match Outcome ===")
    result = simulator.simulate_match_outcome(
        lambda_home=1.5,
        lambda_away=1.2,
        uncertainty_home=0.15,
        uncertainty_away=0.15
    )
    print(f"Mean Outcome: {result.mean_outcome:.4f}")
    print(f"95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
    print(f"Robustness: {result.robustness_score:.1f}")
    print(f"Risk Level: {result.risk_level}")

    # Test 2: Simulazione ROI
    print("\n=== TEST 2: Simulazione Betting ROI ===")
    roi_result = simulator.simulate_betting_roi(
        probability=0.55,
        odds=2.0,
        stake=10.0,
        probability_uncertainty=0.05,
        n_bets=100
    )
    print(f"Expected ROI: {roi_result.mean_outcome:.2f}%")
    print(f"VaR 95%: {roi_result.var_95:.2f}%")
    print(f"CVaR 95%: {roi_result.cvar_95:.2f}%")
    print(f"Prob Positive: {roi_result.probability_positive_ev:.2%}")

    # Test 3: Stress Test
    print("\n=== TEST 3: Stress Test ===")
    stress_results = simulator.stress_test(
        base_probability=0.55,
        base_odds=2.0
    )
    for scenario, result in stress_results.items():
        print(f"\n{scenario}:")
        print(f"  Mean Impact: {result.mean_impact:.2f}%")
        print(f"  Resilience: {result.resilience_score:.1f}")
        print(f"  Recommendations: {result.recommendations[0]}")

    # Test 4: Portfolio
    print("\n=== TEST 4: Portfolio Simulation ===")
    bets = [
        {"probability": 0.55, "odds": 2.0, "stake": 10},
        {"probability": 0.60, "odds": 1.8, "stake": 15},
        {"probability": 0.50, "odds": 2.2, "stake": 8}
    ]
    portfolio_result = simulator.portfolio_simulation(bets)
    print(f"Expected Profit: ${portfolio_result.mean_outcome:.2f}")
    print(f"VaR 95%: ${portfolio_result.var_95:.2f}")
    print(f"Robustness: {portfolio_result.robustness_score:.1f}")
