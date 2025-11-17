"""
BLOCCO 7: Bayesian Uncertainty Quantification System

Questo modulo implementa un sistema Bayesiano avanzato per quantificare
l'incertezza nelle previsioni e calcolare confidence intervals probabilistici.

Features:
- Bayesian credible intervals
- Posterior probability distributions
- Monte Carlo Markov Chain (MCMC) sampling
- Uncertainty propagation
- Beta-Binomial conjugate priors
- Hierarchical Bayesian modeling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.special import beta as beta_func
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BayesianResult:
    """Risultato dell'analisi Bayesiana"""
    mean: float
    median: float
    mode: float
    std: float
    credible_interval_95: Tuple[float, float]
    credible_interval_99: Tuple[float, float]
    posterior_samples: np.ndarray
    uncertainty_level: str  # LOW, MEDIUM, HIGH, VERY_HIGH
    confidence_score: float  # 0-100
    reliability_index: float  # 0-1


class BayesianUncertaintyQuantifier:
    """
    Sistema Bayesiano per quantificazione dell'incertezza.

    Utilizza metodi Bayesiani per calcolare distribuzioni posteriori
    e intervalli di credibilità per le previsioni.
    """

    def __init__(self, n_samples: int = 10000, random_state: int = 42):
        """
        Args:
            n_samples: Numero di campioni per MCMC
            random_state: Seed per riproducibilità
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)

        # Prior parameters (Beta distribution)
        # Inizializzati con prior non informativo (uniform)
        self.alpha_prior = 1.0
        self.beta_prior = 1.0

        # Storage per historical data
        self.historical_predictions = []
        self.historical_outcomes = []

    def update_prior(self, successes: int, failures: int):
        """
        Aggiorna i parametri del prior basandosi sui dati storici.

        Args:
            successes: Numero di successi osservati
            failures: Numero di fallimenti osservati
        """
        self.alpha_prior = 1.0 + successes
        self.beta_prior = 1.0 + failures

    def calculate_posterior(
        self,
        predicted_prob: float,
        historical_success_rate: Optional[float] = None,
        n_historical: Optional[int] = None
    ) -> BayesianResult:
        """
        Calcola la distribuzione posteriore per una probabilità predetta.

        Args:
            predicted_prob: Probabilità predetta dal modello
            historical_success_rate: Tasso di successo storico (opzionale)
            n_historical: Numero di osservazioni storiche (opzionale)

        Returns:
            BayesianResult con statistiche posteriori
        """
        # Se abbiamo dati storici, aggiorniamo il prior
        if historical_success_rate is not None and n_historical is not None:
            successes = int(historical_success_rate * n_historical)
            failures = n_historical - successes
            alpha_post = self.alpha_prior + successes
            beta_post = self.beta_prior + failures
        else:
            # Usiamo la predizione come likelihood con prior debolmente informativo
            # Convertiamo la predizione in "pseudo-observations"
            pseudo_n = 10  # Peso della predizione
            alpha_post = self.alpha_prior + predicted_prob * pseudo_n
            beta_post = self.beta_prior + (1 - predicted_prob) * pseudo_n

        # Generiamo campioni dalla distribuzione posteriore (Beta)
        posterior_samples = np.random.beta(alpha_post, beta_post, self.n_samples)

        # Calcoliamo statistiche
        mean = alpha_post / (alpha_post + beta_post)
        mode = (alpha_post - 1) / (alpha_post + beta_post - 2) if alpha_post > 1 and beta_post > 1 else mean
        median = np.median(posterior_samples)
        std = np.std(posterior_samples)

        # Credible intervals
        ci_95 = (np.percentile(posterior_samples, 2.5), np.percentile(posterior_samples, 97.5))
        ci_99 = (np.percentile(posterior_samples, 0.5), np.percentile(posterior_samples, 99.5))

        # Calcola uncertainty level
        uncertainty_width = ci_95[1] - ci_95[0]
        if uncertainty_width < 0.1:
            uncertainty_level = "LOW"
            confidence_score = 95.0
        elif uncertainty_width < 0.2:
            uncertainty_level = "MEDIUM"
            confidence_score = 75.0
        elif uncertainty_width < 0.3:
            uncertainty_level = "HIGH"
            confidence_score = 50.0
        else:
            uncertainty_level = "VERY_HIGH"
            confidence_score = 25.0

        # Reliability index basato sulla concentrazione della distribuzione
        # Usiamo il coefficiente di variazione inverso
        cv = std / mean if mean > 0 else 1.0
        reliability_index = max(0.0, min(1.0, 1.0 - cv))

        return BayesianResult(
            mean=mean,
            median=median,
            mode=mode,
            std=std,
            credible_interval_95=ci_95,
            credible_interval_99=ci_99,
            posterior_samples=posterior_samples,
            uncertainty_level=uncertainty_level,
            confidence_score=confidence_score,
            reliability_index=reliability_index
        )

    def bayesian_ensemble(
        self,
        predictions: List[float],
        model_reliabilities: Optional[List[float]] = None
    ) -> BayesianResult:
        """
        Combina previsioni multiple usando Bayesian Model Averaging.

        Args:
            predictions: Lista di probabilità predette da diversi modelli
            model_reliabilities: Affidabilità di ciascun modello (0-1)

        Returns:
            BayesianResult con ensemble bayesiano
        """
        predictions = np.array(predictions)

        if model_reliabilities is None:
            # Pesi uniformi
            weights = np.ones(len(predictions)) / len(predictions)
        else:
            # Pesi basati su reliability
            weights = np.array(model_reliabilities)
            weights = weights / np.sum(weights)

        # Bayesian Model Averaging
        # Generiamo campioni da ciascun modello e li combiniamo
        all_samples = []
        for pred, weight in zip(predictions, weights):
            # Generiamo campioni per questo modello
            # Usiamo Beta con concentration parameter basato sulla reliability
            concentration = 20 if model_reliabilities is None else 20 * weight
            alpha = pred * concentration
            beta = (1 - pred) * concentration
            samples = np.random.beta(max(1, alpha), max(1, beta), int(self.n_samples * weight))
            all_samples.extend(samples)

        posterior_samples = np.array(all_samples)

        # Calcoliamo statistiche
        mean = np.mean(posterior_samples)
        median = np.median(posterior_samples)
        mode = 3 * median - 2 * mean  # Approssimazione per mode
        std = np.std(posterior_samples)

        ci_95 = (np.percentile(posterior_samples, 2.5), np.percentile(posterior_samples, 97.5))
        ci_99 = (np.percentile(posterior_samples, 0.5), np.percentile(posterior_samples, 99.5))

        # Uncertainty level
        uncertainty_width = ci_95[1] - ci_95[0]
        if uncertainty_width < 0.1:
            uncertainty_level = "LOW"
            confidence_score = 95.0
        elif uncertainty_width < 0.2:
            uncertainty_level = "MEDIUM"
            confidence_score = 75.0
        elif uncertainty_width < 0.3:
            uncertainty_level = "HIGH"
            confidence_score = 50.0
        else:
            uncertainty_level = "VERY_HIGH"
            confidence_score = 25.0

        cv = std / mean if mean > 0 else 1.0
        reliability_index = max(0.0, min(1.0, 1.0 - cv))

        return BayesianResult(
            mean=mean,
            median=median,
            mode=mode,
            std=std,
            credible_interval_95=ci_95,
            credible_interval_99=ci_99,
            posterior_samples=posterior_samples,
            uncertainty_level=uncertainty_level,
            confidence_score=confidence_score,
            reliability_index=reliability_index
        )

    def probability_calibration_check(
        self,
        predictions: List[float],
        outcomes: List[bool]
    ) -> Dict:
        """
        Verifica la calibrazione delle probabilità usando analisi Bayesiana.

        Args:
            predictions: Lista di probabilità predette
            outcomes: Lista di outcome effettivi (True/False)

        Returns:
            Dict con metriche di calibrazione
        """
        predictions = np.array(predictions)
        outcomes = np.array(outcomes, dtype=float)

        # Dividiamo in bins
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1

        calibration_results = {}
        expected_calibration_error = 0.0

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) == 0:
                continue

            bin_predictions = predictions[mask]
            bin_outcomes = outcomes[mask]

            # Predicted probability (mean of bin)
            pred_prob = np.mean(bin_predictions)

            # Observed frequency
            obs_freq = np.mean(bin_outcomes)
            n_obs = len(bin_outcomes)

            # Bayesian credible interval per observed frequency
            alpha_obs = 1 + np.sum(bin_outcomes)
            beta_obs = 1 + n_obs - np.sum(bin_outcomes)
            ci_lower = stats.beta.ppf(0.025, alpha_obs, beta_obs)
            ci_upper = stats.beta.ppf(0.975, alpha_obs, beta_obs)

            calibration_results[f"bin_{i}"] = {
                "predicted_prob": pred_prob,
                "observed_freq": obs_freq,
                "n_samples": n_obs,
                "credible_interval": (ci_lower, ci_upper),
                "calibration_error": abs(pred_prob - obs_freq)
            }

            # ECE contribution
            expected_calibration_error += (n_obs / len(predictions)) * abs(pred_prob - obs_freq)

        return {
            "expected_calibration_error": expected_calibration_error,
            "bin_results": calibration_results,
            "n_predictions": len(predictions),
            "overall_accuracy": np.mean(outcomes)
        }

    def hierarchical_uncertainty(
        self,
        predictions_by_league: Dict[str, List[float]],
        outcomes_by_league: Dict[str, List[bool]]
    ) -> Dict[str, BayesianResult]:
        """
        Modello gerarchico Bayesiano per incertezza per league.

        Args:
            predictions_by_league: Dict {league: [predictions]}
            outcomes_by_league: Dict {league: [outcomes]}

        Returns:
            Dict {league: BayesianResult}
        """
        results = {}

        # Calcola prior globale da tutti i dati
        all_outcomes = []
        for outcomes in outcomes_by_league.values():
            all_outcomes.extend(outcomes)

        global_success_rate = np.mean(all_outcomes) if all_outcomes else 0.5
        global_n = len(all_outcomes)

        # Per ogni league, calcola posterior
        for league in predictions_by_league.keys():
            preds = predictions_by_league[league]
            outcomes = outcomes_by_league[league]

            if len(outcomes) == 0:
                continue

            # Success rate per questa league
            league_success_rate = np.mean(outcomes)
            league_n = len(outcomes)

            # Usiamo il global come prior, league-specific come likelihood
            # Weighted average tra prior globale e likelihood locale
            weight_prior = global_n / (global_n + league_n)
            weight_likelihood = league_n / (global_n + league_n)

            combined_success_rate = (weight_prior * global_success_rate +
                                   weight_likelihood * league_success_rate)

            # Calcola posterior
            result = self.calculate_posterior(
                predicted_prob=np.mean(preds),
                historical_success_rate=combined_success_rate,
                n_historical=league_n
            )

            results[league] = result

        return results

    def get_adjusted_probability(
        self,
        predicted_prob: float,
        bayesian_result: BayesianResult,
        use_conservative: bool = True
    ) -> float:
        """
        Ottiene probabilità aggiustata basandosi sull'analisi Bayesiana.

        Args:
            predicted_prob: Probabilità originale predetta
            bayesian_result: Risultato dell'analisi Bayesiana
            use_conservative: Se True, usa bound conservativo

        Returns:
            Probabilità aggiustata
        """
        if use_conservative:
            # Usiamo il lower bound del 95% CI per essere conservativi
            if bayesian_result.uncertainty_level in ["HIGH", "VERY_HIGH"]:
                adjusted = bayesian_result.credible_interval_95[0]
            else:
                adjusted = bayesian_result.median
        else:
            # Usiamo la media posteriore
            adjusted = bayesian_result.mean

        # Clip to valid probability range
        return max(0.0, min(1.0, adjusted))


def run_bayesian_analysis(
    prediction: float,
    ensemble_predictions: List[float],
    model_reliabilities: Optional[List[float]] = None,
    historical_data: Optional[Dict] = None
) -> Dict:
    """
    Funzione helper per eseguire analisi Bayesiana completa.

    Args:
        prediction: Probabilità predetta principale
        ensemble_predictions: Predizioni da diversi modelli
        model_reliabilities: Affidabilità dei modelli
        historical_data: Dati storici opzionali

    Returns:
        Dict con tutti i risultati Bayesiani
    """
    quantifier = BayesianUncertaintyQuantifier()

    # Analisi della singola predizione
    single_result = quantifier.calculate_posterior(prediction)

    # Ensemble Bayesiano
    ensemble_result = quantifier.bayesian_ensemble(
        ensemble_predictions,
        model_reliabilities
    )

    # Probabilità aggiustate
    conservative_prob = quantifier.get_adjusted_probability(
        prediction, ensemble_result, use_conservative=True
    )
    optimistic_prob = quantifier.get_adjusted_probability(
        prediction, ensemble_result, use_conservative=False
    )

    return {
        "single_prediction_analysis": single_result,
        "ensemble_analysis": ensemble_result,
        "conservative_probability": conservative_prob,
        "optimistic_probability": optimistic_prob,
        "recommended_probability": ensemble_result.median,
        "uncertainty_level": ensemble_result.uncertainty_level,
        "confidence_score": ensemble_result.confidence_score,
        "reliability_index": ensemble_result.reliability_index,
        "credible_interval_95": ensemble_result.credible_interval_95,
        "credible_interval_99": ensemble_result.credible_interval_99
    }


if __name__ == "__main__":
    # Test del sistema
    quantifier = BayesianUncertaintyQuantifier()

    # Test 1: Singola predizione
    print("=== TEST 1: Singola Predizione ===")
    result = quantifier.calculate_posterior(0.65)
    print(f"Mean: {result.mean:.4f}")
    print(f"95% CI: [{result.credible_interval_95[0]:.4f}, {result.credible_interval_95[1]:.4f}]")
    print(f"Uncertainty: {result.uncertainty_level}")
    print(f"Confidence: {result.confidence_score:.1f}")

    # Test 2: Ensemble
    print("\n=== TEST 2: Bayesian Ensemble ===")
    predictions = [0.60, 0.65, 0.62, 0.68]
    reliabilities = [0.85, 0.90, 0.80, 0.88]
    ensemble_result = quantifier.bayesian_ensemble(predictions, reliabilities)
    print(f"Ensemble Mean: {ensemble_result.mean:.4f}")
    print(f"95% CI: [{ensemble_result.credible_interval_95[0]:.4f}, {ensemble_result.credible_interval_95[1]:.4f}]")
    print(f"Reliability Index: {ensemble_result.reliability_index:.4f}")

    # Test 3: Analisi completa
    print("\n=== TEST 3: Analisi Completa ===")
    full_analysis = run_bayesian_analysis(
        prediction=0.65,
        ensemble_predictions=[0.60, 0.65, 0.62, 0.68],
        model_reliabilities=[0.85, 0.90, 0.80, 0.88]
    )
    print(f"Conservative: {full_analysis['conservative_probability']:.4f}")
    print(f"Recommended: {full_analysis['recommended_probability']:.4f}")
    print(f"Optimistic: {full_analysis['optimistic_probability']:.4f}")
    print(f"Uncertainty: {full_analysis['uncertainty_level']}")
