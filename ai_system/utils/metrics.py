"""
Performance Metrics
===================

Funzioni per calcolare metriche di performance del sistema.
"""

import numpy as np
from typing import List, Tuple


def calculate_brier_score(
    predictions: List[float],
    outcomes: List[int]
) -> float:
    """
    Calcola Brier Score (lower = better).

    Brier Score = mean((prediction - outcome)^2)

    Perfect score = 0.0
    Random score ≈ 0.25
    """
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    return float(np.mean((predictions - outcomes) ** 2))


def calculate_roi(
    stakes: List[float],
    payouts: List[float]
) -> float:
    """
    Calcola Return on Investment.

    ROI = (total_profit / total_staked) * 100
    """
    total_staked = sum(stakes)
    total_returned = sum(payouts)
    total_profit = total_returned - total_staked

    if total_staked == 0:
        return 0.0

    roi = (total_profit / total_staked) * 100
    return float(roi)


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0
) -> float:
    """
    Calcola Sharpe Ratio (risk-adjusted returns).

    Sharpe = (mean_return - risk_free_rate) / std_return

    > 1.0 = Good
    > 2.0 = Very good
    > 3.0 = Excellent
    """
    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe)


def calculate_win_rate(outcomes: List[int]) -> float:
    """Calcola win rate percentage"""
    return float(np.mean(outcomes) * 100)


def calculate_max_drawdown(cumulative_returns: List[float]) -> float:
    """Calcola maximum drawdown (worst peak-to-trough decline)"""
    cumulative = np.array(cumulative_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown))
    return max_dd


if __name__ == "__main__":
    # Test metrics
    predictions = [0.6, 0.7, 0.5, 0.8, 0.55]
    outcomes = [1, 1, 0, 1, 0]
    stakes = [10, 10, 10, 10, 10]
    payouts = [18, 0, 0, 16, 20]  # 3 wins, 2 losses

    print(f"Brier Score: {calculate_brier_score(predictions, outcomes):.4f}")
    print(f"ROI: {calculate_roi(stakes, payouts):.1f}%")
    print(f"Win Rate: {calculate_win_rate(outcomes):.1f}%")

    returns = [(p - s) / s for s, p in zip(stakes, payouts)]
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(returns):.2f}")
