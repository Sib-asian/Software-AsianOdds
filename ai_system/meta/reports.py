"""
Utilities to build human-readable reports of the meta-layer status.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def format_alert(alert: Dict[str, Any]) -> str:
    code = alert.get("code", "unknown")
    level = alert.get("level", "info").upper()
    message = alert.get("message") or ""
    action = alert.get("action")
    text = f"[{level}] {code}: {message}"
    if action:
        text += f" (azione suggerita: {action})"
    return text


def summarize_meta_health(health: Dict[str, Any]) -> str:
    lines: List[str] = []
    if not health:
        return "Meta layer non disponibile."

    summary = health.get("summary") or {}
    lines.append("META LAYER HEALTH")
    lines.append(f"- Store path: {health.get('store_path')}")
    lines.append(f"- Entries: {health.get('entries', 0)}")
    lines.append(f"- Exploration rate: {health.get('exploration_rate')}")
    if summary:
        lines.append(f"- Avg probability: {summary.get('avg_probability'):.3f}")
        if summary.get("probability_rmse") is not None:
            lines.append(f"- Probability RMSE: {summary['probability_rmse']:.3f}")
        lines.append(f"- Outcome ratio: {summary.get('outcome_ratio', 0.0):.1%}")

    reliability = health.get("reliability") or {}
    if reliability:
        lines.append("\nAffidabilità corrente:")
        for model, value in reliability.items():
            lines.append(f"  • {model}: {value:.3f}")

    alerts = health.get("alerts") or []
    if alerts:
        lines.append("\nAlert:")
        for alert in alerts:
            lines.append(f"  - {format_alert(alert)}")
    else:
        lines.append("\nNessun alert attivo.")

    return "\n".join(lines)


__all__ = ["summarize_meta_health", "format_alert"]
