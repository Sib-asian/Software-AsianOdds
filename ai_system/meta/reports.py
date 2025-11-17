"""
Utilities to build human-readable reports of the meta-layer status.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_alert(alert: Dict[str, Any]) -> str:
    code = alert.get("code", "unknown")
    level = alert.get("level", "info").upper()
    message = alert.get("message") or ""
    action = alert.get("action")
    text = f"[{level}] {code}: {message}"
    if action:
        text += f" (azione suggerita: {action})"
    return text


def _format_summary(title: str, summary: Dict[str, Any]) -> List[str]:
    lines: List[str] = [title]
    if not summary:
        lines.append("  (nessun dato)")
        return lines
    lines.append(f"  • Totale entry: {summary.get('total_entries', 0)}")
    if summary.get("avg_probability") is not None:
        lines.append(f"  • Probabilità media: {summary['avg_probability']:.3f}")
    if summary.get("probability_rmse") is not None:
        lines.append(f"  • RMSE probabilità: {summary['probability_rmse']:.3f}")
    if summary.get("outcome_ratio") is not None:
        lines.append(f"  • Outcome ratio: {summary['outcome_ratio']:.1%}")
    return lines


def summarize_meta_health(health: Dict[str, Any]) -> str:
    lines: List[str] = []
    if not health:
        return "Meta layer non disponibile."

    summary = health.get("summary") or {}
    lines.append("META LAYER HEALTH")
    lines.append(f"- Store path: {health.get('store_path')}")
    lines.append(f"- Entries: {health.get('entries', 0)}")
    lines.append(f"- Exploration rate: {health.get('exploration_rate')}")
    lines.extend(_format_summary("- Finestra principale:", summary))

    window_summaries = health.get("window_summaries") or {}
    if window_summaries:
        lines.append("\nFinestre aggiuntive:")
        for window, data in window_summaries.items():
            lines.extend(_format_summary(f"  [{window}]", data))

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
