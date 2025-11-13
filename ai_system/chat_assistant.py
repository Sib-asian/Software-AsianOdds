"""
Conversational Assistant with Lightweight Memory
=================================================

This module exposes a simple conversational interface that wraps the AI
pipeline. It does not rely on external LLM services, but provides deterministic
answers derived from the latest analysis artefacts (probability, confidence,
news sentiment, minor league signals, odds anomalies, etc.).
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ConversationSession:
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    turns: Deque[ConversationTurn] = field(default_factory=deque)

    def add_turn(self, role: str, content: str, max_turns: int) -> None:
        self.turns.append(ConversationTurn(role=role, content=content))
        while len(self.turns) > max_turns:
            self.turns.popleft()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "turns": [{"role": turn.role, "content": turn.content} for turn in self.turns],
        }


class AIAssistantChat:
    """Stateful chat facade for the AI pipeline."""

    def __init__(self, config) -> None:
        self.config = config
        self.sessions: Dict[str, ConversationSession] = {}
        self.latest_analysis: Optional[Dict[str, Any]] = None
        self.news_context: Optional[Dict[str, Any]] = None
        self.minor_league_context: Optional[Dict[str, Any]] = None
        self.anomaly_context: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #

    def start_session(self, session_id: Optional[str] = None, **metadata) -> str:
        session_id = session_id or str(uuid.uuid4())
        session = ConversationSession(session_id=session_id, metadata=metadata)
        self.sessions[session_id] = session
        logger.debug("💬 New chat session started: %s", session_id)
        return session_id

    def _get_session(self, session_id: Optional[str]) -> ConversationSession:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return self.sessions.setdefault(
            "default",
            ConversationSession(session_id="default"),
        )

    # ------------------------------------------------------------------ #
    # Context updates
    # ------------------------------------------------------------------ #

    def update_context(
        self,
        analysis: Optional[Dict[str, Any]] = None,
        news: Optional[Dict[str, Any]] = None,
        minor_league: Optional[Dict[str, Any]] = None,
        anomalies: Optional[Dict[str, Any]] = None,
    ) -> None:
        if analysis:
            self.latest_analysis = analysis
        if news:
            self.news_context = news
        if minor_league:
            self.minor_league_context = minor_league
        if anomalies:
            self.anomaly_context = anomalies

    # ------------------------------------------------------------------ #
    # Core interaction
    # ------------------------------------------------------------------ #

    def respond(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        session = self._get_session(session_id)
        session.add_turn("user", message, self.config.chat_memory_max_turns)

        response = self._generate_response(message)
        session.add_turn("assistant", response, self.config.chat_memory_max_turns)

        return {
            "session_id": session.session_id,
            "response": response,
            "turns": session.to_dict()["turns"],
        }

    # ------------------------------------------------------------------ #
    # Response logic
    # ------------------------------------------------------------------ #

    def _generate_response(self, message: str) -> str:
        if not self.latest_analysis:
            return (
                "Non ho ancora un'analisi disponibile. Avvia l'AI pipeline e poi "
                "ripeti la domanda."
            )

        message_lower = message.lower()
        blocks = []

        # Probability / confidence questions
        if any(kw in message_lower for kw in ("probabilità", "chance", "confidence", "affidabilità")):
            blocks.append(self._format_prob_confidence())

        # Stake / Kelly
        if any(kw in message_lower for kw in ("stake", "kelly", "puntata", "bankroll")):
            blocks.append(self._format_stake_info())

        # News / sentiment
        if any(kw in message_lower for kw in ("news", "rumor", "sentiment", "notizie")) and self.news_context:
            blocks.append(self._format_news_sentiment())

        # Minor league data
        if any(kw in message_lower for kw in ("lega minore", "campionati minori", "dati lega")) and self.minor_league_context:
            blocks.append(self._format_minor_league())

        # Odds anomalies
        if any(kw in message_lower for kw in ("anomalia", "quote strane", "movimento quote")) and self.anomaly_context:
            blocks.append(self._format_anomalies())

        # Generic fallback summarizing latest analysis
        if not blocks:
            blocks.append(self._format_overview())

        return "\n\n".join(blocks)

    def _format_prob_confidence(self) -> str:
        summary = self.latest_analysis.get("summary", {})
        confidence = summary.get("confidence")
        probability = summary.get("probability")
        confidence_level = self.latest_analysis.get("confidence", {}).get("confidence_level")
        recommendation = self.latest_analysis.get("confidence", {}).get("recommendation")

        return (
            f"Probabilità calibrata: {probability:.1%}.\n"
            f"Confidence score: {confidence:.0f}/100 ({confidence_level}).\n"
            f"{recommendation}"
        )

    def _format_stake_info(self) -> str:
        risk_decision = self.latest_analysis.get("risk_decision", {})
        stake = risk_decision.get("final_stake", 0.0)
        decision = risk_decision.get("decision", "N/A")
        kelly_fraction = self.latest_analysis.get("kelly", {}).get("kelly_fraction", 0.0)
        timing = self.latest_analysis.get("timing", {}).get("timing_recommendation", "N/A")

        return (
            f"Decisione finale: {decision}.\n"
            f"Stake consigliato: €{stake:.2f} (Kelly frazione {kelly_fraction:.2f}).\n"
            f"Timing suggerito: {timing}."
        )

    def _format_news_sentiment(self) -> str:
        if not self.news_context:
            return "Non ho segnali di news o sentiment da mostrare."

        aggregate = self.news_context.get("aggregate", {})
        bias = aggregate.get("bias", "neutral")
        confidence = aggregate.get("confidence", 0.0)

        def _join_highlights(side: str) -> str:
            side_ctx = self.news_context.get(side) or {}
            highlights = side_ctx.get("highlights") or []
            if not highlights:
                return "Nessun highlight rilevante."
            return "; ".join(highlights[:3])

        return (
            f"Sentiment aggregator: bias {bias} (confidenza {confidence:.1f}).\n"
            f"Casa → {_join_highlights('home')}\n"
            f"Trasferta → {_join_highlights('away')}"
        )

    def _format_minor_league(self) -> str:
        if not self.minor_league_context:
            return "Nessun dato speciale sui campionati minori disponibile."

        ctx = self.minor_league_context
        adjustments = ctx.get("adjustments", {})
        risks = adjustments.get("flagged_risks", [])
        opportunities = adjustments.get("flagged_opportunities", [])

        return (
            f"Lega: {ctx.get('league', 'N/A')} (copertura {ctx.get('coverage_level', 'n/d')}).\n"
            f"Qualità dataset: {ctx.get('data_quality', 0.0):.2f} "
            f"aggiornata ~{ctx.get('freshness_hours', 0.0):.0f}h fa.\n"
            f"Rischi: {', '.join(risks) if risks else 'nessuno'}.\n"
            f"Opportunità: {', '.join(opportunities) if opportunities else 'nessuna'}."
        )

    def _format_anomalies(self) -> str:
        if not self.anomaly_context:
            return "Quote stabili, nessuna anomalia registrata."

        anomalies = self.anomaly_context.get("anomalies", [])
        status = self.anomaly_context.get("status", "stable")
        explanation = "; ".join(a.get("description") for a in anomalies[:3]) if anomalies else "nessuna"

        return (
            f"Stato movimenti quote: {status}.\n"
            f"Anomalie principali: {explanation}."
        )

    def _format_overview(self) -> str:
        summary = self.latest_analysis.get("summary", {})
        timing = self.latest_analysis.get("timing", {})
        news_bias = (self.news_context or {}).get("aggregate", {}).get("bias", "neutral")

        lines = [
            f"Probabilità: {summary.get('probability', 0.0):.1%}",
            f"Confidence: {summary.get('confidence', 0.0):.0f}/100",
            f"Expected value: {summary.get('expected_value', 0.0):+.1%}",
            f"Stake: €{summary.get('stake', 0.0):.2f}",
            f"Timing: {timing.get('timing_recommendation', 'N/A')} (urgenza {timing.get('urgency', 'N/A')})",
            f"Sentiment bias: {news_bias}",
        ]

        if self.anomaly_context and self.anomaly_context.get("anomalies"):
            lines.append(f"⚠️ Anomalie quote: {len(self.anomaly_context['anomalies'])}")

        return "\n".join(lines)

