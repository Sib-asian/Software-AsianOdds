"""
Generic LLM client wrapper supporting OpenAI-compatible APIs and Anthropic.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    OPENAI_COMPATIBLE = {"openai", "openrouter", "azure", "custom"}

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> str:
        if self.provider == "mock":
            return self._mock_response(messages)

        if not self.api_key:
            raise RuntimeError("LLM API key non configurata.")

        if self.provider in self.OPENAI_COMPATIBLE:
            return self._call_openai_compatible(messages, temperature, max_tokens)
        if self.provider == "anthropic":
            return self._call_anthropic(messages, temperature, max_tokens)

        raise ValueError(f"Provider LLM non supportato: {self.provider}")

    # -------------------------
    # Providers
    # -------------------------

    def _call_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = self.base_url or self._default_base()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Risposta LLM inattesa: {data}")

    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = self.base_url or "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        system_prompt = ""
        chat_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_prompt += content + "\n"
            else:
                chat_messages.append({"role": role, "content": content})

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt.strip() or None,
            "messages": chat_messages,
        }
        response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        try:
            return "".join(part.get("text", "") for part in data["content"]).strip()
        except (KeyError, TypeError):
            raise RuntimeError(f"Risposta LLM inattesa: {data}")

    # -------------------------
    # Helpers
    # -------------------------

    def _default_base(self) -> str:
        if self.provider == "openrouter":
            return "https://openrouter.ai/api/v1/chat/completions"
        if self.provider == "azure":
            raise ValueError("Specificare base_url per provider Azure.")
        return "https://api.openai.com/v1/chat/completions"

    @staticmethod
    def _mock_response(messages: List[Dict[str, str]]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"(Mock LLM) Risposta automatica alla richiesta: {last_user[:120]}"
