"""
Alert playbook executor.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class AlertPlaybook:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, List[str]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def actions_for_alerts(self, alerts) -> List[str]:
        actions: List[str] = []
        for alert in alerts or []:
            code = alert.get("code")
            if not code:
                continue
            cmds = self.rules.get(code)
            if not cmds:
                continue
            if isinstance(cmds, str):
                actions.append(cmds)
            else:
                actions.extend(cmds)
        return actions

    def execute(self, alerts, *, dry_run: bool = False) -> List[subprocess.CompletedProcess]:
        actions = self.actions_for_alerts(alerts)
        results: List[subprocess.CompletedProcess] = []
        for cmd in actions:
            if dry_run:
                print(f"[playbook] DRY-RUN: {cmd}")
                continue
            print(f"[playbook] executing: {cmd}")
            results.append(subprocess.run(cmd, shell=True, check=False))
        return results


__all__ = ["AlertPlaybook"]
