#!/usr/bin/env python3
"""Script per avviare il sistema in background in modo robusto"""

import subprocess
import sys
import os
from pathlib import Path

# Cambia directory
os.chdir(Path(__file__).parent)

# Avvia in background usando subprocess
print("🚀 Avvio sistema in background...")
process = subprocess.Popen(
    [sys.executable, "automation_service_wrapper.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
)

print(f"✅ Sistema avviato (PID: {process.pid})")
print("📊 I log sono in: logs/automation_service_*.log")
print("🛑 Per fermare: FERMA_24H.bat")


