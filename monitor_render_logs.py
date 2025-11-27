#!/usr/bin/env python3
"""
Monitor Logs Render.com in tempo reale
=======================================

Usage:
    python monitor_render_logs.py --api-key YOUR_API_KEY --service-id YOUR_SERVICE_ID

Oppure configura le variabili d'ambiente:
    export RENDER_API_KEY=your_api_key
    export RENDER_SERVICE_ID=your_service_id
    python monitor_render_logs.py
"""

import requests
import time
import sys
import os
import argparse
from datetime import datetime
from typing import Optional

# Colori per output (Windows support)
try:
    from colorama import init, Fore, Style
    init()
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    BLUE = Fore.BLUE
    CYAN = Fore.CYAN
    RESET = Style.RESET_ALL
except ImportError:
    GREEN = YELLOW = RED = BLUE = CYAN = RESET = ""


class RenderLogMonitor:
    """Monitor per log Render.com"""
    
    BASE_URL = "https://api.render.com/v1"
    
    def __init__(self, api_key: str, service_id: str):
        self.api_key = api_key
        self.service_id = service_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        self.last_log_id = None
    
    def get_logs(self, limit: int = 100) -> list:
        """Recupera log dal servizio Render"""
        try:
            url = f"{self.BASE_URL}/services/{self.service_id}/logs"
            params = {
                "limit": limit,
                "cursor": self.last_log_id
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logs = data.get("logs", [])
            
            if logs:
                self.last_log_id = logs[-1].get("id")
            
            return logs
        except requests.exceptions.RequestException as e:
            print(f"{RED}❌ Errore recupero log: {e}{RESET}")
            return []
    
    def format_log(self, log: dict) -> str:
        """Formatta un log per la visualizzazione"""
        timestamp = log.get("timestamp", "")
        level = log.get("level", "INFO").upper()
        message = log.get("message", "")
        
        # Colori per livello
        color_map = {
            "ERROR": RED,
            "WARN": YELLOW,
            "INFO": BLUE,
            "DEBUG": CYAN
        }
        color = color_map.get(level, RESET)
        
        # Formatta timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = timestamp[:8] if len(timestamp) > 8 else timestamp
        
        return f"{CYAN}[{time_str}]{RESET} {color}[{level}]{RESET} {message}"
    
    def monitor(self, interval: float = 2.0, filter_level: Optional[str] = None, filter_text: Optional[str] = None):
        """Monitora log in tempo reale"""
        print(f"{GREEN}🔍 Monitoraggio log Render.com{RESET}")
        print(f"{CYAN}Service ID: {self.service_id}{RESET}")
        print(f"{CYAN}Intervallo: {interval}s{RESET}")
        if filter_level:
            print(f"{CYAN}Filtro livello: {filter_level}{RESET}")
        if filter_text:
            print(f"{CYAN}Filtro testo: {filter_text}{RESET}")
        print(f"{GREEN}{'='*60}{RESET}\n")
        
        # Recupera log iniziali
        initial_logs = self.get_logs(limit=50)
        if initial_logs:
            print(f"{BLUE}📜 Ultimi {len(initial_logs)} log:{RESET}\n")
            for log in initial_logs[-10:]:  # Mostra ultimi 10
                formatted = self.format_log(log)
                if self._should_show_log(log, filter_level, filter_text):
                    print(formatted)
            print(f"\n{GREEN}{'='*60}{RESET}\n")
            print(f"{GREEN}▶️  Monitoraggio live (Ctrl+C per uscire)...{RESET}\n")
        
        try:
            while True:
                logs = self.get_logs(limit=20)
                
                if logs:
                    for log in reversed(logs):  # Mostra dal più vecchio al più nuovo
                        if self._should_show_log(log, filter_level, filter_text):
                            print(self.format_log(log))
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n{GREEN}✅ Monitoraggio interrotto{RESET}")
    
    def _should_show_log(self, log: dict, filter_level: Optional[str], filter_text: Optional[str]) -> bool:
        """Verifica se il log deve essere mostrato in base ai filtri"""
        if filter_level:
            if log.get("level", "").upper() != filter_level.upper():
                return False
        
        if filter_text:
            message = log.get("message", "").lower()
            if filter_text.lower() not in message:
                return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Monitor log Render.com in tempo reale")
    parser.add_argument("--api-key", help="Render API Key", default=os.getenv("RENDER_API_KEY"))
    parser.add_argument("--service-id", help="Render Service ID", default=os.getenv("RENDER_SERVICE_ID"))
    parser.add_argument("--interval", type=float, default=2.0, help="Intervallo aggiornamento (secondi)")
    parser.add_argument("--level", help="Filtra per livello (ERROR, WARN, INFO, DEBUG)")
    parser.add_argument("--filter", help="Filtra per testo nel messaggio")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print(f"{RED}❌ Errore: API Key non fornita{RESET}")
        print("Usa --api-key o imposta RENDER_API_KEY")
        sys.exit(1)
    
    if not args.service_id:
        print(f"{RED}❌ Errore: Service ID non fornito{RESET}")
        print("Usa --service-id o imposta RENDER_SERVICE_ID")
        print("\nPer trovare il Service ID:")
        print("1. Vai su https://dashboard.render.com")
        print("2. Apri il tuo servizio")
        print("3. L'URL sarà: https://dashboard.render.com/web/.../services/YOUR_SERVICE_ID")
        sys.exit(1)
    
    monitor = RenderLogMonitor(args.api_key, args.service_id)
    monitor.monitor(interval=args.interval, filter_level=args.level, filter_text=args.filter)


if __name__ == "__main__":
    main()

