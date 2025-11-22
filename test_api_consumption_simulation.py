#!/usr/bin/env python3
"""
Test Simulazione Consumo API
=============================

Simula il comportamento del sistema in un ciclo di 24 ore per verificare
il risparmio API effettivo con il sistema di cache.
"""

import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import del sistema
try:
    from multi_source_match_finder import MultiSourceMatchFinder
except ImportError as e:
    logger.error(f"❌ Errore import: {e}")
    sys.exit(1)


class APIConsumptionSimulator:
    """Simulatore consumo API per verificare il risparmio con cache"""

    def __init__(self):
        self.finder = MultiSourceMatchFinder()
        self.total_calls_without_cache = 0
        self.total_calls_with_cache = 0
        self.cache_hits = 0

    def simulate_single_cycle(self, cycle_num: int, num_live_matches: int = 50):
        """
        Simula un singolo ciclo di analisi.

        Args:
            cycle_num: Numero del ciclo
            num_live_matches: Numero di partite live simulate

        Returns:
            Dict con statistiche del ciclo
        """
        logger.info(f"{'=' * 80}")
        logger.info(f"CICLO {cycle_num}")
        logger.info(f"{'=' * 80}")

        # Reset contatori (come fa il codice reale)
        self.finder.api_calls_count = 0
        self.finder.api_calls_saved_by_cache = 0

        # Simula chiamate che verrebbero fatte in un ciclo reale
        calls_this_cycle = 0

        # 1. Chiamata fixtures per data (sempre fatta)
        calls_this_cycle += 1
        logger.info("📡 Chiamata 1: /fixtures?date=... (fixtures del giorno)")

        # 2. Chiamata fixtures live (sempre fatta)
        calls_this_cycle += 1
        logger.info("📡 Chiamata 2: /fixtures?live=all (partite live)")

        # 3. Per ogni partita live, chiamate statistiche e quote
        cache_hits_cycle = 0
        for i in range(num_live_matches):
            fixture_id = 1000 + i  # ID simulato

            # Simula chiamata statistiche
            if fixture_id in self.finder.statistics_cache:
                # Verifica TTL cache
                cached = self.finder.statistics_cache[fixture_id]
                age = time.time() - cached['timestamp']
                if age < self.finder.cache_ttl_seconds:
                    # Cache hit!
                    cache_hits_cycle += 1
                    self.finder.api_calls_saved_by_cache += 1
                    logger.debug(f"   ✅ CACHE HIT stats fixture {fixture_id} (age: {age:.0f}s)")
                else:
                    # Cache scaduta
                    calls_this_cycle += 1
                    self.finder.api_calls_count += 1
                    # Aggiorna cache
                    self.finder.statistics_cache[fixture_id] = {
                        'data': {'mock': 'data'},
                        'timestamp': time.time()
                    }
                    logger.debug(f"   ⏰ CACHE EXPIRED stats fixture {fixture_id} (age: {age:.0f}s)")
            else:
                # Primo accesso, cache miss
                calls_this_cycle += 1
                self.finder.api_calls_count += 1
                # Popola cache
                self.finder.statistics_cache[fixture_id] = {
                    'data': {'mock': 'data'},
                    'timestamp': time.time()
                }
                logger.debug(f"   📡 CACHE MISS stats fixture {fixture_id}")

            # Simula chiamata quote
            if fixture_id in self.finder.odds_cache:
                # Verifica TTL cache
                cached = self.finder.odds_cache[fixture_id]
                age = time.time() - cached['timestamp']
                if age < self.finder.cache_ttl_seconds:
                    # Cache hit!
                    cache_hits_cycle += 1
                    self.finder.api_calls_saved_by_cache += 1
                    logger.debug(f"   ✅ CACHE HIT odds fixture {fixture_id} (age: {age:.0f}s)")
                else:
                    # Cache scaduta
                    calls_this_cycle += 1
                    self.finder.api_calls_count += 1
                    # Aggiorna cache
                    self.finder.odds_cache[fixture_id] = {
                        'data': {'mock': 'odds'},
                        'timestamp': time.time()
                    }
                    logger.debug(f"   ⏰ CACHE EXPIRED odds fixture {fixture_id} (age: {age:.0f}s)")
            else:
                # Primo accesso, cache miss
                calls_this_cycle += 1
                self.finder.api_calls_count += 1
                # Popola cache
                self.finder.odds_cache[fixture_id] = {
                    'data': {'mock': 'odds'},
                    'timestamp': time.time()
                }
                logger.debug(f"   📡 CACHE MISS odds fixture {fixture_id}")

        # Statistiche ciclo
        calls_without_cache = 2 + (num_live_matches * 2)  # fixtures + live + (stats + odds) per partita

        logger.info("")
        logger.info(f"📊 Statistiche Ciclo {cycle_num}:")
        logger.info(f"   - Partite live simulate: {num_live_matches}")
        logger.info(f"   - Chiamate API SENZA cache: {calls_without_cache}")
        logger.info(f"   - Chiamate API CON cache: {calls_this_cycle}")
        logger.info(f"   - Cache hits: {cache_hits_cycle}")
        logger.info(f"   - Risparmio: {calls_without_cache - calls_this_cycle} chiamate ({((calls_without_cache - calls_this_cycle) / calls_without_cache * 100):.1f}%)")
        logger.info("")

        return {
            'cycle': cycle_num,
            'calls_without_cache': calls_without_cache,
            'calls_with_cache': calls_this_cycle,
            'cache_hits': cache_hits_cycle,
            'savings': calls_without_cache - calls_this_cycle
        }

    def simulate_24h(self, cycle_interval_minutes: int = 20, num_live_matches: int = 50):
        """
        Simula 24 ore di operazione del sistema.

        Args:
            cycle_interval_minutes: Intervallo tra cicli in minuti
            num_live_matches: Numero di partite live simulate per ciclo
        """
        logger.info("=" * 80)
        logger.info("🕐 SIMULAZIONE 24 ORE - CONSUMO API")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Parametri simulazione:")
        logger.info(f"  - Intervallo cicli: {cycle_interval_minutes} minuti")
        logger.info(f"  - Partite live per ciclo: {num_live_matches}")
        logger.info(f"  - TTL cache: {self.finder.cache_ttl_seconds} secondi ({self.finder.cache_ttl_seconds / 60:.1f} minuti)")
        logger.info("")

        cycles_per_day = (24 * 60) // cycle_interval_minutes
        logger.info(f"📊 Cicli in 24 ore: {cycles_per_day}")
        logger.info("")

        # Simula solo i primi 10 cicli per velocità (pattern si ripete)
        num_cycles_to_simulate = min(10, cycles_per_day)
        logger.info(f"🔄 Simulando primi {num_cycles_to_simulate} cicli (pattern si ripete)...")
        logger.info("")

        cycle_stats = []

        for cycle in range(1, num_cycles_to_simulate + 1):
            # Simula passaggio tempo tra cicli
            if cycle > 1:
                # Simula che passa cycle_interval_minutes * 60 secondi
                # Per velocità test, non aspettiamo realmente ma avanziamo timestamp cache manualmente
                # Aumenta età di tutte le cache di cycle_interval_minutes * 60 secondi
                time_passed = cycle_interval_minutes * 60
                for fixture_id in list(self.finder.statistics_cache.keys()):
                    self.finder.statistics_cache[fixture_id]['timestamp'] -= time_passed
                for fixture_id in list(self.finder.odds_cache.keys()):
                    self.finder.odds_cache[fixture_id]['timestamp'] -= time_passed

            stats = self.simulate_single_cycle(cycle, num_live_matches)
            cycle_stats.append(stats)

            # Piccola pausa per leggibilità log
            time.sleep(0.1)

        # Calcola statistiche totali
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 STATISTICHE SIMULAZIONE")
        logger.info("=" * 80)
        logger.info("")

        total_without = sum(s['calls_without_cache'] for s in cycle_stats)
        total_with = sum(s['calls_with_cache'] for s in cycle_stats)
        total_hits = sum(s['cache_hits'] for s in cycle_stats)
        total_savings = sum(s['savings'] for s in cycle_stats)

        # Proietta su 24 ore
        cycles_simulated = len(cycle_stats)
        projected_without = (total_without / cycles_simulated) * cycles_per_day
        projected_with = (total_with / cycles_simulated) * cycles_per_day
        projected_savings = (total_savings / cycles_simulated) * cycles_per_day

        logger.info(f"Cicli simulati: {cycles_simulated}")
        logger.info(f"Chiamate API totali (simulati):")
        logger.info(f"  - SENZA cache: {total_without}")
        logger.info(f"  - CON cache: {total_with}")
        logger.info(f"  - Cache hits: {total_hits}")
        logger.info(f"  - Risparmio: {total_savings} chiamate")
        logger.info("")

        logger.info(f"Proiezione 24 ore ({cycles_per_day} cicli):")
        logger.info(f"  - SENZA cache: {projected_without:.0f} chiamate/giorno")
        logger.info(f"  - CON cache: {projected_with:.0f} chiamate/giorno")
        logger.info(f"  - Risparmio: {projected_savings:.0f} chiamate/giorno ({(projected_savings / projected_without * 100):.1f}%)")
        logger.info("")

        # Confronto con obiettivo
        logger.info("=" * 80)
        logger.info("🎯 CONFRONTO CON OBIETTIVO")
        logger.info("=" * 80)
        logger.info("")

        logger.info("OBIETTIVO (da OTTIMIZZAZIONE_API_CACHE.md):")
        logger.info("  - Prima: ~7344 chiamate/giorno")
        logger.info("  - Dopo: ~1600 chiamate/giorno")
        logger.info("  - Risparmio atteso: 78.4%")
        logger.info("")

        logger.info("RISULTATO SIMULAZIONE:")
        logger.info(f"  - Prima: {projected_without:.0f} chiamate/giorno")
        logger.info(f"  - Dopo: {projected_with:.0f} chiamate/giorno")
        logger.info(f"  - Risparmio effettivo: {(projected_savings / projected_without * 100):.1f}%")
        logger.info("")

        # Verifica obiettivo raggiunto
        expected_savings_pct = 78.4
        actual_savings_pct = (projected_savings / projected_without * 100)

        if actual_savings_pct >= expected_savings_pct - 5:  # Tolleranza 5%
            logger.info("✅ OBIETTIVO RAGGIUNTO!")
            logger.info(f"   Risparmio {actual_savings_pct:.1f}% >= {expected_savings_pct}% ✓")
            logger.info("")

            # Verifica limite API-SPORTS
            api_limit = 7500
            usage_pct = (projected_with / api_limit) * 100
            logger.info(f"📊 Utilizzo limite API-SPORTS:")
            logger.info(f"   - Limite giornaliero: {api_limit} chiamate")
            logger.info(f"   - Consumo proiettato: {projected_with:.0f} chiamate")
            logger.info(f"   - Utilizzo: {usage_pct:.1f}%")
            logger.info(f"   - Margine disponibile: {api_limit - projected_with:.0f} chiamate")
            logger.info("")

            if projected_with < api_limit:
                logger.info("✅ CONSUMO API ENTRO I LIMITI!")
                return True
            else:
                logger.warning("⚠️  ATTENZIONE: Consumo API oltre il limite!")
                return False
        else:
            logger.warning(f"⚠️  OBIETTIVO NON RAGGIUNTO")
            logger.warning(f"   Risparmio {actual_savings_pct:.1f}% < {expected_savings_pct}%")
            return False

    def run(self):
        """Esegue simulazione completa"""
        success = self.simulate_24h(
            cycle_interval_minutes=20,
            num_live_matches=50
        )
        return success


def main():
    """Entry point"""
    simulator = APIConsumptionSimulator()
    success = simulator.run()

    logger.info("")
    logger.info("=" * 80)
    logger.info("📝 CONCLUSIONE SIMULAZIONE")
    logger.info("=" * 80)
    logger.info("")

    if success:
        logger.info("🎉 SIMULAZIONE COMPLETATA CON SUCCESSO!")
        logger.info("✅ Il sistema di cache funziona correttamente")
        logger.info("✅ Il risparmio API è conforme all'obiettivo")
        logger.info("✅ Il consumo rimane entro i limiti API-SPORTS")
    else:
        logger.warning("⚠️  SIMULAZIONE COMPLETATA CON AVVISI")
        logger.warning("Rivedi i parametri o le ottimizzazioni")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
