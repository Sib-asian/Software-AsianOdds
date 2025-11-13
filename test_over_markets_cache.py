"""
Test suite per mercati over e sistema di caching

Testa:
1. Sistema di caching per mercati over
2. Validazione dati cached
3. Gestione scadenza cache
4. Test integrazione
"""

import unittest
import tempfile
import os
import time
from datetime import datetime

# Import solo CacheManager senza dipendenze pesanti
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_manager import CacheManager


class TestOverMarketsCache(unittest.TestCase):
    """Test per il sistema di caching dei mercati over"""

    def setUp(self):
        """Setup: crea un database temporaneo per i test"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.cache = CacheManager(db_path=self.temp_db.name)

    def tearDown(self):
        """Cleanup: rimuove il database temporaneo"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    def test_cache_set_and_get(self):
        """Test set e get della cache"""
        market_data = {
            "over_05_ht": 0.65,
            "over_15_ht": 0.35,
            "over_05ht_over_05ft": 0.58,
            "over_05ht_over_15ft": 0.45,
            "over_05ht_over_25ft": 0.32
        }

        # Set cache
        self.cache.set_over_markets("Inter", "Milan", "2024-12-20", market_data)

        # Get cache
        cached_data = self.cache.get_over_markets("Inter", "Milan", "2024-12-20")

        self.assertIsNotNone(cached_data)
        self.assertEqual(cached_data["over_05_ht"], 0.65)
        self.assertEqual(cached_data["over_15_ht"], 0.35)
        self.assertEqual(len(cached_data), 5)

    def test_cache_miss(self):
        """Test cache miss (dati non presenti)"""
        cached_data = self.cache.get_over_markets("NonExistent", "Team", "2024-01-01")
        self.assertIsNone(cached_data)

    def test_cache_expiration(self):
        """Test scadenza cache (TTL)"""
        market_data = {"over_05_ht": 0.65}

        # Set cache con timestamp vecchio (simuliamo scadenza)
        import sqlite3
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        old_timestamp = int(time.time()) - 86500  # 24h + 100s fa
        cursor.execute("""
            INSERT OR REPLACE INTO over_markets_cache (home_team, away_team, match_date, market_data, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, ("inter", "milan", "2024-12-20", '{"over_05_ht": 0.65}', old_timestamp))
        conn.commit()
        conn.close()

        # Get cache (dovrebbe essere None perché scaduto)
        cached_data = self.cache.get_over_markets("Inter", "Milan", "2024-12-20")
        self.assertIsNone(cached_data)

    def test_cache_case_insensitive(self):
        """Test cache case insensitive"""
        market_data = {"over_05_ht": 0.65}

        # Set con maiuscole
        self.cache.set_over_markets("INTER", "MILAN", "2024-12-20", market_data)

        # Get con minuscole
        cached_data = self.cache.get_over_markets("inter", "milan", "2024-12-20")
        self.assertIsNotNone(cached_data)
        self.assertEqual(cached_data["over_05_ht"], 0.65)

        # Get con maiuscole/minuscole miste
        cached_data = self.cache.get_over_markets("InTeR", "MiLaN", "2024-12-20")
        self.assertIsNotNone(cached_data)

    def test_cache_update(self):
        """Test aggiornamento cache esistente"""
        market_data_v1 = {"over_05_ht": 0.65}
        market_data_v2 = {"over_05_ht": 0.70, "over_15_ht": 0.40}

        # Prima scrittura
        self.cache.set_over_markets("Inter", "Milan", "2024-12-20", market_data_v1)
        cached = self.cache.get_over_markets("Inter", "Milan", "2024-12-20")
        self.assertEqual(len(cached), 1)

        # Seconda scrittura (update)
        self.cache.set_over_markets("Inter", "Milan", "2024-12-20", market_data_v2)
        cached = self.cache.get_over_markets("Inter", "Milan", "2024-12-20")
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached["over_05_ht"], 0.70)
        self.assertEqual(cached["over_15_ht"], 0.40)

    def test_cache_multiple_matches(self):
        """Test cache per multiple partite"""
        matches = [
            ("Inter", "Milan", "2024-12-20", {"over_05_ht": 0.65}),
            ("Juventus", "Roma", "2024-12-21", {"over_05_ht": 0.58}),
            ("Napoli", "Lazio", "2024-12-22", {"over_05_ht": 0.72})
        ]

        # Set multiple cache entries
        for home, away, date, data in matches:
            self.cache.set_over_markets(home, away, date, data)

        # Get e verifica tutte le entries
        for home, away, date, expected_data in matches:
            cached = self.cache.get_over_markets(home, away, date)
            self.assertIsNotNone(cached)
            self.assertEqual(cached["over_05_ht"], expected_data["over_05_ht"])


def run_tests():
    """Esegue tutti i test"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test class
    suite.addTests(loader.loadTestsFromTestCase(TestOverMarketsCache))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
