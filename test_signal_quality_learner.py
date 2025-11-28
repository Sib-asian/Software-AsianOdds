"""
Test: Sistema di Apprendimento Signal Quality Gate
===================================================

Verifica che:
1. Il learner traccia correttamente i segnali
2. Aggiorna risultati correttamente
3. Apprende dai risultati e aggiorna parametri
4. I parametri appresi vengono applicati correttamente
"""

import unittest
import logging
import os
import sqlite3
from datetime import datetime
from unittest.mock import Mock
import sys

# Aggiungi il path del progetto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_system.signal_quality_learner import SignalQualityLearner

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestSignalQualityLearner(unittest.TestCase):
    
    def setUp(self):
        """Setup test con database temporaneo"""
        self.test_db = "test_signal_quality_learning.db"
        # Rimuovi database esistente se presente
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        
        self.learner = SignalQualityLearner(db_path=self.test_db)
    
    def tearDown(self):
        """Cleanup database temporaneo"""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_record_signal(self):
        """Test: Registrazione segnale"""
        record_id = self.learner.record_signal(
            match_id="test_match_1",
            market="over_2.5",
            minute=30,
            score_home=1,
            score_away=0,
            quality_score=80.0,
            context_score=85.0,
            data_quality_score=75.0,
            logic_score=80.0,
            timing_score=75.0,
            was_approved=True,
            block_reasons=[],
            confidence=82.0,
            ev=12.0
        )
        
        self.assertIsNotNone(record_id)
        logger.info(f"✅ Test 1: Segnale registrato con ID {record_id}")
    
    def test_update_signal_result(self):
        """Test: Aggiornamento risultato"""
        # Registra segnale
        record_id = self.learner.record_signal(
            match_id="test_match_2",
            market="over_2.5",
            minute=30,
            score_home=1,
            score_away=0,
            quality_score=80.0,
            context_score=85.0,
            data_quality_score=75.0,
            logic_score=80.0,
            timing_score=75.0,
            was_approved=True,
            block_reasons=[],
            confidence=82.0,
            ev=12.0
        )
        
        # Aggiorna risultato (finale 2-1 = 3 gol, over_2.5 vince)
        updated_count = self.learner.update_signal_result(
            match_id="test_match_2",
            final_score_home=2,
            final_score_away=1,
            market="over_2.5"
        )
        
        self.assertEqual(updated_count, 1)
        
        # Verifica risultato
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT was_correct, outcome FROM signal_records WHERE id = ?", (record_id,))
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 1)  # was_correct = True
        self.assertEqual(result[1], "win")  # outcome = win
        logger.info(f"✅ Test 2: Risultato aggiornato correttamente (was_correct={result[0]}, outcome={result[1]})")
    
    def test_learn_from_results_insufficient_samples(self):
        """Test: Apprendimento con campioni insufficienti"""
        # Registra solo 10 segnali (minimo è 50)
        for i in range(10):
            self.learner.record_signal(
                match_id=f"test_match_{i}",
                market="over_2.5",
                minute=30,
                score_home=1,
                score_away=0,
                quality_score=80.0,
                context_score=85.0,
                data_quality_score=75.0,
                logic_score=80.0,
                timing_score=75.0,
                was_approved=True,
                block_reasons=[],
                confidence=82.0,
                ev=12.0
            )
            # Aggiorna risultato
            self.learner.update_signal_result(
                match_id=f"test_match_{i}",
                final_score_home=2,
                final_score_away=1
            )
        
        results = self.learner.learn_from_results(min_samples=50)
        
        self.assertEqual(results['status'], 'insufficient_samples')
        self.assertEqual(results['samples'], 10)
        logger.info(f"✅ Test 3: Apprendimento con campioni insufficienti gestito correttamente")
    
    def test_learn_from_results_sufficient_samples(self):
        """Test: Apprendimento con campioni sufficienti"""
        # Registra 60 segnali (sufficienti per apprendere)
        for i in range(60):
            was_approved = i % 2 == 0  # Alterna approvati/bloccati
            was_correct = i % 3 != 0  # 2/3 corretti
            
            self.learner.record_signal(
                match_id=f"test_match_{i}",
                market="over_2.5",
                minute=30,
                score_home=1,
                score_away=0,
                quality_score=80.0 if was_approved else 70.0,
                context_score=85.0 if was_correct else 70.0,  # Context più alto per corretti
                data_quality_score=75.0,
                logic_score=80.0,
                timing_score=75.0,
                was_approved=was_approved,
                block_reasons=[] if was_approved else ["Test block"],
                confidence=82.0,
                ev=12.0
            )
            
            # Aggiorna risultato (over_2.5 vince se was_correct)
            final_home = 2 if was_correct else 1
            final_away = 1 if was_correct else 0
            self.learner.update_signal_result(
                match_id=f"test_match_{i}",
                final_score_home=final_home,
                final_score_away=final_away
            )
        
        results = self.learner.learn_from_results(min_samples=50)
        
        self.assertEqual(results['status'], 'success')
        self.assertGreater(results['samples'], 50)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('accuracy', results)
        self.assertIn('weights', results)
        self.assertIn('min_quality_score', results)
        
        logger.info(
            f"✅ Test 4: Apprendimento completato - "
            f"Precision={results['precision']:.2%}, "
            f"Recall={results['recall']:.2%}, "
            f"Accuracy={results['accuracy']:.2%}"
        )
        logger.info(
            f"   Nuovi pesi: Context={results['weights']['context']:.2%}, "
            f"Data={results['weights']['data_quality']:.2%}, "
            f"Logic={results['weights']['logic']:.2%}, "
            f"Timing={results['weights']['timing']:.2%}"
        )
        logger.info(f"   Nuova soglia: {results['min_quality_score']:.1f}")
    
    def test_learned_parameters_persistence(self):
        """Test: Persistenza parametri appresi"""
        # Registra e apprendi
        for i in range(60):
            self.learner.record_signal(
                match_id=f"test_match_{i}",
                market="over_2.5",
                minute=30,
                score_home=1,
                score_away=0,
                quality_score=80.0,
                context_score=85.0,
                data_quality_score=75.0,
                logic_score=80.0,
                timing_score=75.0,
                was_approved=True,
                block_reasons=[],
                confidence=82.0,
                ev=12.0
            )
            self.learner.update_signal_result(
                match_id=f"test_match_{i}",
                final_score_home=2,
                final_score_away=1
            )
        
        results = self.learner.learn_from_results(min_samples=50)
        learned_weights = self.learner.get_learned_weights()
        learned_min_score = self.learner.get_learned_min_score()
        
        # Crea nuovo learner (simula riavvio)
        new_learner = SignalQualityLearner(db_path=self.test_db)
        new_weights = new_learner.get_learned_weights()
        new_min_score = new_learner.get_learned_min_score()
        
        # Verifica che i parametri siano stati caricati
        self.assertAlmostEqual(learned_weights['context'], new_weights['context'], places=2)
        self.assertAlmostEqual(learned_min_score, new_min_score, places=1)
        logger.info(f"✅ Test 5: Parametri appresi persistiti correttamente")


if __name__ == '__main__':
    print("=" * 60)
    print("Test: Sistema di Apprendimento Signal Quality Gate")
    print("=" * 60)
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)


