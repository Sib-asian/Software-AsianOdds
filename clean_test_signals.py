#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('signal_quality_learning.db')
cursor = conn.cursor()
cursor.execute("DELETE FROM signal_records WHERE match_id LIKE 'test_%'")
deleted = cursor.rowcount
conn.commit()
conn.close()
print(f'✅ Eliminati {deleted} segnali di test dal database')

