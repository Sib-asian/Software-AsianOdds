#!/usr/bin/env python3
"""
Analisi Soglie EV
=================

Analizza se la soglia EV del 5% è appropriata.
"""

print("=" * 70)
print("📊 ANALISI SOGLIE EV")
print("=" * 70)

print("\n💡 COSA SIGNIFICA EV (Expected Value)?")
print("-" * 70)
print("""
EV (Expected Value) = Profitto atteso medio su 100 scommesse

Esempi:
- EV 5% = Su 100 scommesse da €10, guadagni mediamente €50
- EV 10% = Su 100 scommesse da €10, guadagni mediamente €100
- EV -5% = Su 100 scommesse da €10, perdi mediamente €50

Formula: EV = (Probabilità * Quota) - 1
""")

print("\n📈 SOGLIE EV NEL BETTING PROFESSIONALE")
print("-" * 70)
print("""
• EV ≥ 10% = ECCELLENTE (molto raro, opportunità eccezionali)
• EV ≥ 5%  = BUONO (value bet solido, consigliato per professionisti)
• EV ≥ 3%  = ACCETTABILE (value bet decente, buono per principianti)
• EV ≥ 1%  = BASSO (marginale, solo per volumi alti)
• EV < 0%  = NEGATIVO (mai scommettere, perderesti nel lungo periodo)
""")

print("\n🎯 SOGLIA ATTUALE: 5%")
print("-" * 70)
print("""
✅ PRO:
• Filtra solo value bet di qualità
• Riduce il rischio di scommesse marginali
• Più profittevole nel lungo periodo
• Standard professionale

❌ CONTRO:
• Meno notifiche (potresti perdere opportunità)
• Richiede pazienza
• Potrebbe essere troppo conservativo per iniziare
""")

print("\n💭 RACCOMANDAZIONE")
print("-" * 70)
print("""
Per un sistema 24/7 che vuole trovare opportunità:

1. EV 5% = CONSERVATIVO (poche notifiche, alta qualità)
   → Buono se vuoi solo le migliori opportunità
   → Ideale per bankroll management serio

2. EV 3% = EQUILIBRATO (più notifiche, buona qualità)
   → Buon compromesso tra quantità e qualità
   → Consigliato per iniziare e vedere come funziona

3. EV 1-2% = AGRESSIVO (molte notifiche, qualità variabile)
   → Solo se vuoi vedere molte opportunità
   → Richiede più attenzione nella selezione

NOTA: Il problema attuale NON è la soglia EV, ma che le partite
analizzate hanno EV NEGATIVO (-44%, -37%, etc.). Questo significa
che le quote dei bookmaker sono peggiori della probabilità reale.
""")

print("\n🔍 PROBLEMA ATTUALE")
print("-" * 70)
print("""
Le partite analizzate hanno:
• EV: -44% (Belarus vs Greece)
• EV: -37% (Austria vs Bosnia)
• EV: -49% (Platense vs Gimnasia)

Questo significa che anche con EV 0% (soglia minima), queste partite
NON verrebbero notificate perché hanno EV negativo.

Il sistema sta funzionando correttamente - sta proteggendoti da
scommesse perdenti!
""")

print("\n💡 SUGGERIMENTO")
print("-" * 70)
print("""
1. Mantieni EV 5% per qualità (consigliato)
   → Notifiche solo per vere opportunità
   → Più profittevole nel lungo periodo

2. Oppure abbassa a EV 3% per più notifiche
   → Più opportunità da valutare
   → Buon compromesso

3. Il sistema continuerà a cercare - quando troverà una partita
   con EV positivo ≥ soglia, ti notificherà automaticamente!
""")

print("\n" + "=" * 70)
print("✅ CONCLUSIONE: EV 5% è CORRETTO e PROFESSIONALE")
print("=" * 70)
print("""
Il sistema funziona bene. Il problema non è la soglia, ma che
attualmente le quote disponibili non offrono value bet.

Quando il sistema troverà una vera opportunità (EV positivo),
ti notificherà automaticamente!
""")

