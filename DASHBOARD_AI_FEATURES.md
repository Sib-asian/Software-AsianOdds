# 🧠 Dashboard AI Features

## Nuove Funzionalità AI Implementate

### 1. **AI Insights & Raccomandazioni** 🧠

#### Tab "📊 Insights AI"
- **Genera Insights AI**: Analizza automaticamente i pattern di performance usando `PatternAnalyzerLLM`
- **Pattern Identificati**: Mostra pattern trovati nelle scommesse (per lega, market, orario, tipo)
- **Insights Chiave**: Suggerimenti intelligenti basati su analisi AI
- **Raccomandazioni**: Consigli strategici per migliorare performance
- **Summary**: Riepilogo dell'analisi

#### Tab "💡 Raccomandazioni"
- **Market Profittevole**: Identifica i market più redditizi
- **Market in Perdita**: Avvisa sui market che performano male
- **Lega Top Performer**: Indica le leghe più profittevoli
- **Trend Analysis**: Analizza trend positivo/negativo confrontando 7 giorni vs 30 giorni

#### Tab "⚙️ Ottimizzazione"
- **Ottimizzazione Parametri**: Usa AI per suggerire parametri ottimali (Min EV, Min Confidence)
- **Range Ottimizzazione**: Permette di definire range di ricerca
- **Suggerimenti Automatici**: Suggerimenti basati su performance attuale

### 2. **Predizioni e Trend AI** 🔮

#### Trend Analysis
- **Win Rate Trend**: Confronta win rate ultimi 7 giorni vs 30 giorni
- **ROI Trend**: Confronta ROI ultimi 7 giorni vs 30 giorni
- **Predizione Trend**: Indica se il trend è positivo, negativo o neutro

#### Market Intelligence
- **Top 3 Markets**: Mostra i 3 market più profittevoli
- **Bottom 3 Markets**: Mostra i 3 market meno profittevoli
- **Analisi Dettagliata**: Per ogni market mostra count, win rate e profit

### 3. **Funzioni Helper AI**

#### `_generate_basic_insights()`
Genera insights base anche senza AI disponibile:
- Analizza win rate e ROI
- Identifica trend positivi/negativi
- Fornisce feedback sulla performance

#### `_generate_basic_recommendations()`
Genera raccomandazioni base anche senza AI:
- Suggerimenti per ROI negativo
- Consigli per win rate basso
- Focus su market più profittevoli

## Come Usare

1. **Apri la Dashboard**: `streamlit run automation_dashboard.py`
2. **Vai alla sezione "🧠 AI Insights & Raccomandazioni"**
3. **Clicca "🔄 Genera Insights AI"** per analizzare i dati
4. **Esplora i 3 tab**: Insights, Raccomandazioni, Ottimizzazione
5. **Controlla "🔮 Predizioni e Trend AI"** per vedere trend e market intelligence

## Requisiti

- `PatternAnalyzerLLM` disponibile (se installato, usa AI avanzata)
- `ParameterOptimizer` disponibile (opzionale, per ottimizzazione)
- `IntelligentAlertSystem` disponibile (opzionale)

**Nota**: Se i sistemi AI non sono disponibili, la dashboard usa comunque funzioni base per generare insights e raccomandazioni.

## Vantaggi

✅ **Analisi Automatica**: Non devi analizzare manualmente i dati
✅ **Insights Intelligenti**: Pattern che non noteresti facilmente
✅ **Raccomandazioni Pratiche**: Suggerimenti concreti per migliorare
✅ **Trend Prediction**: Vedi dove sta andando la tua performance
✅ **Market Intelligence**: Focus sui market più profittevoli

