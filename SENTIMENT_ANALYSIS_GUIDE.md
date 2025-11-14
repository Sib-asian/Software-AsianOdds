# 🎯 Sentiment Analysis - Guida Professionale

## ✅ Integrazione Completata

Il sistema ora include **sentiment analysis professionale GRATUITA** usando:
- **Hugging Face InferenceClient** - API gratuita per analisi sentiment
- **Google News RSS** - Feed news gratuiti
- **Twitter/Reddit API** - Opzionali, per social media

---

## 🚀 Quick Start

### 1. Installa le dipendenze

```bash
pip install -r requirements.txt
```

Questo installerà:
- `huggingface_hub` - Per sentiment analysis (GRATUITO)
- `feedparser` - Per Google News
- `tweepy` - Per Twitter (opzionale)
- `praw` - Per Reddit (opzionale)

### 2. (Opzionale) Configura API Key Hugging Face

**Senza API key**: Funziona perfettamente, ma con rate limits più bassi.

**Con API key gratuita**: Rate limits più alti.

#### Come ottenere la chiave GRATUITA:

1. Crea account su https://huggingface.co/join
2. Vai su https://huggingface.co/settings/tokens
3. Clicca "New token" → Type: Read
4. Copia il token

#### Aggiungi al file `.env`:

```bash
# Crea file .env nella root del progetto
cp .env.example .env

# Modifica .env e aggiungi:
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxx
```

### 3. Usa il sentiment analyzer

Il sentiment è già **automaticamente integrato** nel sistema!

```python
from ai_system.sentiment_analyzer import SentimentAnalyzer

# Inizializza
analyzer = SentimentAnalyzer()

# Analizza match
result = analyzer.analyze_match_sentiment(
    team_home="Inter",
    team_away="Napoli",
    hours_before=48,
    include_sources=['news']  # news, twitter, reddit
)

# Risultati
print(f"Sentiment casa: {result['overall_sentiment_home']}")
print(f"Sentiment trasferta: {result['overall_sentiment_away']}")
print(f"Segnali rilevati: {len(result['signals'])}")
```

---

## 📊 Come Funziona

### Modello Utilizzato

**Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Ottimizzato per social media e news sportive
- Accuratezza: ~90% su testi sportivi
- Lingua: Inglese (funziona bene anche con testi misti IT/EN)
- Rate: Gratuito, ~100 richieste/ora senza key, ~1000/ora con key

### Fonti Analizzate

1. **Google News** (FREE, sempre attivo)
   - Feed RSS gratuiti
   - Articoli recenti su entrambe le squadre
   - Sentiment da titoli e descrizioni

2. **Twitter** (opzionale, richiede API key)
   - Tweet recenti con menzioni delle squadre
   - Keyword: injury, lineup, rumors
   - Engagement-weighted credibility

3. **Reddit** (opzionale, richiede API key)
   - r/soccer + team subreddits
   - Post e commenti recenti
   - Morale e fan sentiment

### Metriche Prodotte

- **Injury Risk** (0-100%): Probabilità infortuni non annunciati
- **Team Morale** (-100 a +100): Morale generale della squadra
- **Media Pressure** (0-100): Pressione mediatica
- **Fan Confidence** (0-100): Fiducia dei tifosi
- **Overall Sentiment** (-100 a +100): Sentiment complessivo

### Aggiustamento Predizioni

Il sentiment viene automaticamente usato per aggiustare le probabilità:

- **Alto rischio infortuni** (>60%): -10% probabilità vittoria
- **Morale positivo** (>30): +3% probabilità
- **Morale negativo** (<-30): -3% probabilità
- **Sentiment generale**: ±4% max adjustment

---

## 🔧 Configurazione Avanzata

### Configurare Twitter API (opzionale)

1. Vai su https://developer.twitter.com/en/portal/dashboard
2. Crea un'app → Genera Bearer Token
3. Aggiungi a `.env`:

```bash
TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAxxxxxxxxx
```

### Configurare Reddit API (opzionale)

1. Vai su https://www.reddit.com/prefs/apps
2. Crea un'app → Copia client_id e client_secret
3. Aggiungi a `.env`:

```bash
REDDIT_CLIENT_ID=xxxxxxxxxxxxx
REDDIT_CLIENT_SECRET=xxxxxxxxxxxxx
REDDIT_USER_AGENT=AsianOdds/1.0
```

---

## 💡 Best Practices

### Per Analisi Professionali

1. **Usa Hugging Face con API key** - È gratuita e dà rate limits migliori
2. **Analizza 48h prima del match** - Periodo ottimale per insider info
3. **Combina più fonti** - News + Social per migliore accuratezza
4. **Monitora segnali ad alta credibilità** - Filtra rumors poco affidabili

### Rate Limits

**Senza API key**:
- ~100 richieste/ora Hugging Face
- Illimitate per Google News

**Con API key gratuita**:
- ~1000 richieste/ora Hugging Face
- Illimitate per Google News

**Con Twitter API**:
- Free tier: 500k tweets/mese
- Basic tier ($100/mese): 10M tweets/mese

**Con Reddit API**:
- Gratuita: 60 richieste/minuto

---

## 🧪 Testing

### Test rapido senza dipendenze:

```bash
python3 test_hf_simple.py
```

### Test completo con tutte le features:

```bash
python3 test_sentiment_api.py
```

---

## 📈 Integrazione nel Sistema

Il sentiment è **già integrato** in:

1. **Pipeline AI** (`ai_system/pipeline.py`)
2. **Frontend** (`Frontendcloud.py`)
3. **Live Betting** (`ai_system/live_betting.py`)
4. **Live Monitor** (`ai_system/live_monitor.py`)

Non serve fare nulla, il sentiment verrà automaticamente calcolato e usato!

---

## 🆘 Troubleshooting

### "huggingface_hub not installed"

```bash
pip install huggingface_hub
```

### "feedparser not installed"

```bash
pip install feedparser
```

### "Model loading..."

Primo utilizzo: il modello Hugging Face si carica in ~20 secondi.
Riprova dopo 30 secondi.

### Rate limit exceeded

**Soluzione**: Aggiungi API key gratuita Hugging Face (vedi sopra).

---

## 📞 Supporto

- **Hugging Face Docs**: https://huggingface.co/docs/huggingface_hub
- **Model Card**: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
- **Issues**: Apri un issue su GitHub

---

## ✨ Features Future

- [ ] Supporto per più lingue (IT, ES, DE, FR)
- [ ] Cache sentiment per ridurre API calls
- [ ] Alert real-time per injury rumors ad alta credibilità
- [ ] Dashboard sentiment trends nel tempo

---

**🎉 Pronto per analisi professionali con sentiment analysis GRATUITO!**
