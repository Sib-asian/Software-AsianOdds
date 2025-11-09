# Modello Scommesse – Streamlit

App Streamlit che prende le quote da The Odds API, le pulisce e le passa al modello di probabilità.

## Segreti da configurare su Streamlit Cloud

Nel menu "Settings" → "Secrets" aggiungi:

```toml
THE_ODDS_API_KEY = "la_tua_key_di_the_odds_api"
API_FOOTBALL_KEY = "la_tua_key_di_api_football"
