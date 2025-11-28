# Analisi Filtri Mancanti - Signal Quality Gate

## Situazione Attuale

### Filtri Esistenti in `_filter_obvious_opportunities` (Livello 1)
✅ **35+ filtri attivi** che coprono:
- Over/Under markets
- 1X2 markets
- BTTS markets
- Clean Sheet markets
- Exact Score markets
- Goal Range markets
- Next Goal markets
- Win Either Half markets
- Highest Scoring Half markets
- E altri...

### Signal Quality Gate (Livello 2 - Nuovo)
⚠️ **Filtri limitati** che coprono solo:
- Over/Under base (over_1.5, over_2.5, over_3.5, under_1.5, under_2.5)
- 1X2 base (home_win, away_win)
- Win Either Half (solo 0-0 dopo 60')
- Highest Scoring Half (dopo 50')
- Under 2.5 troppo presto (1 gol < 30')

---

## Casi Mancanti nel Signal Quality Gate

### 🔴 PRIORITÀ ALTA (Casi Critici)

1. **Over 0.5 quando c'è già 1+ gol**
   - Filtro esistente: ✅ Presente in `_filter_obvious_opportunities`
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: ALTO (banale)

2. **BTTS Yes quando entrambe hanno già segnato**
   - Filtro esistente: ✅ Presente (FILTRO 22B)
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: ALTO (banale)

3. **BTTS Yes quando una squadra non ha segnato e siamo oltre 85'**
   - Filtro esistente: ✅ Presente (FILTRO 22)
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: ALTO (illogico)

4. **Clean Sheet quando risultato è 2-0+ al 75'**
   - Filtro esistente: ✅ Presente (FILTRO 19)
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: ALTO (banale)

5. **Exact Score quando suggerisce lo score attuale al 70'+**
   - Filtro esistente: ✅ Presente (FILTRO 23)
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: ALTO (banale)

6. **Team to Score First quando NON è 0-0**
   - Filtro esistente: ✅ Presente (FILTRO 33)
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: ALTO (impossibile)

7. **Next Goal quando siamo oltre 85'**
   - Filtro esistente: ✅ Presente (FILTRO 28)
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: ALTO (banale)

8. **Goal Range 0-1 quando c'è già 1 gol al 60'+**
   - Filtro esistente: ✅ Presente (FILTRO 24)
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: MEDIO (illogico)

9. **Goal Range 2-3 quando ci sono già 4+ gol**
   - Filtro esistente: ✅ Presente (FILTRO 25)
   - Signal Quality Gate: ❌ MANCANTE
   - Impatto: MEDIO (illogico)

10. **Win To Nil quando è 2-0+ al 75'**
    - Filtro esistente: ✅ Presente (FILTRO 30)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: MEDIO (banale)

---

### 🟡 PRIORITÀ MEDIA (Casi Importanti)

11. **Over HT banali (Over 0.5 HT al 40'+ quando c'è già 1 gol)**
    - Filtro esistente: ✅ Presente (FILTRO 21)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: MEDIO

12. **Under HT banali (Under 0.5 HT al 42'+ quando è 0-0)**
    - Filtro esistente: ✅ Presente (FILTRO 20)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: MEDIO

13. **Odd/Even banali quando è troppo tardi (85'+)**
    - Filtro esistente: ✅ Presente (FILTRO 27)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: MEDIO

14. **Home/Away Goal Anytime quando hanno già segnato**
    - Filtro esistente: ✅ Presente (FILTRO 27B)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: MEDIO

15. **DNB quando partita è decisa (3+ gol diff al 70'+)**
    - Filtro esistente: ✅ Presente (FILTRO 32)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: MEDIO

16. **Team to Score Last quando partita decisa o 88'+**
    - Filtro esistente: ✅ Presente (FILTRO 34)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: MEDIO

17. **Second Half Over quando 80'+ e 2+ gol totali**
    - Filtro esistente: ✅ Presente (FILTRO 31)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: MEDIO

---

### 🟢 PRIORITÀ BASSA (Casi Secondari)

18. **BTTS No quando una squadra ha già segnato e siamo 80'+**
    - Filtro esistente: ✅ Presente (FILTRO 22C)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: BASSO

19. **BTTS Yes quando una squadra ha cartellino rosso**
    - Filtro esistente: ✅ Presente (FILTRO 22D)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: BASSO

20. **Next Goal Pressure quando 80'+ o partita decisa**
    - Filtro esistente: ✅ Presente (FILTRO 33C)
    - Signal Quality Gate: ❌ MANCANTE
    - Impatto: BASSO

---

## Raccomandazione

### Opzione A: Aggiungere tutti i filtri critici (Priorità Alta)
- **Tempo**: ~30 minuti
- **Beneficio**: Copertura completa casi critici
- **Rischio**: Basso

### Opzione B: Aggiungere filtri critici + media priorità
- **Tempo**: ~1 ora
- **Beneficio**: Copertura completa
- **Rischio**: Basso

### Opzione C: Aggiungere tutti i filtri
- **Tempo**: ~2 ore
- **Beneficio**: Massima copertura
- **Rischio**: Basso (sono già testati nei filtri esistenti)

---

## Casi Specifici dai Log

Dai log recenti, questi segnali sono passati ma dovevano essere bloccati:

1. **Under 2.5 al 15' con 1-0** ✅ GIÀ AGGIUNTO
2. **Highest Scoring Half al 57'** ✅ GIÀ AGGIUNTO
3. **Under 2.5 al 60' con 1-1** - Verificare se è banale

---

## Prossimi Passi

1. Implementare filtri Priorità Alta (10 casi)
2. Testare con casi reali
3. Aggiungere filtri Priorità Media se necessario
4. Monitorare e aggiustare


