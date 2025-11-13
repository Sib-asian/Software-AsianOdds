"""
LLM Sports Analyst
==================

Chat AI intelligente che spiega predizioni in linguaggio naturale.

Funzionalità:
- Spiega perché una scommessa è consigliata/sconsigliata
- Risponde a domande su analisi e strategie
- Confronta partite e mercati
- Fornisce insights e raccomandazioni

Supporta:
- OpenAI GPT-4/GPT-3.5
- Anthropic Claude
- Local LLMs (Ollama)
"""

import logging
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)


class LLMAnalyst:
    """
    LLM-powered analyst per spiegazioni e insights.

    Usage:
        analyst = LLMAnalyst(api_key="your-key", provider="openai")
        explanation = analyst.explain_prediction(match_data, analysis_result)
        answer = analyst.answer_question("Perché questa bet?", context)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "mock",  # "openai", "anthropic", "mock"
        model: str = "gpt-4"
    ):
        """
        Inizializza LLM Analyst.

        Args:
            api_key: API key (not needed for mock)
            provider: "openai", "anthropic", "mock"
            model: Model name (e.g., "gpt-4", "claude-3-opus")
        """
        self.api_key = api_key
        self.provider = provider
        self.model = model

        # Initialize client
        self.client = None
        if provider == "openai" and api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"✅ OpenAI client initialized (model: {model})")
            except ImportError:
                logger.warning("⚠️  openai package not installed. Install with: pip install openai")
                self.provider = "mock"
        elif provider == "anthropic" and api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"✅ Anthropic client initialized (model: {model})")
            except ImportError:
                logger.warning("⚠️  anthropic package not installed. Install with: pip install anthropic")
                self.provider = "mock"
        else:
            logger.info("🤖 Using mock LLM (rule-based responses)")
            self.provider = "mock"

    def explain_prediction(
        self,
        match_data: Dict,
        analysis_result: Dict,
        language: str = "it"
    ) -> str:
        """
        Spiega una predizione in linguaggio naturale.

        Args:
            match_data: Dati match
            analysis_result: Risultato completo analisi pipeline
            language: "it" o "en"

        Returns:
            Spiegazione testuale dettagliata
        """
        # Build context for LLM
        context = self._build_context(match_data, analysis_result)

        # Build prompt
        prompt = self._build_explanation_prompt(context, language)

        # Get response
        if self.provider == "mock":
            response = self._mock_explanation(context, language)
        else:
            response = self._call_llm(prompt)

        return response

    def answer_question(
        self,
        question: str,
        match_data: Dict,
        analysis_result: Dict,
        language: str = "it"
    ) -> str:
        """
        Risponde a domanda specifica su analisi.

        Args:
            question: Domanda utente
            match_data: Dati match
            analysis_result: Risultato analisi
            language: "it" o "en"

        Returns:
            Risposta alla domanda
        """
        context = self._build_context(match_data, analysis_result)

        prompt = f"""
Sei un esperto analista di scommesse sportive. Rispondi alla seguente domanda in modo chiaro e conciso.

CONTESTO:
{self._format_context(context)}

DOMANDA UTENTE:
{question}

Fornisci una risposta professionale e dettagliata in italiano.
"""

        if self.provider == "mock":
            response = self._mock_answer(question, context, language)
        else:
            response = self._call_llm(prompt)

        return response

    def compare_matches(
        self,
        match1_data: Dict,
        match1_analysis: Dict,
        match2_data: Dict,
        match2_analysis: Dict,
        language: str = "it"
    ) -> str:
        """
        Confronta due partite e relative analisi.

        Returns:
            Confronto dettagliato
        """
        context1 = self._build_context(match1_data, match1_analysis)
        context2 = self._build_context(match2_data, match2_analysis)

        prompt = f"""
Confronta queste due opportunità di scommessa e raccomanda quale sia migliore.

PARTITA 1:
{self._format_context(context1)}

PARTITA 2:
{self._format_context(context2)}

Fornisci un confronto dettagliato considerando:
- Expected Value
- Confidence
- Risk/Reward
- Fattori qualitativi

Concludi con una raccomandazione chiara.
"""

        if self.provider == "mock":
            response = self._mock_comparison(context1, context2, language)
        else:
            response = self._call_llm(prompt)

        return response

    def suggest_strategy(
        self,
        recent_analyses: List[Dict],
        bankroll: float,
        language: str = "it"
    ) -> str:
        """
        Suggerisce strategia basandosi su analisi recenti.

        Args:
            recent_analyses: Lista ultime analisi
            bankroll: Bankroll attuale
            language: "it" o "en"

        Returns:
            Suggerimenti strategici
        """
        summary = self._summarize_recent_analyses(recent_analyses)

        prompt = f"""
Sei un consulente di betting strategy. Analizza le performance recenti e suggerisci miglioramenti.

ANALISI RECENTI:
{summary}

BANKROLL ATTUALE: €{bankroll:.2f}

Fornisci suggerimenti su:
1. Quali mercati stanno performando meglio
2. Se aumentare/ridurre stake size
3. Eventuali pattern da sfruttare
4. Raccomandazioni per migliorare ROI

Rispondi in italiano in modo professionale.
"""

        if self.provider == "mock":
            response = self._mock_strategy(summary, bankroll, language)
        else:
            response = self._call_llm(prompt)

        return response

    # ========================================
    # INTERNAL METHODS
    # ========================================

    def _build_context(self, match_data: Dict, analysis_result: Dict) -> Dict:
        """Estrae info chiave per LLM"""
        context = {
            'match': f"{match_data.get('home')} vs {match_data.get('away')}",
            'league': match_data.get('league', 'N/A'),
            'probability': analysis_result['summary']['probability'],
            'confidence': analysis_result['summary']['confidence'],
            'value_score': analysis_result['summary']['value_score'],
            'expected_value': analysis_result['summary']['expected_value'],
            'decision': analysis_result['final_decision']['action'],
            'stake': analysis_result['final_decision']['stake'],
            'odds': analysis_result['summary']['odds'],
        }

        # Add ensemble info if available
        if analysis_result.get('ensemble'):
            ensemble = analysis_result['ensemble']
            context['ensemble'] = {
                'models': ensemble['model_predictions'],
                'weights': ensemble['model_weights'],
                'uncertainty': ensemble['uncertainty']
            }

        # Add risk info
        context['red_flags'] = analysis_result['risk_decision'].get('red_flags', [])
        context['green_flags'] = analysis_result['risk_decision'].get('green_flags', [])

        # Add API context if available
        if analysis_result.get('api_context'):
            api_ctx = analysis_result['api_context']
            context['data_quality'] = api_ctx['metadata']['data_quality']

            # Form
            home_form = api_ctx.get('home_context', {}).get('data', {}).get('form', 'N/A')
            away_form = api_ctx.get('away_context', {}).get('data', {}).get('form', 'N/A')
            context['home_form'] = home_form
            context['away_form'] = away_form

            # Injuries
            home_inj = api_ctx.get('home_context', {}).get('data', {}).get('injuries', [])
            away_inj = api_ctx.get('away_context', {}).get('data', {}).get('injuries', [])
            context['home_injuries'] = len(home_inj)
            context['away_injuries'] = len(away_inj)

        return context

    def _format_context(self, context: Dict) -> str:
        """Formatta context per LLM prompt"""
        lines = []
        lines.append(f"Partita: {context['match']}")
        lines.append(f"Lega: {context['league']}")
        lines.append(f"Probabilità: {context['probability']:.1%}")
        lines.append(f"Confidence: {context['confidence']:.0f}/100")
        lines.append(f"Value Score: {context['value_score']:.0f}/100")
        lines.append(f"Expected Value: {context['expected_value']:+.1%}")
        lines.append(f"Decisione: {context['decision']}")
        lines.append(f"Stake: €{context['stake']:.2f}")
        lines.append(f"Quote: {context['odds']:.2f}")

        if 'ensemble' in context:
            lines.append("\nEnsemble Models:")
            for model, pred in context['ensemble']['models'].items():
                weight = context['ensemble']['weights'][model]
                lines.append(f"  {model}: {pred:.1%} (peso: {weight:.1%})")

        if context.get('home_form'):
            lines.append(f"\nForm Casa: {context['home_form']}")
            lines.append(f"Form Trasferta: {context['away_form']}")

        if context.get('red_flags'):
            lines.append(f"\nRed Flags: {len(context['red_flags'])}")
            for flag in context['red_flags'][:3]:
                lines.append(f"  - {flag}")

        return "\n".join(lines)

    def _build_explanation_prompt(self, context: Dict, language: str) -> str:
        """Build prompt for explanation"""
        lang_text = "italiano" if language == "it" else "English"

        return f"""
Sei un esperto analista di scommesse sportive. Spiega questa predizione in modo chiaro e professionale.

{self._format_context(context)}

Fornisci una spiegazione strutturata che includa:

1. **Ragioni Principali** (2-3 punti chiave)
   - Perché questa scommessa è consigliata/sconsigliata
   - Quali fattori sono più importanti

2. **Analisi Dati**
   - Form recente squadre
   - Eventuali infortuni rilevanti
   - Expected Value e value score

3. **Rischi e Considerazioni**
   - Eventuali red flags
   - Livello di confidence
   - Raccomandazioni sul sizing

Scrivi in {lang_text} in modo professionale ma comprensibile.
"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Sei un esperto analista di scommesse sportive."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return self._mock_explanation({"match": "N/A"}, "it")

    def _mock_explanation(self, context: Dict, language: str) -> str:
        """Mock explanation (rule-based)"""
        decision = context.get('decision', 'SKIP')
        ev = context.get('expected_value', 0)
        confidence = context.get('confidence', 50)

        if decision == "BET":
            explanation = f"""
🎯 **RACCOMANDAZIONE: SCOMMETTI**

**Ragioni Principali:**

1. **Value Eccellente** ({ev:+.1%} EV)
   - Le nostre analisi indicano che le quote attuali ({context.get('odds', 0):.2f}) offrono un valore significativo
   - La probabilità reale stimata è {context.get('probability', 0):.1%}

2. **Confidence Alta** ({confidence:.0f}/100)
   - I nostri modelli concordano sulla predizione
   - La qualità dei dati è buona

3. **Risk/Reward Favorevole**
   - Stake raccomandato: €{context.get('stake', 0):.2f}
   - Potenziale profitto proporzionato al rischio

**Considerazioni:**
"""
            if context.get('red_flags'):
                explanation += f"\n⚠️ Attenzione a {len(context.get('red_flags', []))} red flags rilevati"
            else:
                explanation += "\n✅ Nessun red flag significativo"

            explanation += "\n\n💡 **Conclusione**: Bet consigliata con sizing appropriato al livello di confidence."

        else:
            explanation = f"""
🚫 **RACCOMANDAZIONE: SKIP**

**Ragioni Principali:**

1. **Value Insufficiente** ({ev:+.1%} EV)
   - Le quote attuali non offrono sufficiente margine
   - Expected Value troppo basso per giustificare il rischio

2. **Confidence Non Ottimale** ({confidence:.0f}/100)
   - Incertezza elevata nella predizione
   - I modelli non concordano sufficientemente

**Considerazioni:**
- Meglio attendere opportunità con parametri migliori
- Risk/reward non favorevole in questo momento

💡 **Conclusione**: Meglio evitare questa bet e cercare value altrove.
"""

        return explanation

    def _mock_answer(self, question: str, context: Dict, language: str) -> str:
        """Mock answer to question"""
        question_lower = question.lower()

        if "perché" in question_lower or "why" in question_lower:
            return self._mock_explanation(context, language)
        elif "risk" in question_lower or "rischi" in question_lower:
            return f"""
**Analisi Rischi:**

- Confidence Score: {context.get('confidence', 50):.0f}/100
- Red Flags: {len(context.get('red_flags', []))}
- Data Quality: {context.get('data_quality', 0.5):.0%}

{'⚠️ Presenza di red flags da considerare' if context.get('red_flags') else '✅ Nessun red flag significativo'}

Stake raccomandato: €{context.get('stake', 0):.2f} (sizing prudente basato su confidence)
"""
        else:
            return "Per domande specifiche, fornisci più dettagli sul tipo di analisi che desideri."

    def _mock_comparison(self, ctx1: Dict, ctx2: Dict, language: str) -> str:
        """Mock comparison"""
        ev1 = ctx1.get('expected_value', 0)
        ev2 = ctx2.get('expected_value', 0)
        conf1 = ctx1.get('confidence', 50)
        conf2 = ctx2.get('confidence', 50)

        # Calculate scores
        score1 = ev1 * 100 + conf1 * 0.5
        score2 = ev2 * 100 + conf2 * 0.5

        winner = 1 if score1 > score2 else 2

        return f"""
**CONFRONTO PARTITE**

**{ctx1['match']}**
- EV: {ev1:+.1%}
- Confidence: {conf1:.0f}/100
- Score Complessivo: {score1:.1f}

**{ctx2['match']}**
- EV: {ev2:+.1%}
- Confidence: {conf2:.0f}/100
- Score Complessivo: {score2:.1f}

---

🏆 **RACCOMANDAZIONE**: Partita {winner} offre migliori parametri risk/reward.

{'Entrambe valide, ma consiglio di concentrare stake sulla migliore per ottimizzare bankroll.' if abs(score1 - score2) < 10 else 'Differenza significativa nei parametri chiave.'}
"""

    def _mock_strategy(self, summary: str, bankroll: float, language: str) -> str:
        """Mock strategy suggestions"""
        return f"""
**ANALISI STRATEGICA**

📊 **Bankroll Attuale**: €{bankroll:.2f}

**Raccomandazioni:**

1. **Diversificazione**
   - Mantieni max 5-7 bet attive contemporaneamente
   - Distribusci tra mercati diversi per ridurre correlazione

2. **Stake Sizing**
   - Con confidence >80: usa sizing pieno (Kelly)
   - Con confidence 60-80: riduci a 50% Kelly
   - Con confidence <60: considera skip

3. **Focus sui Mercati**
   - Concentrati su mercati dove hai più esperienza
   - Monitora performance per tipo di bet

4. **Risk Management**
   - Stop-loss giornaliero: -10% bankroll
   - Max stake singola bet: 5% bankroll
   - Review settimanale performance

💡 **Prossimi Steps**: Continua a tracciare performance e aggiusta strategia basandoti sui dati.
"""

    def _summarize_recent_analyses(self, analyses: List[Dict]) -> str:
        """Summarize recent analyses"""
        if not analyses:
            return "Nessuna analisi recente disponibile"

        total = len(analyses)
        bets = sum(1 for a in analyses if a.get('decision') == 'BET')
        avg_ev = sum(a.get('expected_value', 0) for a in analyses) / total if total > 0 else 0

        return f"""
Analisi Recenti: {total}
Bet Piazzate: {bets}
EV Medio: {avg_ev:+.1%}
"""


if __name__ == "__main__":
    # Test LLM Analyst
    logging.basicConfig(level=logging.INFO)

    print("Testing LLM Analyst (Mock Mode)...")
    print("=" * 70)

    analyst = LLMAnalyst(provider="mock")

    # Mock analysis result
    match_data = {
        'home': 'Inter',
        'away': 'Napoli',
        'league': 'Serie A'
    }

    analysis_result = {
        'summary': {
            'probability': 0.65,
            'confidence': 82,
            'value_score': 75,
            'expected_value': 0.12,
            'stake': 25.0,
            'odds': 1.85
        },
        'final_decision': {
            'action': 'BET'
        },
        'risk_decision': {
            'red_flags': [],
            'green_flags': ['High confidence', 'Good value']
        }
    }

    # Test explanation
    explanation = analyst.explain_prediction(match_data, analysis_result)
    print("\n📝 EXPLANATION:")
    print(explanation)

    # Test question
    print("\n" + "=" * 70)
    answer = analyst.answer_question("Quali sono i rischi?", match_data, analysis_result)
    print("\n❓ ANSWER:")
    print(answer)

    print("\n" + "=" * 70)
    print("✅ LLM Analyst test completed!")
