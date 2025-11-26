from datetime import datetime
from typing import Any, Dict, List


def format_multiple_matches_for_telegram(
    analyses: List[Dict[str, Any]],
    telegram_prob_threshold: float = 50.0
) -> str:
    """
    Format a single Telegram message that aggregates multiple match analyses.

    Args:
        analyses: List of dicts with keys: match_name, ris, odds_1, odds_x, odds_2,
                  quality_score, market_conf, value_bets
        telegram_prob_threshold: Minimum probability threshold to include value bets

    Returns:
        HTML-formatted Telegram message string
    """
    total_matches = len(analyses)

    message = f"📊 <b>ANALISI MULTIPLA - {total_matches} PARTIT{'A' if total_matches == 1 else 'E'}</b>\n"
    message += f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
    message += "━━━━━━━━━━━━━━━━━━━━━\n\n"

    total_value_bets = 0
    for i, analysis in enumerate(analyses, 1):
        match_name = analysis["match_name"]
        ris = analysis["ris"]
        odds_1 = analysis["odds_1"]
        odds_x = analysis["odds_x"]
        odds_2 = analysis["odds_2"]
        quality_score = analysis.get("quality_score", 0.0)
        market_conf = analysis.get("market_conf", 0.0)
        value_bets = analysis.get("value_bets", [])

        message += f"<b>{i}. ⚽ {match_name}</b>\n\n"
        message += f"📊 Quality: {quality_score:.0f}/100 | Confidence: {market_conf:.0f}/100\n\n"

        message += "🎯 <b>Probabilità:</b>\n"
        message += f"  1️⃣ Casa: <b>{ris['p_home']*100:.1f}%</b> (q. {odds_1:.2f})\n"
        message += f"  ❌ Pareggio: <b>{ris['p_draw']*100:.1f}%</b> (q. {odds_x:.2f})\n"
        message += f"  2️⃣ Trasferta: <b>{ris['p_away']*100:.1f}%</b> (q. {odds_2:.2f})\n\n"

        message += (
            f"⚽ Over 2.5: {ris['over_25']*100:.1f}% | "
            f"Under: {ris['under_25']*100:.1f}% | "
            f"BTTS: {ris['btts']*100:.1f}% | "
            f"GG+Over 2.5: {ris['gg_over25']*100:.1f}%\n"
        )

        if 'even_ft' in ris and 'odd_ft' in ris:
            message += (
                f"🎲 Pari: {ris['even_ft']*100:.1f}% | "
                f"Dispari: {ris['odd_ft']*100:.1f}%\n"
            )

        if 'cs_home' in ris and 'cs_away' in ris:
            message += (
                f"🛡️ CS Casa: {ris['cs_home']*100:.1f}% | "
                f"CS Trasferta: {ris['cs_away']*100:.1f}%\n"
            )

        ht_markets = []
        if 'over_05_ht' in ris:
            ht_markets.append(f"Over 0.5 HT: {ris['over_05_ht']*100:.1f}%")
        if 'over_15_ht' in ris:
            ht_markets.append(f"Over 1.5 HT: {ris['over_15_ht']*100:.1f}%")
        if 'even_ht' in ris and 'odd_ht' in ris:
            ht_markets.append(f"Pari HT: {ris['even_ht']*100:.1f}%")
            ht_markets.append(f"Dispari HT: {ris['odd_ht']*100:.1f}%")
        if ht_markets:
            message += f"⏱️ {' | '.join(ht_markets)}\n"

        combined_markets = []
        if 'over_05ht_over_25ft' in ris:
            combined_markets.append(f"Over 0.5 HT + Over 2.5 FT: {ris['over_05ht_over_25ft']*100:.1f}%")
        if 'over_15ht_over_25ft' in ris:
            combined_markets.append(f"Over 1.5 HT + Over 2.5 FT: {ris['over_15ht_over_25ft']*100:.1f}%")
        if combined_markets:
            message += f"🔗 {' | '.join(combined_markets)}\n"

        if 'asian_handicap' in ris:
            ah = ris['asian_handicap']
            relevant_ah = [(k, v) for k, v in ah.items() if v >= 0.25]
            if relevant_ah:
                relevant_ah.sort(key=lambda x: x[1], reverse=True)
                ah_str = ' | '.join([f"{k}: {v*100:.1f}%" for k, v in relevant_ah[:2]])
                message += f"🎯 {ah_str}\n"

        if 'ht_ft' in ris:
            ht_ft = ris['ht_ft']
            sorted_ht_ft = sorted(ht_ft.items(), key=lambda x: x[1], reverse=True)
            top_ht_ft = [(combo, prob) for combo, prob in sorted_ht_ft[:3] if prob >= 0.08]
            if top_ht_ft:
                ht_ft_str = ' | '.join([f"{combo}: {prob*100:.1f}%" for combo, prob in top_ht_ft])
                message += f"⏱️🏁 {ht_ft_str}\n"

        message += "\n"

        if "top10" in ris and len(ris["top10"]) > 0:
            message += "🏅 Top 3 Risultati:\n"
            for idx, (h, a, p) in enumerate(ris["top10"][:3], 1):
                message += f"  {idx}. {h}-{a}: {p:.1f}%\n"
            message += "\n"

        filtered_vbs = [
            vb for vb in value_bets
            if float(str(vb.get("Prob %", "0")).replace("%", "").replace(",", ".")) >= telegram_prob_threshold
        ]

        if filtered_vbs:
            total_value_bets += len(filtered_vbs)
            message += f"💎 <b>Value Bets ({len(filtered_vbs)}):</b>\n"
            for vb in filtered_vbs[:3]:
                esito = vb.get("Esito", "")
                prob = vb.get("Prob %", vb.get("Prob Modello %", ""))
                edge = vb.get("Edge %", "0")
                ev = vb.get("EV %", "0")
                prob_str = f"{prob}%" if prob and "%" not in str(prob) else prob
                message += f"  • {esito}: {prob_str} | Edge {edge}% | EV {ev}%\n"
            if len(filtered_vbs) > 3:
                message += f"  <i>(+{len(filtered_vbs)-3} altri)</i>\n"
        else:
            message += f"ℹ️ Nessun value bet sopra soglia ({telegram_prob_threshold:.0f}%)\n"

        message += "\n━━━━━━━━━━━━━━━━━━━━━\n\n"

    message += "📈 <b>Riepilogo:</b>\n"
    message += f"  • Partite analizzate: {total_matches}\n"
    message += f"  • Value bets totali: {total_value_bets}\n"
    message += f"  • Soglia probabilità: ≥{telegram_prob_threshold:.0f}%\n\n"
    message += "🤖 <i>Analisi automatica - Modello Dixon-Coles Bayesiano</i>"

    return message


def split_telegram_message(message: str, max_length: int = 4096) -> List[str]:
    """
    Split a Telegram message into multiple parts respecting Telegram's 4096 char limit.
    """
    if len(message) <= max_length:
        return [message]

    messages: List[str] = []
    sections = message.split("━━━━━━━━━━━━━━━━━━━━━\n\n")

    header = sections[0]
    footer = ""
    if "🤖 <i>Analisi automatica" in sections[-1]:
        footer = "\n\n" + sections[-1].split("📈 <b>Riepilogo:</b>")[0]
        sections[-1] = sections[-1].split("🤖 <i>Analisi automatica")[0]

    current = header + "━━━━━━━━━━━━━━━━━━━━━\n\n"

    for section in sections[1:]:
        section_with_sep = section + "━━━━━━━━━━━━━━━━━━━━━\n\n"

        if len(current) + len(section_with_sep) + len(footer) > max_length:
            messages.append(current.rstrip() + footer)
            current = header + "━━━━━━━━━━━━━━━━━━━━━\n\n" + section_with_sep
        else:
            current += section_with_sep

    if current:
        messages.append(current.rstrip() + footer)

    return messages
