#!/usr/bin/env python3
"""
Market Movement Analyzer
Analizza il movimento del mercato e formatta lo spread correttamente:
- (-) indica casa (home)
- (+) indica trasferta (away)
"""

def format_spread_for_display(spread: float) -> str:
    """
    Formatta lo spread per la visualizzazione.
    
    Args:
        spread: Valore spread (lambda_h - lambda_a)
                - spread > 0: Casa favorita → mostra (-)
                - spread < 0: Trasferta favorita → mostra (+)
    
    Returns:
        Stringa formattata che mostra il segno corretto per casa/trasferta
        - (-) indica casa
        - (+) indica trasferta
    """
    if spread is None:
        return "N/A"
    
    # Se spread viene calcolato come lambda_h - lambda_a:
    # - spread > 0: Casa favorita → mostra (-)
    # - spread < 0: Trasferta favorita → mostra (+)
    # Quindi invertiamo il segno per la visualizzazione
    if spread > 0:
        return f"-{abs(spread):.2f} (Casa favorita)"
    elif spread < 0:
        return f"+{abs(spread):.2f} (Trasferta favorita)"
    else:
        return "0.00 (Bilanciato)"


def format_spread_compact(spread: float) -> str:
    """
    Formatta lo spread in formato compatto.
    
    Args:
        spread: Valore spread (lambda_h - lambda_a)
    
    Returns:
        Stringa formattata compatta
        - (-) indica casa
        - (+) indica trasferta
    """
    if spread is None:
        return "N/A"
    
    # Se spread viene calcolato come lambda_h - lambda_a:
    # - spread > 0: Casa favorita → mostra (-)
    # - spread < 0: Trasferta favorita → mostra (+)
    # Quindi invertiamo il segno per la visualizzazione
    if spread > 0:
        return f"-{abs(spread):.2f}"
    elif spread < 0:
        return f"+{abs(spread):.2f}"
    else:
        return "0.00"


def analyze_market_movement(spread_apertura: float, spread_corrente: float) -> dict:
    """
    Analizza il movimento del mercato tra apertura e corrente.
    
    Args:
        spread_apertura: Spread all'apertura (lambda_h - lambda_a)
        spread_corrente: Spread corrente (lambda_h - lambda_a)
    
    Returns:
        Dizionario con analisi del movimento
    """
    if spread_apertura is None or spread_corrente is None:
        return {"error": "Spread mancante"}
    
    movement = spread_corrente - spread_apertura
    movement_abs = abs(movement)
    
    # Determina chi è favorito
    # Se spread viene calcolato come lambda_h - lambda_a:
    # - spread > 0: Casa favorita → mostra (-)
    # - spread < 0: Trasferta favorita → mostra (+)
    if spread_corrente > 0:
        favorito = "Casa"
        segno_favorito = "-"
    elif spread_corrente < 0:
        favorito = "Trasferta"
        segno_favorito = "+"
    else:
        favorito = "Bilanciato"
        segno_favorito = "0"
    
    # Determina la direzione del movimento
    # Se spread aumenta (più positivo), casa guadagna vantaggio
    # Se spread diminuisce (più negativo), trasferta guadagna vantaggio
    if movement > 0.1:
        direzione = "Casa guadagna vantaggio"
    elif movement < -0.1:
        direzione = "Trasferta guadagna vantaggio"
    else:
        direzione = "Stabile"
    
    return {
        "spread_apertura": spread_apertura,
        "spread_corrente": spread_corrente,
        "spread_apertura_formatted": format_spread_compact(spread_apertura),
        "spread_corrente_formatted": format_spread_compact(spread_corrente),
        "movement": movement,
        "movement_abs": movement_abs,
        "favorito": favorito,
        "segno_favorito": segno_favorito,
        "direzione": direzione,
        "movement_type": "HIGH" if movement_abs > 0.4 else "MODERATE" if movement_abs > 0.2 else "STABLE"
    }


def print_market_movement_analysis(spread_apertura: float, spread_corrente: float):
    """
    Stampa l'analisi del movimento del mercato.
    
    Args:
        spread_apertura: Spread all'apertura (lambda_h - lambda_a)
        spread_corrente: Spread corrente (lambda_h - lambda_a)
    """
    analysis = analyze_market_movement(spread_apertura, spread_corrente)
    
    if "error" in analysis:
        print(f"❌ Errore: {analysis['error']}")
        return
    
    print("=" * 80)
    print("ANALISI MOVIMENTO MERCATO")
    print("=" * 80)
    print(f"\n📊 SPREAD APERTURA:")
    print(f"   Valore: {analysis['spread_apertura_formatted']}")
    if analysis['spread_apertura'] > 0:
        print(f"   Interpretazione: Casa favorita (-)")
    elif analysis['spread_apertura'] < 0:
        print(f"   Interpretazione: Trasferta favorita (+)")
    else:
        print(f"   Interpretazione: Bilanciato")
    
    print(f"\n📊 SPREAD CORRENTE:")
    print(f"   Valore: {analysis['spread_corrente_formatted']}")
    if analysis['spread_corrente'] > 0:
        print(f"   Interpretazione: Casa favorita (-)")
    elif analysis['spread_corrente'] < 0:
        print(f"   Interpretazione: Trasferta favorita (+)")
    else:
        print(f"   Interpretazione: Bilanciato")
    
    print(f"\n📈 MOVIMENTO:")
    print(f"   Variazione: {analysis['movement']:+.3f}")
    print(f"   Direzione: {analysis['direzione']}")
    print(f"   Tipo: {analysis['movement_type']}")
    print(f"   Favorito corrente: {analysis['favorito']} ({analysis['segno_favorito']})")
    print("=" * 80)


# Esempio di utilizzo
if __name__ == "__main__":
    # Esempio 1: Casa favorita (spread negativo)
    print("\n🔍 ESEMPIO 1: Casa favorita")
    print_market_movement_analysis(spread_apertura=-0.5, spread_corrente=-0.75)
    
    # Esempio 2: Trasferta favorita (spread positivo)
    print("\n🔍 ESEMPIO 2: Trasferta favorita")
    print_market_movement_analysis(spread_apertura=0.25, spread_corrente=0.50)
    
    # Esempio 3: Movimento da casa a trasferta
    print("\n🔍 ESEMPIO 3: Movimento da casa a trasferta")
    print_market_movement_analysis(spread_apertura=-0.25, spread_corrente=0.25)

