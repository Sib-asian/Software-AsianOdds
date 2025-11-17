from telegram_multi_formatter import (
    format_multiple_matches_for_telegram,
    split_telegram_message,
)


def _build_ris(**overrides):
    base = {
        "p_home": 0.55,
        "p_draw": 0.25,
        "p_away": 0.20,
        "over_25": 0.62,
        "under_25": 0.38,
        "btts": 0.58,
        "gg_over25": 0.44,
        "even_ft": 0.49,
        "odd_ft": 0.51,
        "cs_home": 0.30,
        "cs_away": 0.22,
        "over_05_ht": 0.74,
        "over_15_ht": 0.46,
        "even_ht": 0.50,
        "odd_ht": 0.50,
        "over_05ht_over_25ft": 0.60,
        "over_15ht_over_25ft": 0.41,
        "asian_handicap": {"AH -0.5": 0.54, "AH +0.5": 0.66},
        "ht_ft": {"1-1": 0.32, "X-1": 0.18, "2-2": 0.10},
        "top10": [(2, 1, 14.2), (1, 1, 12.0), (3, 1, 7.5)],
        "multigol": {"1-3": 0.64},
    }
    base.update(overrides)
    return base


def test_format_multiple_matches_includes_each_match():
    analyses = [
        {
            "match_name": "Inter vs Milan",
            "ris": _build_ris(),
            "odds_1": 1.90,
            "odds_x": 3.40,
            "odds_2": 3.80,
            "quality_score": 82.4,
            "market_conf": 77.2,
            "value_bets": [
                {"Esito": "1 fisso", "Prob %": "62", "Edge %": "+5", "EV %": "+8"},
            ],
        },
        {
            "match_name": "Juventus vs Roma",
            "ris": _build_ris(
                p_home=0.48,
                p_draw=0.27,
                p_away=0.25,
                top10=[(1, 0, 13.2), (2, 1, 11.5), (2, 2, 8.4)],
            ),
            "odds_1": 2.10,
            "odds_x": 3.30,
            "odds_2": 3.60,
            "quality_score": 75.0,
            "market_conf": 70.0,
            "value_bets": [],
        },
    ]

    message = format_multiple_matches_for_telegram(analyses, telegram_prob_threshold=55.0)

    assert "Inter vs Milan" in message
    assert "Juventus vs Roma" in message
    assert "Value Bets (1)" in message
    assert "Nessun value bet sopra soglia" in message


def test_split_telegram_message_respects_custom_limit():
    analyses = []
    for idx in range(8):
        ris = _build_ris(
            p_home=0.45 + idx * 0.02,
            p_draw=0.30,
            p_away=0.25 - idx * 0.01,
        )
        analyses.append(
            {
                "match_name": f"Match {idx + 1}",
                "ris": ris,
                "odds_1": 1.80 + idx * 0.05,
                "odds_x": 3.20,
                "odds_2": 4.00,
                "quality_score": 70 + idx,
                "market_conf": 65 + idx,
                "value_bets": [
                    {"Esito": "1 fisso", "Prob %": "55", "Edge %": "+3", "EV %": "+4"},
                ],
            }
        )

    message = format_multiple_matches_for_telegram(analyses, telegram_prob_threshold=40.0)
    chunks = split_telegram_message(message, max_length=900)

    assert len(chunks) > 1
    assert all(len(chunk) <= 900 for chunk in chunks)
