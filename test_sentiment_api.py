#!/usr/bin/env python3
"""
Test Sentiment Analysis with Hugging Face API
==============================================

Testa l'integrazione della Hugging Face API per sentiment analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ai_system.sentiment_analyzer import SentimentAnalyzer
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_huggingface_api():
    """Test Hugging Face sentiment API"""
    print("=" * 80)
    print("🧪 TESTING HUGGING FACE SENTIMENT API")
    print("=" * 80)

    # Initialize analyzer
    print("\n1️⃣  Initializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()

    # Test with sample texts
    test_texts = [
        "Great performance by the team! Very confident about the next match.",
        "Terrible injury news, key player out for 3 weeks.",
        "The match was okay, nothing special.",
        "Manchester United injury concern: Rashford limping after training",
        "Liverpool fans are very excited about the upcoming derby!"
    ]

    print("\n2️⃣  Testing sentiment analysis on sample texts:")
    print("-" * 80)

    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Text {i}: {text[:60]}...")

        result = analyzer._analyze_text_sentiment_hf(text)

        if result.get('success'):
            label = result.get('label', 'UNKNOWN')
            score = result.get('score', 0.0)
            print(f"   ✅ Sentiment: {label} (confidence: {score:.2%})")
        else:
            print(f"   ⚠️  Failed: {result.get('error', 'Unknown error')}")

    # Test full match analysis
    print("\n" + "=" * 80)
    print("3️⃣  Testing full match sentiment analysis:")
    print("-" * 80)

    print("\n🔍 Analyzing: Inter vs Napoli")
    result = analyzer.analyze_match_sentiment(
        team_home="Inter",
        team_away="Napoli",
        hours_before=48,
        include_sources=['news']  # Use only news (free, no API keys needed)
    )

    print(f"\n📊 RESULTS:")
    print(f"   🏠 HOME (Inter):")
    print(f"      Morale: {result['team_morale_home']:+.1f}")
    print(f"      Fan Confidence: {result['fan_confidence_home']:.1f}%")
    print(f"      Overall Sentiment: {result['overall_sentiment_home']:+.1f}")

    print(f"\n   ✈️  AWAY (Napoli):")
    print(f"      Morale: {result['team_morale_away']:+.1f}")
    print(f"      Fan Confidence: {result['fan_confidence_away']:.1f}%")
    print(f"      Overall Sentiment: {result['overall_sentiment_away']:+.1f}")

    print(f"\n   🚨 Signals detected: {len(result['signals'])}")
    for signal in result['signals'][:5]:  # Show first 5
        print(f"      - {signal['type']}: {signal['text'][:50]}...")
        if 'sentiment' in signal:
            print(f"        Sentiment: {signal['sentiment']} ({signal.get('sentiment_score', 0):.2%})")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED!")
    print("=" * 80)

    # Instructions
    print("\n💡 NEXT STEPS:")
    print("   1. Create a FREE Hugging Face account at: https://huggingface.co/join")
    print("   2. Get your API token at: https://huggingface.co/settings/tokens")
    print("   3. Add to .env file: HUGGINGFACE_API_KEY=your_token_here")
    print("   4. This will give you higher rate limits (still FREE!)")
    print("\n   Without API key: Works fine for testing and small usage")
    print("   With API key: Better for production usage\n")

if __name__ == "__main__":
    test_huggingface_api()
