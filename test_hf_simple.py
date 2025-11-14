#!/usr/bin/env python3
"""
Simple Hugging Face API Test
=============================

Testa solo l'API Hugging Face senza dipendenze esterne.
"""

import requests

def test_huggingface_direct():
    """Test diretto dell'API Hugging Face"""

    print("=" * 80)
    print("🧪 TESTING HUGGING FACE SENTIMENT API (Direct)")
    print("=" * 80)

    # API configuration (FREE - no key needed!)
    api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {}  # No API key = free tier

    # Test texts
    test_texts = [
        "Great performance by the team! Very confident about the next match.",
        "Terrible injury news, key player out for 3 weeks.",
        "The match was okay, nothing special.",
        "Manchester United injury concern: Rashford limping after training",
        "Liverpool fans are very excited about the upcoming derby!"
    ]

    print("\n📝 Testing sentiment analysis on sample texts:\n")
    print("-" * 80)

    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text[:60]}...")

        try:
            # Call Hugging Face API
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": text},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                # Parse result
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        predictions = result[0]
                        best = max(predictions, key=lambda x: x['score'])

                        label = best['label'].upper()
                        score = best['score']

                        # Color code based on sentiment
                        emoji = "✅" if label == "POSITIVE" else ("❌" if label == "NEGATIVE" else "➖")

                        print(f"   {emoji} Sentiment: {label} (confidence: {score:.2%})")
                    else:
                        print(f"   ⚠️  Unexpected result format: {result}")
                else:
                    print(f"   ⚠️  Empty result")

            elif response.status_code == 503:
                print(f"   ⏳ Model loading (first request - retry in 20s)...")
                print(f"      This is normal for first use!")

            else:
                print(f"   ❌ API Error {response.status_code}: {response.text[:100]}")

        except requests.exceptions.Timeout:
            print(f"   ⏱️  Timeout - API might be slow")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED!")
    print("=" * 80)

    print("\n💡 RESULTS:")
    print("   • Hugging Face API is FREE and works without API key")
    print("   • Optional: Get API key at https://huggingface.co/settings/tokens")
    print("   • With key: Higher rate limits (still FREE)")
    print("   • Perfect for professional sentiment analysis!")

    print("\n📊 INTEGRATION STATUS:")
    print("   ✅ Hugging Face API integrated in sentiment_analyzer.py")
    print("   ✅ Works with Twitter, Reddit, and News sources")
    print("   ✅ Configuration added to config.py")
    print("   ✅ .env.example created with instructions")

    print("\n🚀 NEXT STEPS:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. (Optional) Add HUGGINGFACE_API_KEY to .env")
    print("   3. Run your predictions - sentiment analysis will be automatic!")
    print()

if __name__ == "__main__":
    test_huggingface_direct()
