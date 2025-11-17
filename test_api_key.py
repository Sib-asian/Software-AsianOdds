#!/usr/bin/env python3
"""
Test Hugging Face API Key
==========================
"""

import os

# Load .env file manually
env_file = '/home/user/Software-AsianOdds/.env'
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Get API key
api_key = os.getenv('HUGGINGFACE_API_KEY')

print("=" * 80)
print("🔑 TESTING HUGGING FACE API KEY")
print("=" * 80)

if api_key:
    print(f"\n✅ API Key found: {api_key[:10]}...{api_key[-5:]}")
    print(f"   Length: {len(api_key)} characters")

    # Test the API
    print("\n📡 Testing API connection...")

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=api_key)

        # Test with a simple sentiment analysis
        test_text = "Great match! The team played amazing football today."

        print(f"   Test text: '{test_text}'")
        print(f"   Calling Hugging Face API...")

        result = client.text_classification(
            text=test_text,
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        if result:
            best = max(result, key=lambda x: x['score'])
            print(f"\n✅ API WORKING PERFECTLY!")
            print(f"   Sentiment: {best['label'].upper()}")
            print(f"   Confidence: {best['score']:.2%}")
            print(f"\n🎉 You now have HIGHER RATE LIMITS (~1000 req/hour)")
            print(f"   vs ~100 req/hour without key")

    except ImportError:
        print("\n⚠️  huggingface_hub not installed")
        print("   Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"\n❌ API Error: {e}")

else:
    print("\n❌ No API key found in .env file")

print("\n" + "=" * 80)
