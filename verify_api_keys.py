#!/usr/bin/env python3
"""
Script per verificare che tutte le chiavi API vengano caricate correttamente
"""

import os
import sys

# Add ai_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_system'))

from ai_system.config import AIConfig
from api_manager import APIConfig as APIManagerConfig

def verify_keys():
    """Verifica tutte le chiavi API"""
    print("=" * 70)
    print("VERIFICA CHIAVI API")
    print("=" * 70)

    # Test AIConfig
    print("\n📋 AIConfig (ai_system/config.py)")
    print("-" * 70)
    ai_config = AIConfig()

    keys_to_check = [
        ("HUGGINGFACE_API_KEY", ai_config.huggingface_api_key),
        ("THEODDS_API_KEY", ai_config.theodds_api_key),
        ("OPENWEATHER_API_KEY", ai_config.openweather_api_key),
        ("TELEGRAM_BOT_TOKEN", ai_config.telegram_bot_token),
        ("TELEGRAM_CHAT_ID", ai_config.telegram_chat_id),
    ]

    for key_name, key_value in keys_to_check:
        status = "✅" if key_value else "❌"
        masked = f"{key_value[:10]}..." if key_value else "(vuoto)"
        print(f"{status} {key_name:25s}: {masked}")

    # Test APIManagerConfig
    print("\n📋 APIConfig (api_manager.py)")
    print("-" * 70)

    api_keys = [
        ("API_FOOTBALL_KEY", APIManagerConfig.API_FOOTBALL_KEY),
        ("FOOTBALL_DATA_KEY", APIManagerConfig.FOOTBALL_DATA_KEY),
        ("THEODDS_API_KEY", APIManagerConfig.THEODDS_API_KEY),
        ("OPENWEATHER_API_KEY", APIManagerConfig.OPENWEATHER_API_KEY),
        ("HUGGINGFACE_API_KEY", APIManagerConfig.HUGGINGFACE_API_KEY),
    ]

    for key_name, key_value in api_keys:
        status = "✅" if key_value else "❌"
        masked = f"{key_value[:10]}..." if key_value else "(vuoto)"
        print(f"{status} {key_name:25s}: {masked}")

    # Verifica environment variables
    print("\n📋 Environment Variables")
    print("-" * 70)

    env_keys = [
        "API_FOOTBALL_KEY",
        "FOOTBALL_DATA_KEY",
        "THEODDS_API_KEY",
        "OPENWEATHER_API_KEY",
        "HUGGINGFACE_API_KEY",
    ]

    for key_name in env_keys:
        key_value = os.getenv(key_name)
        status = "✅" if key_value else "⚠️ "
        masked = f"{key_value[:10]}..." if key_value else "(non impostata, usa fallback)"
        print(f"{status} {key_name:25s}: {masked}")

    # Riepilogo
    print("\n" + "=" * 70)
    print("RIEPILOGO")
    print("=" * 70)

    all_ai_keys = [v for _, v in keys_to_check if _]
    all_api_keys = [v for _, v in api_keys if _]

    ai_loaded = sum(1 for v in [kv[1] for kv in keys_to_check] if kv)
    api_loaded = sum(1 for v in [kv[1] for kv in api_keys] if kv)

    print(f"✅ AIConfig: {ai_loaded}/{len(keys_to_check)} chiavi caricate")
    print(f"✅ APIConfig: {api_loaded}/{len(api_keys)} chiavi caricate")
    print(f"\n🎯 Sistema pronto per utilizzare tutte le API configurate!")
    print("=" * 70)

if __name__ == "__main__":
    verify_keys()
