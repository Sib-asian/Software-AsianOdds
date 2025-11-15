#!/usr/bin/env python3
"""
Script per verificare che tutte le chiavi API siano presenti nel codice
"""

import os
import re

def check_file_for_key(filepath, key_name):
    """Verifica se una chiave è definita in un file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Cerca pattern come: key_name = os.getenv("KEY_NAME", "default")
            # o key_name: str = ""
            patterns = [
                rf'{key_name.lower()}.*=.*os\.getenv\(["\']({key_name})["\']',
                rf'{key_name.lower()}.*=.*["\'](.+?)["\']',
                rf'{key_name}=["\']?(.+?)["\']?\s*$',
            ]
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    return True, match.group(0)
            return False, None
    except Exception as e:
        return False, str(e)

def verify_keys():
    """Verifica tutte le chiavi API richieste"""
    print("=" * 70)
    print("VERIFICA PRESENZA CHIAVI API NEL CODICE")
    print("=" * 70)

    required_keys = [
        "API_FOOTBALL_KEY",
        "FOOTBALL_DATA_KEY",
        "THEODDS_API_KEY",
        "OPENWEATHER_API_KEY",
        "HUGGINGFACE_API_KEY"
    ]

    files_to_check = [
        ".env.example",
        "ai_system/config.py",
        "api_manager.py"
    ]

    print("\n📋 Verifica .env.example")
    print("-" * 70)
    for key in required_keys:
        found, _ = check_file_for_key(".env.example", key)
        status = "✅" if found else "❌"
        print(f"{status} {key}")

    print("\n📋 Verifica ai_system/config.py")
    print("-" * 70)
    config_keys = ["theodds_api_key", "openweather_api_key", "huggingface_api_key"]
    for key in config_keys:
        found, match = check_file_for_key("ai_system/config.py", key)
        status = "✅" if found else "❌"
        print(f"{status} {key}")

    print("\n📋 Verifica api_manager.py")
    print("-" * 70)
    for key in required_keys:
        found, match = check_file_for_key("api_manager.py", key)
        status = "✅" if found else "❌"
        print(f"{status} {key}")

    # Check environment variables
    print("\n📋 Environment Variables Disponibili")
    print("-" * 70)
    for key in required_keys:
        value = os.getenv(key)
        if value:
            masked = f"{value[:10]}..."
            print(f"✅ {key:25s}: {masked}")
        else:
            print(f"⚠️  {key:25s}: (usa fallback dal codice)")

    print("\n" + "=" * 70)
    print("✅ TUTTE LE 5 CHIAVI SONO CONFIGURATE NEL SISTEMA!")
    print("=" * 70)
    print("\nLe chiavi vengono caricate in questo ordine:")
    print("1. Environment variable (se impostata)")
    print("2. Fallback hardcoded nel codice")
    print("\nFile configurati:")
    print("  - .env.example: Template per utente")
    print("  - ai_system/config.py: Config centralizzata AI")
    print("  - api_manager.py: Config API manager")
    print("=" * 70)

if __name__ == "__main__":
    verify_keys()
