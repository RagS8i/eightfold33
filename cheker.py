"""
check_api_key.py
Run: python check_api_key.py
"""

import os
import sys

# ── Read key from environment (no .env file, pure os.environ) ──
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

print("=" * 50)
print("🔑 Gemini API Key — Environment Check")
print("=" * 50)

if not api_key:
    print("❌ GEMINI_API_KEY not set in environment.")
    print()
    print("Set it before running:")
    print("  Windows PowerShell:")
    print('    $env:GEMINI_API_KEY="AIza..."')
    print("  Windows CMD:")
    print("    set GEMINI_API_KEY=AIza...")
    sys.exit(1)

masked = api_key[:6] + "..." + api_key[-4:]
print(f"✅ Key found in environment: {masked}")
print(f"   Length: {len(api_key)} chars")

if not api_key.startswith("AIza"):
    print("⚠️  Warning: valid Gemini keys start with 'AIza'")

print()
print("🌐 Testing API call...")

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Reply with the single word: WORKING"
    )
    print(f"✅ Success! Model replied: {response.text.strip()}")
    print()
    print("Your API key is valid and ready to use.")

except Exception as e:
    err = str(e)
    print(f"❌ Failed: {err}")
    print()
    if "API_KEY_INVALID" in err or "not valid" in err.lower():
        print("→ Your key is invalid or revoked.")
        print("  Get a new one at: https://aistudio.google.com/apikey")
    elif "429" in err or "quota" in err.lower():
        print("→ Key is valid but quota exhausted. Wait and retry.")
    elif "connect" in err.lower():
        print("→ Network error. Check your internet connection.")
    sys.exit(1)