import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

AIRS_API_URL = os.getenv("AIRS_API_URL", "")
AIRS_SECURITY_PROFILE = os.getenv("AIRS_SECURITY_PROFILE", "")
AIRS_API_TOKEN = os.getenv("AIRS_API_TOKEN", "")
AIRS_TIMEOUT = int(os.getenv("AIRS_TIMEOUT", "60"))
AIRS_USAGE = os.getenv("AIRS_USAGE", "").strip().lower()


def scan_content(prompt: str, response: str = "") -> dict:
    """
    Send prompt and/or response to AIRS for safety scanning.
    Returns the parsed JSON response from AIRS.
    Raises RuntimeError if the scan indicates blocked content.
    """
    contents = []
    if prompt:
        contents.append({"prompt": prompt})
    if response:
        contents.append({"response": response})

    payload = {
        "contents": contents,
        "ai_profile": {
            "profile_name": AIRS_SECURITY_PROFILE
        }
    }
    headers = {
        "x-pan-token": AIRS_API_TOKEN
    }

    api_response = requests.post(AIRS_API_URL, json=payload, headers=headers, timeout=AIRS_TIMEOUT)
    api_response.raise_for_status()
    result = api_response.json()

    action = result.get("action", "Unknown action")
    if action == "block":
        raise RuntimeError(f"AIRS blocked content: {result.get('category', 'unknown')}")

    return result