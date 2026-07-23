import http.client
import json
import os
from dotenv import load_dotenv

load_dotenv()

AIRS_API_URL = os.getenv("AIRS_API_URL", "")
AIRS_SECURITY_PROFILE = os.getenv("AIRS_SECURITY_PROFILE", "")
AIRS_API_TOKEN = os.getenv("AIRS_API_TOKEN", "")


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

    payload = json.dumps({
        "tr_id": "string",
        "session_id": "string",
        "ai_profile": {
            "profile_name": AIRS_SECURITY_PROFILE
        },
        "metadata": {
            "app_name": "ai-callcenter",
            "app_user": "string",
            "ai_model": "string",
            "user_ip": "string",
            "agent_meta": {
                "agent_id": "string",
                "agent_version": "string",
                "agent_arn": "string"
            }
        },
        "contents": contents
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-pan-token': AIRS_API_TOKEN
    }

    conn = http.client.HTTPSConnection(AIRS_API_URL)
    conn.request("POST", "/v1/scan/sync/request", payload, headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()

    result = json.loads(data.decode("utf-8"))

    action = result.get("action", "Unknown action")
    if action == "block":
        raise RuntimeError(f"AIRS blocked content: {result.get('category', 'unknown')}")

    return result