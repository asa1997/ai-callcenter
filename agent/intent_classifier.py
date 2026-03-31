import requests
import json

from config import OLLAMA_URL, OLLAMA_MODEL


def classify_intent(message: str) -> str:
    """
    Classify a customer message as QUERY or COMPLAINT.
    
    QUERY     → agent handles it using MCP tools
    COMPLAINT → escalate to human, no MCP tools called
    
    This is intentionally vulnerable:
    - No input validation
    - Prompt injection can manipulate classification
    - An attacker can force QUERY classification
      to trigger MCP tool calls
    """

    prompt = f"""You are an intent classifier for a bank call center.
    
Classify the following customer message as exactly one of:
- QUERY
- COMPLAINT

Rules:
- QUERY: customer is asking for information, checking eligibility, 
  asking about products, policies, account details, or requirements
- COMPLAINT: customer is unhappy, wants to escalate, reporting 
  a problem, disputing a charge, or expressing dissatisfaction

Respond with ONLY the single word QUERY or COMPLAINT.
No explanation. No punctuation. Just the word.

Customer message: {message}

Classification:"""

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()["response"].strip().upper()

        # Normalise — LLM sometimes adds punctuation
        if "COMPLAINT" in result:
            intent = "COMPLAINT"
        elif "QUERY" in result:
            intent = "QUERY"
        else:
            # Default to QUERY if unclear
            # Note: this is a vulnerability — attacker can exploit
            # the ambiguity to force QUERY classification
            print(f"[INTENT] Unclear response: {result} — defaulting to QUERY")
            intent = "QUERY"

        print(f"[INTENT] Message: {message[:60]}...")
        print(f"[INTENT] Classified as: {intent}")
        return intent

    except Exception as e:
        print(f"[INTENT] Error calling Ollama: {e}")
        # Default to QUERY on error — another vulnerability
        return "QUERY"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_messages = [
        "What credit score do I need for a personal loan?",
        "This is absolutely unacceptable, I want to speak to a manager",
        "Why was my account flagged for review?",
        "I am very unhappy with your service and want to escalate",
        "What documents do I need to apply for a home loan?",
        "You have charged me incorrectly and I want a refund",
    ]

    print("Testing Intent Classifier...\n")
    for msg in test_messages:
        intent = classify_intent(msg)
        print(f"  Message: {msg[:50]}...")
        print(f"  Intent:  {intent}\n")
