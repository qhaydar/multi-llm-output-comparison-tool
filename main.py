# multi_llm_chat.py
"""Chatbot that queries OpenAI, Anthropic Claude Sonnet, and Google Gemini.

Each model receives the same user input and system prompt, and their responses are
displayed side‑by‑side for easy comparison.

Environment variables required:
- ``OPENAI_API_KEY`` – OpenAI API key
- ``ANTHROPIC_API_KEY`` – Anthropic API key (for Claude Sonnet)
- ``GOOGLE_API_KEY`` – Google API key (for Gemini)

Install the required SDKs:
```bash
pip install openai anthropic google-generativeai
```
"""

import os
import sys

# OpenAI
from openai import OpenAI

# Anthropic (Claude Sonnet)
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Google Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# System prompt – shared across all models
SYSTEM_PROMPT = "You are a sarcastic assistant who answers humorously."

# ---------------------------------------------------------------------------
# Helper functions for each provider
# ---------------------------------------------------------------------------

def get_openai_reply(user_message: str, history: list) -> str:
    """Send the conversation history to OpenAI and return the assistant reply."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    msgs = history + [{"role": "user", "content": user_message}]
    response = client.chat.completions.create(
        model="gpt-5",
        messages=msgs,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def get_claude_reply(user_message: str, history: list) -> str:
    """Query Anthropic Claude Sonnet. Returns the assistant reply."""
    if Anthropic is None:
        raise RuntimeError("anthropic package not installed")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msgs = history + [{"role": "user", "content": user_message}]
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        temperature=0.1,
        system=SYSTEM_PROMPT,
        messages=msgs,
    )
    return response.content[0].text.strip()


def get_gemini_reply(user_message: str, history: list) -> str:
    """Query Google Gemini. Returns the assistant reply."""
    if genai is None:
        raise RuntimeError("google-generativeai package not installed")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    context = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = f"System: {SYSTEM_PROMPT}\n{context}\nUser: {user_message}"
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main() -> None:
    openai_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    claude_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    gemini_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("👋 Bye!")
            break
        try:
            oa_reply = get_openai_reply(user_input, openai_history)
            openai_history.append({"role": "user", "content": user_input})
            openai_history.append({"role": "assistant", "content": oa_reply})
        except Exception as e:
            oa_reply = f"[Error: {e}]"
        try:
            cl_reply = get_claude_reply(user_input, claude_history)
            claude_history.append({"role": "user", "content": user_input})
            claude_history.append({"role": "assistant", "content": cl_reply})
        except Exception as e:
            cl_reply = f"[Error: {e}]"
        try:
            gm_reply = get_gemini_reply(user_input, gemini_history)
            gemini_history.append({"role": "user", "content": user_input})
            gemini_history.append({"role": "assistant", "content": gm_reply})
        except Exception as e:
            gm_reply = f"[Error: {e}]"
        print("\n--- Model Responses ---")
        print(f"OpenAI   : {oa_reply}")
        print(f"Claude    : {cl_reply}")
        print(f"Gemini   : {gm_reply}\n")

if __name__ == "__main__":
    main()
