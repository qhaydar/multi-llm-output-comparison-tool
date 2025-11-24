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
import time

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

import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_get_reply(func, user_message: str, history: list) -> tuple[str, float]:
    """Run a synchronous LLM reply function in a thread pool and return its result plus elapsed time.
    Returns a tuple of (reply, duration_seconds)."""
    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    result = await loop.run_in_executor(None, func, user_message, history)
    elapsed = time.perf_counter() - start
    return result, elapsed

async def main_async() -> None:
    openai_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    claude_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    gemini_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    executor = ThreadPoolExecutor(max_workers=3)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("👋 Bye!")
            break
        # Schedule all three model calls concurrently and capture timing
        oa_task = asyncio.create_task(async_get_reply(get_openai_reply, user_input, openai_history))
        cl_task = asyncio.create_task(async_get_reply(get_claude_reply, user_input, claude_history))
        gm_task = asyncio.create_task(async_get_reply(get_gemini_reply, user_input, gemini_history))
        # Await all results (they run in parallel threads) – each returns (reply, duration)
        (oa_reply, oa_time), (cl_reply, cl_time), (gm_reply, gm_time) = await asyncio.gather(oa_task, cl_task, gm_task)
        # Update histories after successful calls
        if not oa_reply.startswith("[Error"):
            openai_history.append({"role": "user", "content": user_input})
            openai_history.append({"role": "assistant", "content": oa_reply})
        if not cl_reply.startswith("[Error"):
            claude_history.append({"role": "user", "content": user_input})
            claude_history.append({"role": "assistant", "content": cl_reply})
        if not gm_reply.startswith("[Error"):
            gemini_history.append({"role": "user", "content": user_input})
            gemini_history.append({"role": "assistant", "content": gm_reply})
        print("\n--- Model Responses ---")
        print(f"OpenAI   : {oa_reply} (took {oa_time:.2f}s)")
        print(f"Claude   : {cl_reply} (took {cl_time:.2f}s)")
        print(f"Gemini  : {gm_reply} (took {gm_time:.2f}s)\n")

if __name__ == "__main__":
    asyncio.run(main_async())
