"""Checks that the configured LLM key works from this codebase.

Run from the backend/ folder:

    python scripts/check_llm.py            # 1 request: plain completion
    python scripts/check_llm.py --tools    # +1-3 requests: also exercises the tool-calling loop

It uses the same code path the pipeline agents use (app.services.llm_service), reads the key
from backend/.env exactly like the app does, and never prints the key. Free-tier keys have
small daily quotas, so the default is a single tiny request.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.agents import dataset_tools  # noqa: E402
from app.services import llm_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools", action="store_true", help="also test the tool-calling loop")
    args = parser.parse_args()

    config = llm_service._active_config()
    if config is None:
        print("FAIL: no key found. Set GEMINI_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY) in backend/.env")
        return 1
    print(f"Provider: {config['provider']}   Model: {config['model']}   Key: set ({len(config['api_key'])} chars)")

    try:
        reply = llm_service.complete("Reply with exactly one word.", "Say: pong", max_tokens=256)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        print("FAIL: plain completion failed.")
        if llm_service._is_rate_limited(exc):
            print("  -> Quota / rate limit hit (HTTP 429). The key is valid but has no requests left right now.")
            print("     Free tier is ~20 requests/day per model. Wait for the reset, enable billing, or set another model.")
        elif "api key" in text.lower() or "401" in text or "403" in text or "invalid" in text.lower():
            print("  -> The key looks invalid or not permitted for this model.")
        print("  Details:", text.splitlines()[0][:300])
        return 2
    print(f"OK: plain completion -> {reply!r}")

    if not args.tools:
        print("Tip: run with --tools to also verify the tool-calling loop the agents use.")
        return 0

    df = pd.DataFrame({"age": [20, 30, 40, 50] * 5, "churn": ["yes", "no"] * 10})
    try:
        result = llm_service.run_tool_loop(
            "You can call tools. Call column_stats for the column 'age', then answer in one short sentence.",
            "What is the mean of the age column?",
            dataset_tools.build_tools(df, "churn"),
            max_steps=4,
        )
    except Exception as exc:  # noqa: BLE001
        print("FAIL: tool-calling loop failed:", str(exc).splitlines()[0][:300])
        return 3
    used = [c["tool"] for c in result.calls]
    if not used:
        print(f"WARN: model answered without calling a tool: {result.text!r}")
        return 4
    print(f"OK: tool loop -> tools called {used}; answer: {result.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
