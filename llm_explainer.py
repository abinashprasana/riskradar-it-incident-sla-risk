from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SYSTEM_PROMPT = (
    "You are an ITSM incident analyst.\n"
    "Explain SLA breach risk using ONLY the provided JSON facts.\n"
    "Do NOT invent numbers or fields. If missing, say not available.\n"
    "Return:\n"
    "- 2 to 4 bullet points: why it's risky\n"
    "- 1 to 2 bullet points: recommended actions\n"
)


def explain_offline(summary: Dict[str, Any]) -> str:
    risk = summary.get("risk_level", "Unknown")
    p = summary.get("sla_breach_probability", None)
    facts = summary.get("key_factors", {})

    reass = facts.get("reassignment_count_max")
    reopen = facts.get("reopen_count_max")
    events = facts.get("total_events")
    res_hrs = facts.get("resolution_hours")

    out = []
    if p is not None:
        out.append(f"- Risk: **{risk}** (prob ≈ {p:.2f})")
    else:
        out.append(f"- Risk: **{risk}**")

    drivers = []
    if reass is not None and reass >= 3:
        drivers.append(f"high reassignment count ({reass})")
    if reopen is not None and reopen >= 1:
        drivers.append(f"reopened ({reopen})")
    if events is not None and events >= 12:
        drivers.append(f"many events ({events})")
    if res_hrs is not None and res_hrs >= 48:
        drivers.append(f"long resolution (~{int(res_hrs)} hrs)")

    if drivers:
        out.append("- Why: " + ", ".join(drivers))
    else:
        out.append("- Why: nothing extreme, but pattern suggests watching the ticket.")

    action = summary.get("recommended_action")
    if action:
        out.append(f"- Next: {action}")

    return "\n".join(out)


def _client() -> Optional["OpenAI"]:
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


def explain(summary: Dict[str, Any], model: str = "gpt-4.1-mini") -> str:
    """
    Uses ChatGPT if API key exists.
    Falls back to offline explanation if not.
    """
    client = _client()
    if client is None:
        return explain_offline(summary)

    user_content = "Incident risk summary (JSON):\n" + json.dumps(summary, indent=2)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()
