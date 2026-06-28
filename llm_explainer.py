import os
import json
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

SYSTEM_PROMPT = (
    "You are an ITSM incident analyst.\n"
    "Explain SLA breach risk using ONLY the provided JSON facts.\n"
    "Do NOT invent numbers or fields. If a value is missing, say it is not available.\n"
    "Return:\n"
    "- 2 to 4 bullet points explaining why this ticket is risky\n"
    "- 1 to 2 bullet points on recommended next actions\n"
)


def explain_offline(summary: dict[str, Any]) -> str:
    risk  = summary.get("risk_level", "Unknown")
    p     = summary.get("sla_breach_probability")
    facts = summary.get("key_factors", {})

    lines = [f"- Risk: **{risk}**" + (f" (prob {p:.2f})" if p is not None else "")]

    drivers = []
    if (v := facts.get("reassignment_count_max")) is not None and v >= 3:
        drivers.append(f"high reassignment count ({v})")
    if (v := facts.get("reopen_count_max")) is not None and v >= 1:
        drivers.append(f"reopened {v} time(s)")
    if (v := facts.get("total_events")) is not None and v >= 12:
        drivers.append(f"many events ({v})")
    if (v := facts.get("resolution_hours")) is not None and v >= 48:
        drivers.append(f"long resolution time (~{int(v)} hrs)")

    lines.append("- Why: " + (", ".join(drivers) if drivers else "no extreme signals, but pattern suggests monitoring."))

    if action := summary.get("recommended_action"):
        lines.append(f"- Next: {action}")

    return "\n".join(lines)


def _client() -> "OpenAI | None":
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


def explain(summary: dict[str, Any], model: str = "gpt-4.1-mini") -> str:
    client = _client()
    if client is None:
        return explain_offline(summary)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Incident risk summary (JSON):\n" + json.dumps(summary, indent=2)},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
