from __future__ import annotations


def risk_band(p: float) -> str:
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"


def recommended_action(band: str) -> str:
    b = band.lower()
    if b == "high":
        return "Escalate now. Assign correctly, reduce reassignment loops, senior review."
    if b == "medium":
        return "Monitor. Check missing details and confirm ownership early."
    return "Normal queue. Keep updates clean and avoid unnecessary reassignment."