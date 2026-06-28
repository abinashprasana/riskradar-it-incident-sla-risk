import pandas as pd

DATE_COLS = ["opened_at", "resolved_at", "closed_at", "sys_updated_at", "sys_created_at"]


def load_event_log(csv_path_or_file) -> pd.DataFrame:
    df = pd.read_csv(csv_path_or_file)

    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    if "made_sla" in df.columns:
        s = df["made_sla"].astype(str).str.strip().str.lower()
        df["made_sla"] = s.map({"true": True, "false": False}).where(s.isin(["true", "false"]), df["made_sla"])

    return df


def build_incident_summary(df_events: pd.DataFrame) -> pd.DataFrame:
    if "number" not in df_events.columns:
        raise ValueError("Expected a 'number' column for incident ID but it was not found.")

    g = df_events.groupby("number", dropna=False)
    out = pd.DataFrame(index=g.size().index)
    out.index.name = "number"

    out["total_events"] = g.size().astype(int)

    if "sys_mod_count" in df_events.columns:
        out["sys_mod_count_max"]  = g["sys_mod_count"].max()
        out["sys_mod_count_mean"] = g["sys_mod_count"].mean()

    if "reassignment_count" in df_events.columns:
        out["reassignment_count_max"] = g["reassignment_count"].max()

    if "reopen_count" in df_events.columns:
        out["reopen_count_max"] = g["reopen_count"].max()

    for col in ["category", "subcategory", "priority", "assignment_group", "caller_id", "opened_by", "location"]:
        if col in df_events.columns:
            out[col] = g[col].agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else None)

    if "opened_at" in df_events.columns:
        out["opened_at_min"] = g["opened_at"].min()

    if "resolved_at" in df_events.columns:
        out["resolved_at_max"] = g["resolved_at"].max()

    if "opened_at" in df_events.columns and "resolved_at" in df_events.columns:
        out["resolution_hours"] = (out["resolved_at_max"] - out["opened_at_min"]).dt.total_seconds() / 3600.0

    if "made_sla" in df_events.columns:
        out["made_sla"] = g["made_sla"].agg(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else None)
        out["sla_breached"] = out["made_sla"].map(lambda v: 0 if v is True else (1 if v is False else None))
        out = out.dropna(subset=["sla_breached"])

    return out.reset_index()
