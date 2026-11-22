"""
150-Bed Hospital Command Dashboard  |  Streamlit Cloud one-shot app
AI layer: xAI Grok (OpenAI-compatible API at https://api.x.ai/v1)

Secrets (Streamlit Cloud -> App settings -> Secrets):
    XAI_API_KEY    = "xai-..."
    HOSPITAL_TZ    = "Asia/Karachi"      # optional
    ADMIN_PIN      = "1234"              # optional, protects data entry
    MODEL_ANALYSIS = "grok-4"            # optional, deep reasoning model
    MODEL_CHAT     = "grok-4-fast"       # optional, fast/cheap model
(Verify current model names at https://docs.x.ai before deploying.)

DECISION SUPPORT ONLY. Never enter patient-identifiable information.
"""
import json
import os
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Hospital Command Dashboard", page_icon="🏥", layout="wide")


def get_secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


TZ = ZoneInfo(get_secret("HOSPITAL_TZ", "Asia/Karachi"))
SHIFT_START, SHIFT_END = time(8, 0), time(15, 0)
TOTAL_BEDS = 150
STALE_HOURS = 24


def now():
    return datetime.now(TZ)


# ----------------------------------------------------------------------------
# Static metric metadata + defaults
# ----------------------------------------------------------------------------
META = {
    "falls": dict(
        label="Patient falls", better="lower", fmt=lambda v: f"{v:g} / 10",
        definition="Unplanned descents to the floor among inpatients, per the stated denominator "
                   "(international benchmark is per 1,000 patient-days; confirm your denominator).",
    ),
    "pi": dict(
        label="Pressure injury rate", better="lower", fmt=lambda v: f"{v:g} / 100",
        definition="Hospital-acquired pressure injuries (stage 2 or higher) per 100 patients at risk / "
                   "admitted, per local policy.",
    ),
    "hh": dict(
        label="Hand hygiene compliance", better="higher", fmt=lambda v: f"{v:g}%",
        definition="Compliant observed opportunities ÷ total observed opportunities × 100 "
                   "(WHO 5 Moments for Hand Hygiene).",
    ),
    "med": dict(
        label="Medication errors", better="lower", fmt=lambda v: f"{v:g} / 1000",
        definition="Reported medication errors (any stage: prescribing, dispensing, administration) "
                   "per 1,000 patient-days or doses, per local policy.",
    ),
    "ratio": dict(
        label="Nurse-to-patient ratio", better="lower", fmt=lambda v: f"1:{v:g}",
        definition="Patients assigned per nurse (shown 1:N). Lower N = safer staffing. Reported value "
                   "should reflect nurses actually delivering bedside care per ward.",
    ),
    "cauti": dict(
        label="CAUTI rate", better="lower", fmt=lambda v: f"{v:g} / 10",
        definition="Catheter-associated urinary tract infections per stated denominator "
                   "(standard: per 1,000 catheter-days; confirm).",
    ),
    "clabsi": dict(
        label="CLABSI rate", better="lower", fmt=lambda v: f"{v:g} / 100",
        definition="Central line-associated bloodstream infections per stated denominator "
                   "(standard: per 1,000 central-line-days; confirm).",
    ),
}

# Targets are ILLUSTRATIVE placeholders - replace with your hospital policy.
DEFAULTS = [
    ("falls", 3, 1), ("pi", 2, 1), ("hh", 67, 90), ("med", 8, 5),
    ("ratio", 25, 6), ("cauti", 3, 1), ("clabsi", 2, 1),
]

ROSTER = [
    ("Specialists (SPLTs)", 15), ("Trainees", 25), ("House job doctors", 13), ("Nurses", 60),
]

SUS_ITEMS = [
    "I think that I would like to use this dashboard frequently.",
    "I found the dashboard unnecessarily complex.",
    "I thought the dashboard was easy to use.",
    "I think I would need technical support to be able to use this dashboard.",
    "I found the various functions in this dashboard were well integrated.",
    "I thought there was too much inconsistency in this dashboard.",
    "I would imagine most people would learn to use this dashboard very quickly.",
    "I found the dashboard very cumbersome to use.",
    "I felt very confident using the dashboard.",
    "I needed to learn a lot of things before I could get going with this dashboard.",
]

FEEDBACK_CATEGORIES = ["Staffing / workload", "Equipment / supplies", "Patient safety concern",
                       "Infection control", "Medication process", "Communication / handover",
                       "Dashboard suggestion", "Other"]


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    if "metrics" not in st.session_state:
        t = now().isoformat(timespec="seconds")
        st.session_state.metrics = pd.DataFrame(
            [dict(key=k, value=float(v), target=float(tg), updated=t) for k, v, tg in DEFAULTS]
        )
    if "roster" not in st.session_state:
        st.session_state.roster = pd.DataFrame(ROSTER, columns=["Role", "On duty"])
    st.session_state.setdefault("census", TOTAL_BEDS)
    st.session_state.setdefault("sus", [])
    st.session_state.setdefault("feedback", [])
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("brief", None)


init_state()


# ----------------------------------------------------------------------------
# Logic
# ----------------------------------------------------------------------------
def metric_status(key, value, target):
    """green / amber / red / missing"""
    if value is None or pd.isna(value):
        return "missing"
    better = META[key]["better"]
    if better == "lower":
        if value <= target:
            return "green"
        return "amber" if value <= target * 1.25 else "red"
    if value >= target:
        return "green"
    return "amber" if value >= target * 0.9 else "red"


def parse_ts(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def build_status_table():
    rows = []
    for _, r in st.session_state.metrics.iterrows():
        k = r["key"]
        v = None if pd.isna(r["value"]) else float(r["value"])
        tg = None if pd.isna(r["target"]) else float(r["target"])
        ts = parse_ts(r["updated"]) if isinstance(r["updated"], str) else None
        st_ = metric_status(k, v, tg) if tg is not None else "missing"
        rows.append(dict(key=k, label=META[k]["label"], value=v, target=tg, status=st_,
                         updated=ts, better=META[k]["better"]))
    return rows


def sus_score(answers):
    total = 0
    for i, a in enumerate(answers, start=1):
        total += (a - 1) if i % 2 == 1 else (5 - a)
    return total * 2.5


def missing_warnings(rows):
    w = []
    for r in rows:
        if r["value"] is None:
            w.append(f"**{r['label']}** has no value entered.")
        if r["target"] is None:
            w.append(f"**{r['label']}** has no target set.")
        if r["updated"] is None:
            w.append(f"**{r['label']}** has no update timestamp.")
        elif now() - r["updated"] > timedelta(hours=STALE_HOURS):
            hrs = int((now() - r["updated"]).total_seconds() // 3600)
            w.append(f"**{r['label']}** is stale (last updated {hrs} h ago).")
    if not st.session_state.sus:
        w.append("No **SUS evaluation** responses collected yet.")
    if not st.session_state.feedback:
        w.append("No **nurse feedback** submitted yet.")
    if not get_secret("XAI_API_KEY"):
        w.append("**XAI_API_KEY** not configured - AI features are disabled.")
    nurses = int(st.session_state.roster.loc[st.session_state.roster["Role"] == "Nurses", "On duty"].sum())
    ratio_row = next((r for r in rows if r["key"] == "ratio"), None)
    if nurses and ratio_row and ratio_row["value"]:
        implied = st.session_state.census / nurses
        if abs(implied - ratio_row["value"]) / ratio_row["value"] > 0.5:
            w.append(f"**Data inconsistency:** reported ratio 1:{ratio_row['value']:g} but census "
                     f"{st.session_state.census} ÷ {nurses} nurses on roster implies ~1:{implied:.1f}. "
                     "Verify the ratio definition or the roster.")
    return w


def on_duty_now():
    t = now().time()
    return SHIFT_START <= t < SHIFT_END


def build_context(rows):
    sus = st.session_state.sus
    fb = st.session_state.feedback[-30:]
    return json.dumps(dict(
        hospital=dict(beds=TOTAL_BEDS, census=st.session_state.census,
                      shift=f"{SHIFT_START:%H:%M}-{SHIFT_END:%H:%M}",
                      local_time=now().isoformat(timespec="minutes"),
                      shift_active=on_duty_now()),
        staffing_on_duty=st.session_state.roster.to_dict("records"),
        metrics=[dict(metric=r["label"], value=r["value"], target=r["target"], direction=r["better"] + "_is_better",
                      status=r["status"], last_updated=r["updated"].isoformat(timespec="minutes") if r["updated"] else None)
                 for r in rows],
        missing_data_warnings=missing_warnings(rows),
        sus=dict(n=len(sus), mean=round(sum(s["score"] for s in sus) / len(sus), 1) if sus else None),
        recent_nurse_feedback=[dict(category=f["category"], rating=f["rating"], comment=f["comment"][:300]) for f in fb],
    ), indent=1)


SYSTEM_PROMPT = (
    "You are a hospital operations and patient-safety decision-support assistant for a 150-bed hospital. "
    "You advise charge nurses, matrons and the medical director. You are NOT a clinician and never give "
    "patient-specific treatment advice. Use only the data supplied; if data is missing or inconsistent, say so "
    "explicitly rather than guessing. Be concise, prioritised, and actionable. Always recommend escalation per "
    "local policy for red alerts. Do not request or repeat patient-identifiable information."
)


def grok_client():
    key = get_secret("XAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key, base_url="https://api.x.ai/v1", timeout=180)


def ask_grok(model, messages):
    client = grok_client()
    if client is None:
        return "⚠️ XAI_API_KEY is not set. Add it in Streamlit Cloud → Settings → Secrets."
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content
    except Exception as e:  # surface API problems without crashing the app
        return f"⚠️ Grok API error: {e}"


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.alert-red {background:#fdecea;border-left:6px solid #d32f2f;padding:10px 14px;margin:6px 0;border-radius:6px;color:#7f1d1d;}
.alert-title {font-weight:700;color:#b71c1c;font-size:1.05rem;}
.pill {display:inline-block;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600;color:#fff;}
.green{background:#2e7d32}.amber{background:#ef6c00}.red{background:#c62828}.missing{background:#616161}
.big-target {font-size:1.6rem;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_analysis = st.text_input("Analysis model (deep reasoning)", get_secret("MODEL_ANALYSIS", "grok-4"))
    model_chat = st.text_input("Chat model (fast)", get_secret("MODEL_CHAT", "grok-4-fast"))
    st.caption("Use the reasoning model for shift briefs and the fast model for Q&A. "
               "Check docs.x.ai for the latest model names.")
    st.divider()
    st.metric("Beds", TOTAL_BEDS)
    st.session_state.census = st.number_input("Current census (occupied beds)", 0, TOTAL_BEDS,
                                              int(st.session_state.census))
    st.caption(f"Shift: {SHIFT_START:%H:%M}–{SHIFT_END:%H:%M} · {TZ.key}")
    st.markdown("🟢 **Shift ACTIVE**" if on_duty_now() else "⚪ **Outside shift hours**")
    st.divider()
    st.warning("Do not enter patient names, MRNs or any identifiers anywhere in this app.")

rows = build_status_table()
reds = [r for r in rows if r["status"] == "red"]
valid_ts = [r["updated"] for r in rows if r["updated"]]
last_update = max(valid_ts) if valid_ts else None

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
h1, h2 = st.columns([3, 1])
with h1:
    st.title("🏥 150-Bed Hospital Command Dashboard")
    st.caption("Nursing quality & patient-safety monitoring with Grok AI decision support")
with h2:
    st.metric("Last update", last_update.strftime("%d %b %Y %H:%M") if last_update else "—")
    st.caption(f"Now: {now():%d %b %Y %H:%M} ({TZ.key})")

# Red nursing alerts
if reds:
    for r in reds:
        fmt = META[r["key"]]["fmt"]
        st.markdown(
            f"<div class='alert-red'><span class='alert-title'>🚨 RED NURSING ALERT – {r['label']}</span><br>"
            f"Current <b>{fmt(r['value'])}</b> vs target <b>{fmt(r['target'])}</b> "
            f"({'must be ≤' if r['better']=='lower' else 'must be ≥'} target). Escalate per policy.</div>",
            unsafe_allow_html=True)
else:
    st.success("No red nursing alerts.")

warns = missing_warnings(rows)
if warns:
    with st.expander(f"⚠️ Missing / inconsistent data warnings ({len(warns)})", expanded=bool(warns)):
        for w in warns:
            st.warning(w)

tab_dash, tab_ai, tab_sus, tab_fb, tab_def = st.tabs(
    ["📊 Dashboard", "🤖 Grok AI Command", "📝 SUS Evaluation", "💬 Nurse Feedback", "📖 Definitions & Data"])

# ----------------------------------------------------------------------------
# Dashboard tab
# ----------------------------------------------------------------------------
with tab_dash:
    st.subheader("Quality & safety metrics")
    cols = st.columns(4)
    for i, r in enumerate(rows):
        with cols[i % 4]:
            with st.container(border=True):
                fmt = META[r["key"]]["fmt"]
                st.markdown(f"**{r['label']}**")
                st.markdown(f"<span class='big-target'>{fmt(r['value']) if r['value'] is not None else 'N/A'}</span> "
                            f"<span class='pill {r['status']}'>{r['status'].upper()}</span>", unsafe_allow_html=True)
                st.markdown(f"🎯 **Target at bedside:** {fmt(r['target']) if r['target'] is not None else 'not set'} "
                            f"({'≤' if r['better']=='lower' else '≥'})")
                st.caption(f"Updated: {r['updated']:%d %b %H:%M}" if r["updated"] else "Updated: never")

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Staffing on duty (08:00–15:00)")
        ros = st.session_state.roster
        st.dataframe(ros.assign(Status=("On duty" if on_duty_now() else "Off shift")), hide_index=True,
                     use_container_width=True)
        st.metric("Total clinical staff", int(ros["On duty"].sum()))
        nurses = int(ros.loc[ros["Role"] == "Nurses", "On duty"].sum())
        if nurses:
            st.caption(f"Roster-implied nurse:patient ≈ 1:{st.session_state.census / nurses:.1f} "
                       f"(census {st.session_state.census} ÷ {nurses} nurses)")
    with right:
        st.subheader("Performance vs target")
        chart_rows = []
        for r in rows:
            if r["value"] is not None and r["target"]:
                pct = (r["value"] / r["target"] * 100) if r["better"] == "lower" else (r["target"] / r["value"] * 100 if r["value"] else 0)
                chart_rows.append(dict(Metric=r["label"], **{"% of target (100 = on target, higher = worse)": round(pct)}))
        if chart_rows:
            st.bar_chart(pd.DataFrame(chart_rows).set_index("Metric"))

    st.divider()
    pin_needed = get_secret("ADMIN_PIN")
    with st.expander("✏️ Update metrics, targets & roster (shift lead)"):
        ok = True
        if pin_needed:
            ok = st.text_input("Admin PIN", type="password") == str(pin_needed)
            if not ok:
                st.info("Enter PIN to edit.")
        if ok:
            df = st.session_state.metrics.copy()
            df.insert(1, "metric", df["key"].map(lambda k: META[k]["label"]))
            edited = st.data_editor(
                df, hide_index=True, use_container_width=True, key="metric_editor",
                disabled=["key", "metric", "updated"],
                column_config={"key": None,
                               "value": st.column_config.NumberColumn("Current value", help="Leave blank if unknown"),
                               "target": st.column_config.NumberColumn("Target"),
                               "updated": st.column_config.TextColumn("Last updated")})
            roster_ed = st.data_editor(st.session_state.roster, hide_index=True, disabled=["Role"],
                                       use_container_width=True, key="roster_editor")
            if st.button("💾 Save changes", type="primary"):
                t = now().isoformat(timespec="seconds")
                for i in edited.index:
                    old, new = st.session_state.metrics.loc[i], edited.loc[i]
                    changed = (pd.isna(old["value"]) != pd.isna(new["value"])) or \
                              (not pd.isna(new["value"]) and old["value"] != new["value"])
                    st.session_state.metrics.loc[i, "value"] = new["value"]
                    st.session_state.metrics.loc[i, "target"] = new["target"]
                    if changed:
                        st.session_state.metrics.loc[i, "updated"] = t
                st.session_state.roster = roster_ed
                st.rerun()
            up = st.file_uploader("…or upload CSV (columns: key,value,target)", type="csv")
            if up is not None and st.button("Apply CSV"):
                try:
                    csv = pd.read_csv(up)
                    t = now().isoformat(timespec="seconds")
                    for _, rr in csv.iterrows():
                        m = st.session_state.metrics["key"] == rr["key"]
                        if m.any():
                            st.session_state.metrics.loc[m, ["value", "target", "updated"]] = [rr["value"], rr["target"], t]
                    st.rerun()
                except Exception as e:
                    st.error(f"CSV error: {e}")

# ----------------------------------------------------------------------------
# AI tab
# ----------------------------------------------------------------------------
with tab_ai:
    st.subheader("🤖 Grok shift brief")
    st.caption("Decision support only. Always confirm with clinical leadership and local policy.")
    if st.button("Generate shift brief", type="primary"):
        with st.spinner(f"Asking {model_analysis}…"):
            prompt = (
                "Using the JSON hospital snapshot below, write a shift brief with these sections:\n"
                "1. Situation summary (3 lines)\n2. Red-alert actions (owner role + timeframe)\n"
                "3. Staffing & ratio recommendations (flag any inconsistency)\n"
                "4. Infection prevention actions (CAUTI, CLABSI, hand hygiene)\n"
                "5. Falls, pressure injury & medication safety actions\n"
                "6. Staff feedback themes & SUS insights\n7. Data-quality issues & questions to verify\n\n"
                f"SNAPSHOT:\n{build_context(rows)}")
            st.session_state.brief = ask_grok(model_analysis, [
                {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    if st.session_state.brief:
        st.markdown(st.session_state.brief)
        st.download_button("Download brief", st.session_state.brief, "shift_brief.md")

    st.divider()
    st.subheader("Ask Grok about the hospital")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("e.g. Which wards should get extra nurses this afternoon?")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCURRENT SNAPSHOT:\n" + build_context(rows)}]
        msgs += st.session_state.chat[-10:]
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                ans = ask_grok(model_chat, msgs)
            st.markdown(ans)
        st.session_state.chat.append({"role": "assistant", "content": ans})

# ----------------------------------------------------------------------------
# SUS tab
# ----------------------------------------------------------------------------
with tab_sus:
    st.subheader("System Usability Scale (SUS) – evaluate this dashboard")
    with st.form("sus_form", clear_on_submit=True):
        answers = []
        for i, item in enumerate(SUS_ITEMS, 1):
            answers.append(st.radio(f"{i}. {item}", [1, 2, 3, 4, 5], index=2, horizontal=True,
                                    help="1 = strongly disagree … 5 = strongly agree", key=f"sus_{i}"))
        role = st.selectbox("Your role (optional)", ["—", "Nurse", "House job doctor", "Trainee", "Specialist", "Manager"])
        if st.form_submit_button("Submit SUS"):
            st.session_state.sus.append(dict(time=now().isoformat(timespec="seconds"), role=role,
                                             score=sus_score(answers), answers=answers))
            st.success("Thank you – response recorded.")
    sus = st.session_state.sus
    if sus:
        mean = sum(s["score"] for s in sus) / len(sus)
        c1, c2 = st.columns(2)
        c1.metric("Mean SUS score", f"{mean:.1f}", help="68 = average; ≥80.3 = excellent")
        c2.metric("Responses", len(sus))
        grade = "Excellent" if mean >= 80.3 else "Good" if mean >= 68 else "Marginal" if mean >= 51 else "Poor"
        st.progress(min(int(mean), 100) / 100, text=f"Usability: {grade}")
        sdf = pd.DataFrame(sus).drop(columns="answers")
        st.dataframe(sdf, hide_index=True, use_container_width=True)
        st.download_button("Download SUS CSV", sdf.to_csv(index=False), "sus_responses.csv")
    else:
        st.info("No SUS responses yet.")

# ----------------------------------------------------------------------------
# Feedback tab
# ----------------------------------------------------------------------------
with tab_fb:
    st.subheader("Nurse feedback (anonymous)")
    st.caption("Do not include patient names or identifiers.")
    with st.form("fb_form", clear_on_submit=True):
        cat = st.selectbox("Category", FEEDBACK_CATEGORIES)
        rating = st.slider("How manageable was your shift? (1 = overwhelming, 5 = very manageable)", 1, 5, 3)
        comment = st.text_area("Comment", max_chars=600)
        if st.form_submit_button("Submit feedback"):
            if comment.strip():
                st.session_state.feedback.append(dict(time=now().isoformat(timespec="seconds"),
                                                      category=cat, rating=rating, comment=comment.strip()))
                st.success("Thank you – feedback recorded.")
            else:
                st.error("Please write a comment.")
    fb = st.session_state.feedback
    if fb:
        fdf = pd.DataFrame(fb)
        st.metric("Average shift manageability", f"{fdf['rating'].mean():.1f} / 5")
        st.dataframe(fdf, hide_index=True, use_container_width=True)
        st.download_button("Download feedback CSV", fdf.to_csv(index=False), "nurse_feedback.csv")
        if st.button("🤖 Summarise themes with Grok"):
            with st.spinner("Analysing feedback…"):
                st.markdown(ask_grok(model_chat, [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Summarise the main themes, urgent safety concerns and 3 practical "
                     "improvements from this anonymous nurse feedback:\n" + fdf.to_json(orient="records")}]))
    else:
        st.info("No feedback yet.")
    st.caption("Note: Streamlit Cloud storage is ephemeral. Download CSVs regularly or connect a database "
               "(e.g. Google Sheets / Supabase) for permanent storage.")

# ----------------------------------------------------------------------------
# Definitions tab
# ----------------------------------------------------------------------------
with tab_def:
    st.subheader("Metric definitions")
    for k, m in META.items():
        row = st.session_state.metrics[st.session_state.metrics["key"] == k].iloc[0]
        with st.container(border=True):
            st.markdown(f"**{m['label']}** — target {m['fmt'](row['target'])} "
                        f"({'lower' if m['better']=='lower' else 'higher'} is better)")
            st.write(m["definition"])
    st.info("Targets are illustrative placeholders. Replace them with your hospital's approved policy targets.")
    st.markdown("**Status rules:** 🟢 meets target · 🟠 within 25% (lower-better) / 10% (higher-better) of target · "
                "🔴 beyond that (red nursing alert) · ⚪ missing data.")
