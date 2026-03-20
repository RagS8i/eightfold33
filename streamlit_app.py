"""
streamlit_app.py — Agentic Candidate Evaluator
Full end-to-end Streamlit UI with live Gemini-powered pipeline.
"""

import asyncio
import json
import os
import sys
import time

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic Candidate Evaluator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.verdict-card {
    padding: 1.5rem;
    border-radius: 14px;
    text-align: center;
    margin: 1rem 0;
}

.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.skill-chip-green {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.35);
    color: #34d399;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
}

.skill-chip-red {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.35);
    color: #f87171;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
}

.skill-chip-blue {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #818cf8;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
}

.agent-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

.dot-green { background: #10b981; }
.dot-red   { background: #ef4444; }
.dot-blue  { background: #6366f1; }
.dot-amber { background: #f59e0b; }

div[data-testid="stSidebar"] {
    background: #0f172a;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

VERDICT_CONFIG = {
    "STRONG_HIRE":    {"emoji": "🟢🟢", "color": "#10b981", "bg": "rgba(16,185,129,0.12)"},
    "HIRE":           {"emoji": "🟢",   "color": "#34d399", "bg": "rgba(52,211,153,0.10)"},
    "LEAN_HIRE":      {"emoji": "🟡",   "color": "#f59e0b", "bg": "rgba(245,158,11,0.10)"},
    "LEAN_NO_HIRE":   {"emoji": "🟠",   "color": "#fb923c", "bg": "rgba(251,146,60,0.10)"},
    "NO_HIRE":        {"emoji": "🔴",   "color": "#ef4444", "bg": "rgba(239,68,68,0.10)"},
    "STRONG_NO_HIRE": {"emoji": "🔴🔴", "color": "#b91c1c", "bg": "rgba(185,28,28,0.12)"},
}

SAMPLE_JD = {
    "title": "Senior Backend Engineer",
    "company": "TechCorp",
    "description": "Looking for a Senior Backend Engineer to build scalable services.",
    "required_skills": [
        {"name": "Python", "years": 4}, {"name": "Node.js", "years": 2},
        {"name": "PostgreSQL", "years": 3}, {"name": "Docker", "years": 2},
        {"name": "Kubernetes", "years": 2}, {"name": "AWS", "years": 3},
        {"name": "REST API Design", "years": 3}, {"name": "CI/CD", "years": 2},
    ],
    "preferred_skills": [
        {"name": "Terraform", "years": 1}, {"name": "Redis", "years": 1},
        {"name": "GraphQL", "years": 1}, {"name": "Kafka", "years": 1},
    ],
    "min_years_experience": 5,
}

SAMPLE_CANDIDATE = {
    "name": "Demo Candidate",
    "email": "demo@example.com",
    "years_of_experience": 6,
    "skills": [
        {"name": "Python", "years": 5}, {"name": "NodeJS", "years": 3},
        {"name": "PostgreSQL", "years": 4}, {"name": "Docker", "years": 3},
        {"name": "AWS", "years": 4}, {"name": "REST", "years": 5},
        {"name": "Redis", "years": 2}, {"name": "MongoDB", "years": 2},
        {"name": "FastAPI", "years": 2}, {"name": "Django", "years": 3},
        {"name": "Git", "years": 6}, {"name": "CICD", "years": 3},
        {"name": "Linux", "years": 5},
    ],
    "work_experience": [
        {"company": "BigTech", "role": "Backend Engineer II", "duration": "2022-present",
         "highlights": ["Designed microservices handling 10K RPS", "Led Python-to-Go migration"]},
        {"company": "Startup", "role": "Software Engineer", "duration": "2020-2022",
         "highlights": ["Built order management system", "Implemented Redis caching (40% latency reduction)"]},
    ],
    "certifications": ["AWS Solutions Architect Associate", "CKAD"],
}


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_async(coro):
    loop = get_event_loop()
    return loop.run_until_complete(coro)


def render_verdict_card(verdict: dict):
    v = verdict.get("verdict", "UNKNOWN")
    cfg = VERDICT_CONFIG.get(v, {"emoji": "⚪", "color": "#94a3b8", "bg": "rgba(148,163,184,0.1)"})
    confidence = verdict.get("confidence", 0)

    st.markdown(f"""
    <div class="verdict-card" style="background:{cfg['bg']}; border: 2px solid {cfg['color']}40;">
        <div style="font-size:3rem; margin-bottom:0.3rem;">{cfg['emoji']}</div>
        <div style="font-size:1.8rem; font-weight:800; color:{cfg['color']};">{v.replace('_', ' ')}</div>
        <div style="color:#94a3b8; font-size:0.9rem; margin-top:0.3rem;">
            Confidence: <strong style="color:{cfg['color']}">{confidence:.0%}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_skill_chips(skills: list, chip_class: str):
    html = "".join(f'<span class="{chip_class}">{s}</span>' for s in skills)
    st.markdown(html, unsafe_allow_html=True)


def render_similarity_bar(label: str, value: float, color: str = "#6366f1"):
    pct = max(0.0, min(1.0, value)) * 100
    st.markdown(f"""
    <div style="margin-bottom:0.5rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
            <span style="font-size:0.82rem; color:#94a3b8;">{label}</span>
            <span style="font-size:0.82rem; color:{color}; font-weight:600;">{pct:.0f}%</span>
        </div>
        <div style="background:#1e293b; border-radius:999px; height:7px; overflow:hidden;">
            <div style="width:{pct}%; background:{color}; height:100%; border-radius:999px; transition:width 0.5s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get a free key at aistudio.google.com/apikey",
        placeholder="AIza...",
    )

    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key

    from config import settings
    settings.GEMINI_API_KEY = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY", "")
    )

    if settings.GEMINI_API_KEY:
        st.success("🔑 API key set — live Gemini reasoning enabled")
    else:
        st.error("⚠️ No API key — evaluation will fail. Add your key above.")
        st.markdown("[Get a free Gemini API key →](https://aistudio.google.com/apikey)")

    st.divider()
    st.markdown("### 📖 How it works")
    st.markdown("""
1. **Anonymize** — PII stripped for bias-free evaluation  
2. **Normalize** — Skills mapped to canonical taxonomy  
3. **FAISS** — Vector similarity computed locally  
4. **Skill Graph** — Gaps and extras identified  
5. **AI Debate** — Advocate vs Critic vs Fairness agents  
6. **Verdict** — Judge renders final hiring recommendation  
    """)

    st.divider()
    st.markdown("### 🔗 Links")
    st.markdown("[Get Gemini API Key](https://aistudio.google.com/apikey)")
    st.markdown("[GitHub Repo](https://github.com)")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">🤖 Agentic Candidate Evaluator</h1>', unsafe_allow_html=True)
st.markdown("**Bias-free · Skills-verified · Multi-agent AI debate · Explainable verdicts**")
st.divider()

# ── Input section ─────────────────────────────────────────────────────────────
tab_jd, tab_candidate, tab_run = st.tabs(["📋 Job Description", "👤 Candidate Profile", "🚀 Run Evaluation"])

with tab_jd:
    st.subheader("Job Description")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📥 Load Sample JD", use_container_width=True):
            st.session_state["jd_json"] = json.dumps(SAMPLE_JD, indent=2)

    jd_input_mode = st.radio("Input mode", ["JSON", "Form"], horizontal=True, key="jd_mode")

    if jd_input_mode == "JSON":
        jd_text = st.text_area(
            "Paste JD as JSON",
            value=st.session_state.get("jd_json", json.dumps(SAMPLE_JD, indent=2)),
            height=380,
            key="jd_text_area",
        )
        try:
            jd_data = json.loads(jd_text)
            st.success(f"✅ Valid JSON — {len(jd_data.get('required_skills', []))} required skills detected")
            st.session_state["jd_data"] = jd_data
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.session_state["jd_data"] = SAMPLE_JD
    else:
        with st.form("jd_form"):
            title = st.text_input("Job Title", value="Senior Backend Engineer")
            company = st.text_input("Company", value="TechCorp")
            skills_raw = st.text_area(
                "Required Skills (one per line, optionally: SkillName,years)",
                value="Python,4\nNode.js,2\nPostgreSQL,3\nDocker,2\nKubernetes,2\nAWS,3",
                height=180,
            )
            pref_raw = st.text_area(
                "Preferred Skills (one per line)",
                value="Terraform,1\nRedis,1\nGraphQL,1\nKafka,1",
                height=100,
            )
            min_exp = st.number_input("Min Years Experience", min_value=0, value=5)
            submitted = st.form_submit_button("Save JD")
            if submitted:
                required = []
                for line in skills_raw.strip().splitlines():
                    parts = line.strip().split(",")
                    required.append({"name": parts[0].strip(), "years": int(parts[1]) if len(parts) > 1 else 0})
                preferred = []
                for line in pref_raw.strip().splitlines():
                    parts = line.strip().split(",")
                    preferred.append({"name": parts[0].strip(), "years": int(parts[1]) if len(parts) > 1 else 0})
                st.session_state["jd_data"] = {
                    "title": title, "company": company,
                    "required_skills": required, "preferred_skills": preferred,
                    "min_years_experience": min_exp,
                }
                st.success("✅ JD saved!")

with tab_candidate:
    st.subheader("Candidate Profile")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📥 Load Sample Candidate", use_container_width=True):
            st.session_state["candidate_json"] = json.dumps(SAMPLE_CANDIDATE, indent=2)

    cand_input_mode = st.radio("Input mode", ["JSON", "Form"], horizontal=True, key="cand_mode")

    if cand_input_mode == "JSON":
        cand_text = st.text_area(
            "Paste Candidate Profile as JSON",
            value=st.session_state.get("candidate_json", json.dumps(SAMPLE_CANDIDATE, indent=2)),
            height=380,
            key="cand_text_area",
        )
        try:
            cand_data = json.loads(cand_text)
            skills_count = len(cand_data.get("skills", []))
            st.success(f"✅ Valid JSON — {skills_count} skills detected")
            st.session_state["cand_data"] = cand_data
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.session_state["cand_data"] = SAMPLE_CANDIDATE
    else:
        with st.form("cand_form"):
            exp_years = st.number_input("Years of Experience", min_value=0, value=6)
            skills_raw = st.text_area(
                "Skills (one per line: SkillName,years)",
                value="Python,5\nNodeJS,3\nPostgreSQL,4\nDocker,3\nAWS,4\nREST,5\nRedis,2\nCICD,3\nLinux,5",
                height=200,
            )
            summary = st.text_area("Professional Summary", value="Backend engineer with 6 years experience.", height=80)
            certs = st.text_input("Certifications (comma-separated)", value="AWS Solutions Architect, CKAD")
            submitted_c = st.form_submit_button("Save Candidate")
            if submitted_c:
                skills = []
                for line in skills_raw.strip().splitlines():
                    parts = line.strip().split(",")
                    skills.append({"name": parts[0].strip(), "years": int(parts[1]) if len(parts) > 1 else 0})
                st.session_state["cand_data"] = {
                    "years_of_experience": exp_years,
                    "skills": skills,
                    "professional_summary": summary,
                    "certifications": [c.strip() for c in certs.split(",")],
                }
                st.success("✅ Candidate saved!")

with tab_run:
    st.subheader("Run Evaluation Pipeline")

    jd_final = st.session_state.get("jd_data", SAMPLE_JD)
    cand_final = st.session_state.get("cand_data", SAMPLE_CANDIDATE)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**JD:** {jd_final.get('title', 'Not set')} @ {jd_final.get('company', 'N/A')}")
        req_count = len(jd_final.get("required_skills", []))
        pref_count = len(jd_final.get("preferred_skills", []))
        st.caption(f"{req_count} required + {pref_count} preferred skills")
    with col_r:
        skill_count = len(cand_final.get("skills", []))
        exp_yrs = cand_final.get("years_of_experience", "?")
        st.markdown(f"**Candidate:** {exp_yrs} years exp · {skill_count} skills")
        st.caption("PII will be anonymized before evaluation")

    st.divider()

    if not settings.GEMINI_API_KEY:
        st.warning("⚠️ Add your Gemini API key in the sidebar before running.")

    run_col, _ = st.columns([1, 2])
    with run_col:
        run_btn = st.button(
            "🚀 Run Full Evaluation",
            type="primary",
            use_container_width=True,
            disabled=not bool(settings.GEMINI_API_KEY),
        )

    if run_btn:
        st.session_state.pop("report", None)

        progress = st.progress(0, text="Initializing pipeline...")
        status_container = st.empty()

        try:
            from core.anonymizer import anonymize
            from core.skill_taxonomy import normalize_list
            from core.vector_engine import get_vector_engine
            from core.skill_graph import build_skill_graph, _extract_jd_skills, _extract_candidate_skills
            from agents.orchestrator import run_debate
            from report.generator import build_report

            # Step 1
            progress.progress(10, "🔒 Anonymizing candidate profile...")
            status_container.info("Stripping PII and bias signals...")
            anonymized = run_async(anonymize(cand_final))

            # Step 2
            progress.progress(25, "🔧 Normalizing skill taxonomy...")
            status_container.info("Mapping skills to canonical taxonomy...")
            jd_skills = normalize_list(_extract_jd_skills(jd_final))
            cand_skills = normalize_list(_extract_candidate_skills(anonymized))

            # Step 3
            progress.progress(40, "🔢 Computing FAISS vector similarities...")
            status_container.info(f"Comparing {len(jd_skills)} JD skills vs {len(cand_skills)} candidate skills...")
            engine = get_vector_engine()
            overall_sim = engine.compute_overall_similarity(jd_skills, cand_skills)
            skill_matches = engine.find_skill_matches(jd_skills, cand_skills)

            # Step 4
            progress.progress(55, "📊 Building weighted skill graph...")
            status_container.info("Analysing experience gaps and extra skills...")
            analysis = build_skill_graph(jd_final, anonymized, skill_matches)

            evidence_bundle = {
                "overall_similarity": round(overall_sim, 4),
                "match_percentage": analysis.match_percentage,
                "skill_matches": [
                    {"jd_skill": m.jd_skill, "candidate_skill": m.candidate_skill, "similarity": m.similarity}
                    for m in analysis.matched_skills
                ],
                "missing_skills": analysis.missing_skills,
                "extra_skills": analysis.extra_skills,
                "experience_analysis": [
                    {
                        "skill": e.skill, "required": e.required_years,
                        "actual": e.actual_years, "meets": e.meets_requirement,
                        "reasoning": e.reasoning,
                    }
                    for e in analysis.experience_comparisons
                ],
                "graph_reasoning": analysis.reasoning,
                "jd_skills": jd_skills,
                "candidate_skills": cand_skills,
            }

            # Step 5
            progress.progress(70, "🗣 Running multi-agent debate...")
            status_container.info("Advocate ↔ Critic ↔ Fairness ↔ Judge (live Gemini reasoning)...")

            debate_result = None
            for attempt in range(3):
                try:
                    debate_result = run_async(run_debate(evidence_bundle))
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        wait = 40 * (attempt + 1)
                        status_container.warning(
                            f"Rate limited by Gemini API. Waiting {wait}s before retry "
                            f"(attempt {attempt + 1}/3)..."
                        )
                        time.sleep(wait)
                        if attempt == 2:
                            raise
                    else:
                        raise

            # Step 6
            progress.progress(90, "📋 Generating report...")
            status_container.info("Compiling final report...")
            report = build_report(jd_final, anonymized, evidence_bundle, debate_result)
            report["_skill_matches_raw"] = [
                {"jd_skill": m.jd_skill, "candidate_skill": m.candidate_skill, "similarity": m.similarity}
                for m in skill_matches
            ]

            progress.progress(100, "✅ Complete!")
            status_container.success("Pipeline complete! Scroll down to see results.")
            st.session_state["report"] = report

        except Exception as e:
            progress.empty()
            status_container.empty()
            err_str = str(e)
            st.error(f"❌ Pipeline failed: {err_str}")
            if "No GEMINI_API_KEY" in err_str or "api_key" in err_str.lower():
                st.warning("Please add a valid Gemini API key in the sidebar to run the evaluation.")
            elif "expired" in err_str.lower() or "invalid" in err_str.lower():
                st.warning("Your API key appears to be invalid or expired. Please check and re-enter it in the sidebar.")
            elif "quota" in err_str.lower() or "429" in err_str:
                st.warning(
                    "Gemini API quota exhausted. Please wait a few minutes and try again, "
                    "or upgrade your Google AI plan at https://ai.google.dev/pricing"
                )
            st.exception(e)


# ── Results ───────────────────────────────────────────────────────────────────

if "report" in st.session_state:
    report = st.session_state["report"]
    verdict = report.get("verdict", {})
    skill_analysis = report.get("skill_analysis", {})
    debate = report.get("debate_summary", {})

    st.divider()
    st.markdown("## 📊 Evaluation Results")

    # ── Top metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        sim_pct = skill_analysis.get("overall_similarity", 0) * 100
        st.metric("Overall Similarity", f"{sim_pct:.1f}%")
    with m2:
        match_pct = skill_analysis.get("match_percentage", 0)
        st.metric("Skill Match", f"{match_pct:.1f}%")
    with m3:
        matched_count = len(skill_analysis.get("matched_skills", []))
        missing_count = len(skill_analysis.get("missing_skills", []))
        st.metric("Matched Skills", f"{matched_count}", delta=f"-{missing_count} missing")
    with m4:
        conf = verdict.get("confidence", 0)
        st.metric("AI Confidence", f"{conf:.0%}")

    st.divider()

    # ── Verdict + Skill columns ──
    left, right = st.columns([1, 2])

    with left:
        st.markdown("### 🏛 AI Verdict")
        render_verdict_card(verdict)

        if verdict.get("reasoning"):
            with st.expander("📖 Full Reasoning"):
                st.write(verdict["reasoning"])

        if verdict.get("key_factors"):
            st.markdown("**Key Factors:**")
            for factor in verdict["key_factors"]:
                st.markdown(f"- {factor}")

        if verdict.get("conditions"):
            st.markdown("**Conditions:**")
            for cond in verdict["conditions"]:
                st.markdown(f"- {cond}")

    with right:
        st.markdown("### 🔍 Skill Analysis")

        render_similarity_bar("Overall Similarity", skill_analysis.get("overall_similarity", 0), "#6366f1")
        render_similarity_bar("Skill Match Rate", skill_analysis.get("match_percentage", 0) / 100, "#10b981")

        matched = skill_analysis.get("matched_skills", [])
        missing = skill_analysis.get("missing_skills", [])
        extra = skill_analysis.get("extra_skills", [])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**✅ Matched ({len(matched)})**")
            render_skill_chips([m["candidate_skill"] for m in matched if m.get("candidate_skill")], "skill-chip-green")
        with c2:
            st.markdown(f"**❌ Missing ({len(missing)})**")
            render_skill_chips(missing, "skill-chip-red")
        with c3:
            st.markdown(f"**➕ Extra ({len(extra)})**")
            render_skill_chips(extra, "skill-chip-blue")

    st.divider()

    # ── Per-skill similarity table ──
    st.markdown("### 📐 Per-Skill Similarity Scores")
    raw_matches = report.get("_skill_matches_raw", [])
    if raw_matches:
        cols = st.columns(min(len(raw_matches), 4))
        for i, match in enumerate(raw_matches):
            col = cols[i % 4]
            sim = match["similarity"]
            color = "#10b981" if sim >= 0.7 else ("#f59e0b" if sim >= 0.5 else "#ef4444")
            cand = match.get("candidate_skill") or "(none)"
            col.markdown(f"""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:0.6rem; margin-bottom:0.5rem;">
                <div style="font-size:0.75rem; color:#94a3b8;">{match['jd_skill']}</div>
                <div style="font-size:0.82rem; color:{color}; font-weight:600;">↔ {cand}</div>
                <div style="font-size:0.72rem; color:{color};">{sim:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Debate transcript ──
    st.markdown("### 🗣 Multi-Agent Debate")
    adv_args = debate.get("advocate", {}).get("arguments", [])
    crit_args = debate.get("critic", {}).get("arguments", [])
    fair_reviews = debate.get("fairness_reviews", [])

    for i, (adv, crit) in enumerate(zip(adv_args, crit_args), 1):
        with st.expander(f"Round {i} — Advocate vs Critic", expanded=(i == 1)):
            col_adv, col_crit = st.columns(2)
            with col_adv:
                st.markdown("**🟢 Advocate**")
                conf_a = adv.get("confidence", 0)
                st.caption(f"Position: HIRE · Confidence: {conf_a:.0%}")
                for arg in adv.get("key_arguments", []):
                    st.markdown(f"- {arg}")
                ev = adv.get("evidence_cited", [])
                if ev:
                    st.caption("Evidence: " + " · ".join(ev[:3]))
            with col_crit:
                st.markdown("**🔴 Critic**")
                conf_c = crit.get("confidence", 0)
                st.caption(f"Position: NO HIRE · Confidence: {conf_c:.0%}")
                for arg in crit.get("key_arguments", []):
                    st.markdown(f"- {arg}")
                risks = crit.get("risks_identified", [])
                if risks:
                    st.caption("Risks: " + " · ".join(risks[:2]))

    if fair_reviews:
        with st.expander("⚖️ Fairness Review"):
            for i, review in enumerate(fair_reviews, 1):
                st.markdown(f"**Round {i}**")
                if review.get("balance_assessment"):
                    st.info(review["balance_assessment"])
                if review.get("fairness_score") is not None:
                    st.metric("Fairness Score", f"{review['fairness_score']:.0%}")
                verified = review.get("verified_claims", [])
                if verified:
                    st.markdown("**Verified claims:**")
                    for c in verified:
                        st.markdown(f"✅ {c}")
                disputed = review.get("disputed_claims", [])
                if disputed:
                    st.markdown("**Disputed claims:**")
                    for c in disputed:
                        st.markdown(f"⚠️ {c}")
                bias_flags = review.get("bias_flags", [])
                if bias_flags:
                    st.markdown("**Bias flags:**")
                    for flag in bias_flags:
                        st.warning(f"🚩 {flag}")

    st.divider()

    # ── Experience analysis ──
    exp_analysis = skill_analysis.get("experience_analysis", [])
    if exp_analysis:
        st.markdown("### 📅 Experience Gap Analysis")
        for ea in exp_analysis:
            meets = ea.get("meets", False)
            icon = "✅" if meets else "❌"
            req = ea.get("required", 0)
            actual = ea.get("actual", 0)
            st.markdown(f"{icon} **{ea.get('skill')}** — Required: {req}yr · Actual: {actual}yr")

    st.divider()

    # ── Downloads ──
    st.markdown("### 💾 Export Results")
    col_d1, col_d2, col_d3 = st.columns(3)

    from report.generator import format_report_markdown
    report_md = format_report_markdown(report)

    with col_d1:
        st.download_button(
            "📄 Download Markdown Report",
            data=report_md,
            file_name="evaluation_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_d2:
        safe_report = {k: v for k, v in report.items() if k not in ("conversation_log",)}
        st.download_button(
            "📦 Download JSON Report",
            data=json.dumps(safe_report, indent=2, default=str),
            file_name="evaluation_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_d3:
        log_entries = report.get("immutable_log", [])
        from agents.conversation_log import ConversationLog
        log = ConversationLog()
        for entry in log_entries:
            log.add_entry(
                agent_name=entry["agent_name"], role=entry["role"],
                round_number=entry["round_number"], content=entry["content"],
                entry_type=entry["entry_type"],
            )
        st.download_button(
            "📜 Download Debate Transcript",
            data=log.export_markdown(),
            file_name="debate_transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # ── Raw JSON expander ──
    with st.expander("🔧 Raw Report JSON"):
        st.json({k: v for k, v in report.items() if k != "conversation_log"})