import os
import re
import traceback
from datetime import datetime

import streamlit as st

# On Streamlit Cloud, copy secrets into environment variables so the existing code keeps working.
# This must run before importing the pipeline, because agents.py creates the Groq client at import time.
for _key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
    try:
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = st.secrets[_key]
    except Exception:
        # No secrets file locally: fall back to the .env file loaded by python-dotenv
        pass

from pipeline import run_research_pipeline

st.set_page_config(
    page_title="Research Agent Studio",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px;}

    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #6d28d9 60%, #db2777 100%);
        padding: 2rem 2.2rem; border-radius: 18px; color: white;
        margin-bottom: 1.5rem; box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }
    .hero h1 {margin: 0; font-size: 2.2rem; color: white; font-weight: 800;}
    .hero p {margin: .45rem 0 0 0; opacity: .92; font-size: 1.02rem;}

    .card {
        background: rgba(128,128,128,0.07);
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;
    }
    .card-title {font-weight: 700; font-size: .85rem; letter-spacing: .06em;
        text-transform: uppercase; opacity: .7; margin-bottom: .5rem;}

    .agent-row {display: flex; gap: .8rem; flex-wrap: wrap;}
    .agent {
        flex: 1 1 200px; border-radius: 12px; padding: .9rem 1rem;
        background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.25);
    }
    .agent .num {font-size: .75rem; font-weight: 700; color: #7c3aed; letter-spacing: .05em;}
    .agent .name {font-weight: 700; margin: .2rem 0;}
    .agent .desc {font-size: .85rem; opacity: .75;}

    .score-badge {
        display: inline-block; font-size: 2.4rem; font-weight: 800;
        line-height: 1; padding: .6rem 1rem; border-radius: 14px; color: white;
    }
    .score-good {background: #059669;}
    .score-mid {background: #d97706;}
    .score-low {background: #dc2626;}

    .stat {font-size: 1.6rem; font-weight: 800; line-height: 1.1;}
    .stat-label {font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em;}

    div[data-testid="stTabs"] button {font-weight: 600;}
    .source-item {padding: .45rem 0; border-bottom: 1px dashed rgba(128,128,128,.25); word-break: break-all;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EXAMPLE_TOPICS = [
    "Recent developments in AI regulation",
    "Latest advances in solid-state batteries",
    "Impact of AI on software engineering jobs",
]

# ---------- Session state ----------
st.session_state.setdefault("history", [])
st.session_state.setdefault("current", None)
st.session_state.setdefault("topic_input", "")


def set_topic(value: str):
    st.session_state["topic_input"] = value


def extract_score(feedback: str):
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback)
    return float(match.group(1)) if match else None


def extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]>\"']+", text or "")
    seen, unique = set(), []
    for u in urls:
        u = u.rstrip(".,;")
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def score_class(score):
    if score is None:
        return "score-mid"
    if score >= 8:
        return "score-good"
    if score >= 5:
        return "score-mid"
    return "score-low"


def build_markdown_bundle(run: dict) -> str:
    return (
        f"# Research Report: {run['topic']}\n\n"
        f"_Generated: {run['time']}_\n\n"
        f"{run['report']}\n\n---\n\n"
        f"## Critic Feedback\n\n{run['feedback']}\n"
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🔎 Research Studio")
    st.caption("Multi-agent research powered by Groq, LangChain and Tavily")

    st.markdown("**Try an example**")
    for i, example in enumerate(EXAMPLE_TOPICS):
        st.button(example, key=f"ex_{i}", on_click=set_topic, args=(example,), use_container_width=True)

    st.divider()
    st.markdown("**Past runs**")
    if not st.session_state.history:
        st.caption("Your completed research runs will appear here.")
    else:
        for idx, run in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - idx
            label = f"{run['topic'][:38]}{'...' if len(run['topic']) > 38 else ''}"
            if st.button(label, key=f"hist_{real_idx}", use_container_width=True):
                st.session_state["current"] = real_idx
        if st.button("Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["current"] = None
            st.rerun()

# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
    <h1>Research Agent Studio</h1>
    <p>Give it a topic. Agents search the web, read the best source, write a structured report, and a critic scores it.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Pipeline overview ----------
st.markdown(
    """
<div class="agent-row">
    <div class="agent"><div class="num">STEP 01</div><div class="name">🔍 Search Agent</div><div class="desc">Finds recent, reliable sources with Tavily</div></div>
    <div class="agent"><div class="num">STEP 02</div><div class="name">📖 Reader Agent</div><div class="desc">Scrapes the most relevant page with BeautifulSoup</div></div>
    <div class="agent"><div class="num">STEP 03</div><div class="name">✍️ Writer Chain</div><div class="desc">Drafts a structured, sourced report</div></div>
    <div class="agent"><div class="num">STEP 04</div><div class="name">🧐 Critic Chain</div><div class="desc">Scores the report and lists improvements</div></div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ---------- Input ----------
with st.container(border=True):
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Recent developments in AI regulation",
    )
    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("🚀 Run research", type="primary", use_container_width=True)
    with col_hint:
        st.caption("A full run usually takes 1 to 3 minutes, because each agent makes several LLM and web calls.")

if run_clicked:
    if not topic.strip():
        st.warning("Pehle ek topic enter karo.")
    else:
        clean_topic = topic.strip()
        started = datetime.now()
        try:
            with st.status("Agents are working on your topic...", expanded=True) as status:
                st.write("🔍 Search agent: finding sources")
                st.write("📖 Reader agent: scraping the best page")
                st.write("✍️ Writer chain: drafting the report")
                st.write("🧐 Critic chain: reviewing the report")
                result = run_research_pipeline(clean_topic)
                status.update(label="Research complete", state="complete", expanded=False)

            elapsed = (datetime.now() - started).seconds
            run = {
                "topic": clean_topic,
                "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                "elapsed": elapsed,
                "report": result["report"],
                "feedback": result["feedback"],
                "search_results": result["search_results"],
                "scraped_content": result["scraped_content"],
                "score": extract_score(result["feedback"]),
            }
            st.session_state["history"].append(run)
            st.session_state["current"] = len(st.session_state["history"]) - 1
            st.toast("Report ready!", icon="✅")
        except Exception as e:
            st.error(f"Pipeline fail ho gaya: {e}")
            # Show the full traceback so the exact failing file and line is visible
            with st.expander("Error details (traceback)", expanded=True):
                st.code(traceback.format_exc(), language="text")

# ---------- Results ----------
current_idx = st.session_state.get("current")
if current_idx is not None and current_idx < len(st.session_state["history"]):
    run = st.session_state["history"][current_idx]
    report_words = len(run["report"].split())
    sources = extract_urls(run["search_results"]) + extract_urls(run["scraped_content"])
    sources = list(dict.fromkeys(sources))

    st.write("")
    st.markdown(f"### 📑 {run['topic']}")
    st.caption(f"Generated {run['time']} · took about {run['elapsed']}s")

    # Stat cards
    s1, s2, s3 = st.columns(3)
    with s1:
        score_text = f"{run['score']:g}/10" if run["score"] is not None else "N/A"
        st.markdown(
            f'<div class="card"><div class="card-title">Critic score</div>'
            f'<span class="score-badge {score_class(run["score"])}">{score_text}</span></div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f'<div class="card"><div class="card-title">Report length</div>'
            f'<div class="stat">{report_words:,}</div><div class="stat-label">words</div></div>',
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f'<div class="card"><div class="card-title">Sources found</div>'
            f'<div class="stat">{len(sources)}</div><div class="stat-label">unique URLs</div></div>',
            unsafe_allow_html=True,
        )

    tab_report, tab_critic, tab_sources, tab_raw = st.tabs(
        ["📄 Report", "🧐 Critic feedback", "🔗 Sources", "🗂️ Raw data"]
    )

    with tab_report:
        with st.container(border=True):
            st.markdown(run["report"])
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Download report (.md)",
                data=run["report"],
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Download full bundle (.md)",
                data=build_markdown_bundle(run),
                file_name="research_bundle.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with tab_critic:
        with st.container(border=True):
            st.markdown(run["feedback"])

    with tab_sources:
        if sources:
            with st.container(border=True):
                for url in sources:
                    st.markdown(f'<div class="source-item">🔗 <a href="{url}" target="_blank">{url}</a></div>', unsafe_allow_html=True)
        else:
            st.info("Koi URL nahi mila.")

    with tab_raw:
        with st.expander("Search results", expanded=False):
            st.code(run["search_results"], language="text")
        with st.expander("Scraped content", expanded=False):
            st.code(run["scraped_content"], language="text")
else:
    st.info("Upar topic enter karo ya sidebar se example choose karo, phir **Run research** dabao.")