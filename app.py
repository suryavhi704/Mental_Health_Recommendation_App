"""
Mental Health Score Predictor — Streamlit App
================================================
A polished, production-style front end for the Mental Health Score
Prediction model (Random Forest pipeline), plus an AI Wellness Coach
powered by Groq's free LLM API that turns the prediction into
personalized, actionable suggestions and supports follow-up chat.

Run:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

from utils import (
    load_model, get_categories, activity_level, usage_level,
    score_band, build_input_frame, get_feature_importance, DATASET_AVERAGES,
)
from groq_client import get_ai_response, AVAILABLE_MODELS

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="MindScore AI | Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom CSS — the "OP level UI" layer
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 45%, #EC4899 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(99,102,241,0.35);
}
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0; }
.hero p { font-size: 1.02rem; opacity: 0.92; margin-top: 0.4rem; }

.card {
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(120,120,120,0.15);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.section-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.metric-box {
    border-radius: 14px;
    padding: 0.9rem 1rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(236,72,153,0.10));
    border: 1px solid rgba(120,120,120,0.12);
}
.metric-box .val { font-size: 1.5rem; font-weight: 800; }
.metric-box .lbl { font-size: 0.8rem; opacity: 0.75; }

.badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    color: white;
}

.stButton>button {
    border-radius: 12px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    border: none;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white;
    transition: transform 0.15s ease;
}
.stButton>button:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 MindScore AI")
    st.caption("Student Mental Health Score Predictor")
    st.divider()

    st.markdown("### 🔑 AI Coach Settings")
    groq_secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

    if groq_secret_key:
        groq_api_key = groq_secret_key
        st.success("🔒 API key loaded securely from secrets — nothing shown or stored in the page.")
    else:
        groq_api_key = st.text_input(
            "Groq API Key", type="password",
            help="Free key from console.groq.com/keys. Not saved anywhere — used only for this session's API calls.",
        )
        st.caption("⚠️ Tip: for real use, add your key to `.streamlit/secrets.toml` instead of pasting it here — it's more private and you won't need to re-enter it.")
    groq_model_default = os.environ.get("GROQ_MODEL", AVAILABLE_MODELS[0])
    default_idx = AVAILABLE_MODELS.index(groq_model_default) if groq_model_default in AVAILABLE_MODELS else 0
    groq_model = st.selectbox("Groq model", AVAILABLE_MODELS, index=default_idx)

    st.divider()
    with st.expander("ℹ️ About this app"):
        st.write(
            "Predicts a student's Mental Health Score (0-10) from study, "
            "sleep, screen-time, activity and stress data using a trained "
            "Random Forest pipeline, then uses Groq's LLM API to turn that "
            "prediction into personalized wellness suggestions."
        )
    if st.button("🔄 Reset session", use_container_width=True):
        st.session_state.prediction = None
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🧠 MindScore AI</h1>
  <p>Predict a student's Mental Health Score from lifestyle & social media habits —
  then get an AI Wellness Coach's personalized take on your results.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
pipeline = load_model()
if pipeline is None:
    st.error(
        "⚠️ Couldn't find **Mental_Health_Model.pkl** in the app directory.\n\n"
        "Run the training notebook (or `train_model.py`) first so that file "
        "exists next to `app.py`, then reload this page."
    )
    st.stop()

categories = get_categories(pipeline)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_predict, tab_coach, tab_insights = st.tabs(
    ["🎯 Predict My Score", "💬 AI Wellness Coach", "📊 Model Insights"]
)

# ========================================================================
# TAB 1 — PREDICT
# ========================================================================
with tab_predict:
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="section-title">📚 Academic & Lifestyle</div>', unsafe_allow_html=True)
            study_hours = st.slider("Study Hours / day", 0.0, 12.0, 3.0, 0.1)
            age = st.slider("Age", 15, 35, 21)
            sleep_hours = st.slider("Sleep Hours / night", 0.0, 12.0, 7.0, 0.1)
            physical_activity_hours = st.slider("Physical Activity Hours / day", 0.0, 6.0, 1.7, 0.1)

        with c2:
            st.markdown('<div class="section-title">📱 Digital Habits</div>', unsafe_allow_html=True)
            avg_daily_usage_hours = st.slider("Avg Daily Social Media Usage (hrs)", 0.0, 14.0, 5.0, 0.1)
            daily_unlocks = st.slider("Daily Phone Unlocks", 0, 400, 170, 5)
            most_used_platform = st.selectbox("Most Used Platform", categories["Most_Used_Platform"])
            purpose_of_use = st.selectbox("Primary Purpose of Use", categories["Purpose_Of_Use"])

        with c3:
            st.markdown('<div class="section-title">🌍 Background & Wellbeing</div>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", categories["Gender"])
            academic_level = st.selectbox("Academic Level", categories["Academic_Level"])
            grouped_country = st.selectbox("Country", categories["Grouped_country"])
            stress_level = st.select_slider(
                "Stress Level", options=categories["Stress_Level"],
                value=categories["Stress_Level"][min(1, len(categories["Stress_Level"]) - 1)],
            )

        submitted = st.form_submit_button("🔮 Predict My Mental Health Score", use_container_width=True)

    if submitted:
        input_values = {
            "Study_Hours": study_hours,
            "Age": age,
            "Avg_Daily_Usage_Hours": avg_daily_usage_hours,
            "Daily_Unlocks": daily_unlocks,
            "Physical_Activity_Hours": physical_activity_hours,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Stress_Level": stress_level,
            "Gender": gender,
            "Academic_Level": academic_level,
            "Most_Used_Platform": most_used_platform,
            "Purpose_Of_Use": purpose_of_use,
            "Grouped_country": grouped_country,
        }
        X = build_input_frame(input_values)
        with st.spinner("Running the model..."):
            score = float(pipeline.predict(X)[0])
            score = max(0.0, min(10.0, score))

        st.session_state.prediction = {**input_values, "score": score}
        st.session_state.chat_history = []  # reset coach chat for the new prediction

        if score >= 8:
            st.balloons()

    pred = st.session_state.prediction
    if pred:
        st.divider()
        label, color = score_band(pred["score"])

        left, right = st.columns([1, 1.3])

        with left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(pred["score"], 1),
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 4.5], "color": "#FEE2E2"},
                        {"range": [4.5, 6], "color": "#FEF3C7"},
                        {"range": [6, 8], "color": "#D1FAE5"},
                        {"range": [8, 10], "color": "#A7F3D0"},
                    ],
                },
                number={"suffix": " / 10", "font": {"size": 40}},
            ))
            fig.update_layout(height=280, margin=dict(t=20, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div style="text-align:center;"><span class="badge" style="background:{color};">{label}</span></div>',
                unsafe_allow_html=True,
            )

        with right:
            st.markdown('<div class="section-title">📈 You vs. the Average Student</div>', unsafe_allow_html=True)
            compare_fields = [
                ("Study_Hours", "Study hrs/day"),
                ("Sleep_Hours_Per_Night", "Sleep hrs/night"),
                ("Physical_Activity_Hours", "Activity hrs/day"),
                ("Avg_Daily_Usage_Hours", "Screen time hrs/day"),
            ]
            you_vals = [pred[k] for k, _ in compare_fields]
            avg_vals = [DATASET_AVERAGES[k] for k, _ in compare_fields]
            labels = [lbl for _, lbl in compare_fields]

            cfig = go.Figure()
            cfig.add_trace(go.Bar(name="You", x=labels, y=you_vals, marker_color="#6366F1"))
            cfig.add_trace(go.Bar(name="Avg. Student", x=labels, y=avg_vals, marker_color="#D1D5DB"))
            cfig.update_layout(barmode="group", height=280, margin=dict(t=10, b=10, l=10, r=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(cfig, use_container_width=True)

        st.markdown("#### 🤖 Want personalized suggestions?")
        st.caption("Head to the **AI Wellness Coach** tab to get an AI-generated take on your results.")

        report = (
            f"MindScore AI — Prediction Report\n"
            f"---------------------------------\n"
            f"Predicted Mental Health Score: {pred['score']:.1f} / 10 ({label})\n\n"
            + "\n".join(f"{k}: {v}" for k, v in pred.items() if k != "score")
        )
        st.download_button("⬇️ Download Report", report, file_name="mindscore_report.txt", use_container_width=False)

# ========================================================================
# TAB 2 — AI WELLNESS COACH (Groq-powered innovation)
# ========================================================================
with tab_coach:
    st.markdown('<div class="section-title">💬 AI Wellness Coach</div>', unsafe_allow_html=True)
    pred = st.session_state.prediction

    if not pred:
        st.info("👈 Get a prediction in the **Predict My Score** tab first — the coach uses your results to personalize its advice.")
    elif not groq_api_key:
        st.warning("🔑 Add your free Groq API key in the sidebar to activate the AI Wellness Coach. Get one at console.groq.com/keys")
    else:
        profile = {
            "predicted_score": pred["score"],
            "daily_mobile_usage_hours": pred["Avg_Daily_Usage_Hours"],
            "physical_activity_hours": pred["Physical_Activity_Hours"],
            "physical_activity_level": activity_level(pred["Physical_Activity_Hours"]),
            "sleep_hours": pred["Sleep_Hours_Per_Night"],
            "stress_level": pred["Stress_Level"],
            "study_hours": pred["Study_Hours"],
            "age": pred["Age"],
        }

        if not st.session_state.chat_history:
            if st.button("✨ Generate My Personalized Suggestions", use_container_width=True):
                with st.spinner("Your AI coach is thinking..."):
                    try:
                        reply = get_ai_response(groq_api_key, profile=profile, model=groq_model)
                        st.session_state.chat_history.append(
                            {"role": "user", "content": "(auto) Give me suggestions based on my profile."}
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't reach Groq's API: {e}")

        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(msg["content"])
            elif msg["content"].startswith("(auto)"):
                continue
            else:
                with st.chat_message("user"):
                    st.markdown(msg["content"])

        if st.session_state.chat_history:
            follow_up = st.chat_input("Ask a follow-up question, e.g. 'How do I sleep better?'")
            if follow_up:
                st.session_state.chat_history.append({"role": "user", "content": follow_up})
                with st.spinner("Thinking..."):
                    try:
                        reply = get_ai_response(
                            groq_api_key, chat_history=st.session_state.chat_history, model=groq_model
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't reach Groq's API: {e}")

# ========================================================================
# TAB 3 — MODEL INSIGHTS
# ========================================================================
with tab_insights:
    st.markdown('<div class="section-title">📊 What Drives the Model\'s Predictions</div>', unsafe_allow_html=True)
    labels, values = get_feature_importance(pipeline)
    if labels:
        ifig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#8B5CF6"))
        ifig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10),
                            xaxis_title="Relative importance")
        st.plotly_chart(ifig, use_container_width=True)
        st.caption("Global feature importance from the trained Random Forest — how much each engineered feature contributes to predictions on average, across all students.")
    else:
        st.info("All the predictions and generations have been made based on the following informations.")

    st.divider()
    st.markdown('<div class="section-title">📚 Training Dataset Snapshot</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    snapshot = [
        ("Avg. Score", f"{DATASET_AVERAGES['Mental_Health_Score']:.1f} / 10"),
        ("Avg. Sleep", f"{DATASET_AVERAGES['Sleep_Hours_Per_Night']:.1f} hrs"),
        ("Avg. Screen Time", f"{DATASET_AVERAGES['Avg_Daily_Usage_Hours']:.1f} hrs"),
        ("Avg. Activity", f"{DATASET_AVERAGES['Physical_Activity_Hours']:.1f} hrs"),
    ]
    for col, (lbl, val) in zip(cols, snapshot):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption("Based on the 5,000-student training dataset used to build this model.")