"""
utils.py
Helper functions for the Mental Health Score Predictor Streamlit app.

Keeps app.py focused on layout/UX, and keeps all "know the model" logic
(feature order, category values, dataset benchmarks) in one place.
"""
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "Mental_Health_Model.pkl"

# Exact order the pipeline was trained on (from the notebook's X_train.columns)
FEATURE_ORDER = [
    "Study_Hours",
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_country",
]

# Fallback category lists (used only if we can't read them straight out of
# the fitted encoders inside the pipeline — see get_categories() below).
# These mirror the values actually seen in the training notebook's EDA.
FALLBACK_CATEGORIES = {
    "Stress_Level": ["Low", "Medium", "High", "Very High"],
    "Gender": ["Male", "Female", "Other"],
    "Academic_Level": ["High School", "Undergraduate", "Graduate"],
    "Most_Used_Platform": [
        "Instagram", "TikTok", "Facebook", "LinkedIn", "YouTube",
        "Twitter", "Snapchat", "WhatsApp", "LINE", "VKontakte",
        "KakaoTalk", "WeChat",
    ],
    "Purpose_Of_Use": [
        "Entertainment", "Education", "Networking",
        "Social Interaction", "Professional/Career",
    ],
    "Grouped_country": [
        "India", "USA", "Canada", "Australia", "UK", "Germany",
        "Mexico", "Turkey", "France", "Other",
    ],
}

# Real training-set stats (straight from the notebook's df.describe() /
# value_counts() output) — used only to show the user a "you vs. the
# average student in this dataset" comparison. Purely illustrative.
DATASET_AVERAGES = {
    "Study_Hours": 3.01,
    "Age": 20.82,
    "Avg_Daily_Usage_Hours": 5.08,
    "Daily_Unlocks": 171,
    "Physical_Activity_Hours": 1.75,
    "Sleep_Hours_Per_Night": 6.63,
    "Mental_Health_Score": 6.23,
}


@st.cache_resource(show_spinner=False)
def load_model(path: str = MODEL_PATH):
    """Loads the saved sklearn Pipeline (preprocessing + RandomForest)."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def get_categories(pipeline) -> dict:
    """
    Reads the exact category lists the model was actually trained on
    directly out of the fitted OneHotEncoder / OrdinalEncoder objects
    living inside the pipeline's ColumnTransformer step named
    'preprocessor'. Falls back to known notebook values if that fails
    for any reason, so the UI never breaks.
    """
    categories = {k: list(v) for k, v in FALLBACK_CATEGORIES.items()}
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            return categories
        for _name, transformer, cols in preprocessor.transformers_:
            encoder = None
            if hasattr(transformer, "named_steps"):
                encoder = transformer.named_steps.get("encode")
            if encoder is None or not hasattr(encoder, "categories_"):
                continue
            for col, cats in zip(cols, encoder.categories_):
                categories[col] = list(cats)
    except Exception:
        pass
    return categories


def activity_level(hours: float) -> str:
    """Buckets physical activity hours/day into a human label for the AI prompt."""
    if hours < 1:
        return "Low"
    if hours < 2.5:
        return "Moderate"
    return "High"


def usage_level(hours: float) -> str:
    """Buckets daily screen-time hours into a human label for the AI prompt."""
    if hours < 3:
        return "Low"
    if hours < 6:
        return "Moderate"
    return "High"


def score_band(score: float) -> tuple:
    """Returns (label, color) describing where a predicted score sits."""
    if score < 4.5:
        return "Needs Attention", "#EF4444"
    if score < 6:
        return "Fair", "#F59E0B"
    if score < 8:
        return "Good", "#22C55E"
    return "Thriving", "#10B981"


def build_input_frame(values: dict) -> pd.DataFrame:
    """Builds a single-row DataFrame in the exact column order the model expects."""
    return pd.DataFrame([{col: values[col] for col in FEATURE_ORDER}])


def get_feature_importance(pipeline, top_n: int = 10):
    """
    Returns a (labels, importances) tuple for the top_n most influential
    engineered features, using the fitted RandomForest's feature_importances_
    mapped back to human-readable names via ColumnTransformer.get_feature_names_out().
    Returns (None, None) if unavailable (older sklearn / different pipeline shape).
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps.get("random forest") or pipeline.named_steps.get("regressor")
        names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:top_n]
        labels = [p[0].split("__")[-1] for p in pairs]
        values = [p[1] for p in pairs]
        return labels[::-1], values[::-1]
    except Exception:
        return None, None
