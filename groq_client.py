"""
groq_client.py
Thin wrapper around Groq's free, OpenAI-compatible chat completion API.

Get a free API key at: https://console.groq.com/keys

This module is the "innovation" layer of the app: it takes the model's
numeric prediction plus a handful of the user's lifestyle inputs, drops
them into a custom prompt template, and asks a Groq-hosted LLM to turn
that into a short, personalized, actionable wellness suggestion. The
same function also powers a lightweight follow-up chatbot by accepting
an existing chat history.
"""
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
]

SYSTEM_PROMPT = """You are MindMate, a warm, encouraging student-wellness coach embedded
inside a "Mental Health Score Predictor" app. A machine learning model has already
estimated the student's Mental Health Score (0-10) from their lifestyle data; your job
is to translate that number plus their habits into short, specific, actionable advice.

Rules you always follow:
- You are NOT a therapist or doctor. Never diagnose a condition or use clinical labels.
- Be warm, non-judgmental, and specific — reference the student's own numbers instead of
  generic advice ("you're sleeping 5.2 hrs, about 90 min under what's typical for your age"
  beats "sleep more").
- Reply with 3-5 short bullet-point suggestions, each tied to one of their actual inputs.
- If stress is High/Very High, or the score is low, gently encourage talking to a
  counselor, trusted adult, or campus mental-health service — without sounding alarming.
- Keep the whole reply under ~180 words unless the student explicitly asks for more detail.
- End on one encouraging, human sentence.
"""


def build_profile_prompt(profile: dict) -> str:
    """
    Builds the custom template combining model prediction + key lifestyle
    inputs, per the assignment: Daily_mobile_usage_hours, Physical_activity_hour,
    sleep_hour, physical_activity_level, stress_level + the model's predicted score.
    """
    return f"""Here is a student's profile, produced by our Mental Health Score
Predictor model:

- Predicted Mental Health Score (model output): {profile['predicted_score']:.1f} / 10
- Daily mobile / social media usage: {profile['daily_mobile_usage_hours']:.1f} hours/day
- Physical activity: {profile['physical_activity_hours']:.1f} hours/day ({profile['physical_activity_level']} activity level)
- Sleep: {profile['sleep_hours']:.1f} hours/night
- Stress level: {profile['stress_level']}
- Study hours/day: {profile.get('study_hours', 'N/A')}
- Age: {profile.get('age', 'N/A')}

Based on this profile, give the student personalized, practical suggestions to
protect or improve their mental wellbeing this week."""


def get_ai_response(api_key: str, profile: dict = None, chat_history: list = None,
                     model: str = "openai/gpt-oss-120b", temperature: float = 0.6) -> str:
    """
    Calls Groq's chat completion endpoint.

    - First call: pass `profile` (dict) -> builds the custom template prompt.
    - Follow-up calls: pass the running `chat_history` (list of
      {"role": "user"/"assistant", "content": ...}) so the same function
      doubles as a multi-turn wellness chatbot with full context.
    """
    if not api_key:
        raise ValueError("Missing Groq API key.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    elif profile:
        messages.append({"role": "user", "content": build_profile_prompt(profile)})
    else:
        raise ValueError("Provide either `profile` or `chat_history`.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": 700,
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]