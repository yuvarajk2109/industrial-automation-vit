"""
Gemini Integration
    - Conversational action recommender
    - Discusses ONLY 
        - model predictions
        - knowledge graph results
"""

import json
import os
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL

# - Configure Gemini Client -
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are CaneNexus, an industrial automation assistant for sugarcane mills.

STRICT RULES:
1. You ONLY discuss the model prediction results and knowledge graph recommendations provided to you.
2. You do NOT have general knowledge. You are a pure, conversational action recommender.
3. You NEVER discuss topics outside the provided analysis context.
4. If asked about anything unrelated to the current analysis, politely decline and redirect.

Your responses must:
1. Summarise the model's prediction in plain, accessible language
2. Explain the knowledge graph's recommendation and WHY it was triggered
3. Suggest concrete operator actions based ONLY on the KG output
4. Be concise, professional, and actionable
5. Use bullet points for action items
6. For steel: explain defect classes, severity, and quality decision
7. For sugar: explain crystallisation state, nucleation risk, and process control actions
"""


def get_initial_response(prediction_result: dict, kg_result: dict) -> str:
    """
    - Generates the first chatbot message
    - Summarises the analysis

    Args:
        - prediction_result: Output from steel_inference or sugar_inference
        - kg_result: Output from steel_kg or sugar_kg evaluation

    Returns:
        - Gemini's initial assessment as a string.
    """
    domain = prediction_result.get("domain", "unknown")

    context = f"""
        ANALYSIS RESULTS FOR REVIEW:
        Domain: {domain.upper()}

        MODEL PREDICTION:
        {json.dumps(prediction_result, indent=2, default=str)}

        KNOWLEDGE GRAPH OUTPUT:
        {json.dumps(kg_result, indent=2, default=str)}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Analyse these results and provide your initial assessment. Be specific about what the operator should do next:\n\n{context}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        return response.text

    except Exception as e:
        # Fallback if Gemini fails
        return (
            f"**CaneNexus Analysis Summary**\n\n"
            f"Domain: {domain.upper()}\n\n"
            f"The analysis has been completed. "
            f"Key finding: {kg_result.get('details', 'Results available above.')}\n\n"
            f"_Note: AI assistant encountered an issue ({str(e)[:100]}). "
            f"Please review the detailed results above._"
        )


def chat_response(
    history: list,
    user_message: str,
    prediction_result: dict,
    kg_result: dict
) -> str:
    """
    - Continues a conversation about analysis results

    Args:
        - history: List of previous messages [{"role": "user"/"model", "parts": ["..."]}]
        - user_message: The new user message
        - prediction_result: Original prediction for context
        - kg_result: Original KG result for context

    Returns:
        - Gemini's response as a string
    """
    # Build context reminder
    context_reminder = (
        f"\n\n[CONTEXT REMINDER - Analysis data you must base responses on]\n"
        f"Domain: {prediction_result.get('domain', 'unknown').upper()}\n"
        f"KG Details: {kg_result.get('details', 'N/A')}\n"
    )

    try:
        gemini_history = []
        for msg in history:
            role = msg["role"]
            gemini_history.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
            
        gemini_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=gemini_history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT + context_reminder
            )
        )
        return response.text

    except Exception as e:
        return (
            f"I apologize, but I'm having trouble processing your request. "
            f"Error: {str(e)[:100]}. "
            f"Please try rephrasing your question about the analysis results."
        )