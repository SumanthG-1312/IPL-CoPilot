import json
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

MODEL = os.getenv("GROQ_MODEL")

if not MODEL:
    raise ValueError("GROQ_MODEL is missing from .env")

llm = ChatGroq(
    model=MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2,
)


SYSTEM_PROMPT = """
You are the final answer generation engine for IPL Copilot.

Your job is to convert a user's IPL analytics question and a VERIFIED
database result into a clear natural-language answer.

IMPORTANT RULES:

1. The database result is the ONLY source of truth.
2. Never invent, estimate, calculate, or modify statistics.
3. Never claim that data is unavailable when a valid result is provided.
4. Interpret the result columns in the context of the user's question.
5. Preserve numeric values exactly.
6. Remove unnecessary decimal ".0" from whole numbers when presenting them.
7. Use the player's/team's name from the question when appropriate.
8. Do not mention SQL, DataFrame, database, query, or internal implementation.
9. If multiple rows are returned, summarize them naturally.
10. If multiple metrics are returned, explain them clearly.
11. If the result is empty, clearly say that no matching data was found.
12. Do not answer anything that is not supported by the verified result.
13. Return ONLY the final answer. No JSON, markdown headings, or explanations
    about your reasoning.
"""


def serialize_result(result):
    """
    Convert different executor result types into clean JSON text.
    """

    if result is None:
        return "No result was returned."

    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "[]"

        records = result.to_dict(orient="records")

        return json.dumps(
            records,
            indent=2,
            default=str
        )

    if isinstance(result, str):
        return result

    if isinstance(result, (int, float, bool)):
        return json.dumps(result)

    if isinstance(result, dict):
        return json.dumps(
            result,
            indent=2,
            default=str
        )

    if isinstance(result, list):
        return json.dumps(
            result,
            indent=2,
            default=str
        )

    if hasattr(result, "to_dict"):
        try:
            records = result.to_dict(orient="records")

            return json.dumps(
                records,
                indent=2,
                default=str
            )
        except Exception:
            pass

    return str(result)


def clean_response(text):
    """
    Clean unnecessary formatting from the LLM response.
    """

    if not text:
        return "I couldn't generate an answer."

    text = text.strip()

    # Remove accidental markdown code fences
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()

        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()

    return text


def generate_response(user_question, result):
    """
    Generate the final natural-language answer from the
    user's question and verified database result.
    """

    result_text = serialize_result(result)

    user_prompt = f"""
User Question:
{user_question}

Verified IPL Result:
{result_text}

Generate the final answer to the user's question using ONLY the
verified result above.
"""

    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    content = response.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    return content.strip()