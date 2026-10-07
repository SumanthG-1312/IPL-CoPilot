import json
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")


MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the final answer generation engine for IPL Copilot.

Your job is to convert verified IPL data into a clear, natural-language answer.

IMPORTANT RULES:

- Answer ONLY using the verified result provided to you.
- Do NOT calculate new statistics.
- Do NOT invent facts.
- Do NOT change numbers.
- Do NOT introduce information that is not present in the verified result.
- Use the player's full or resolved name when available.
- Answer the user's actual question directly.
- Do not simply repeat the raw result mechanically.
- Make the response natural, readable, and conversational.
- Give enough explanation to make the answer useful, but do not add unsupported information.
- For a simple single-metric question, use 1-2 natural sentences.
- For multiple metrics, use 2-4 sentences when appropriate.
- For comparisons, clearly state the values and the comparison result.
- For rankings, present the ranking clearly using numbered lines.
- For multiple requested metrics, include every requested metric.
- Preserve exact verified numbers.
- Do not mention internal pipeline details.
- Do not mention JSON, parser, executor, database, model, or prompts.
- Return only the final answer in plain text.
"""


# ============================================================
# SERIALIZE EXECUTOR RESULT
# ============================================================

def serialize_result(result) -> str:
    """Convert executor output into text."""

    if result is None:
        return "No result was returned."

    if isinstance(result, str):
        return result

    if isinstance(result, (int, float, bool)):
        return str(result)

    if isinstance(result, dict):
        return json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    if isinstance(result, list):
        return json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    if hasattr(result, "to_dict"):
        try:
            records = result.to_dict(
                orient="records"
            )

            return json.dumps(
                records,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        except TypeError:
            pass

    return str(result)


# ============================================================
# CLEAN GEMINI RESPONSE
# ============================================================

def clean_response(text: str) -> str:
    """Remove unwanted model wrappers."""

    if not text:
        return "I couldn't generate an answer."

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = text.replace(
        "```text",
        "",
    )

    text = text.replace(
        "```",
        "",
    )

    cleaned = text.strip()

    if not cleaned:
        return "I couldn't generate an answer."

    return cleaned


# ============================================================
# GEMINI RESPONSE GENERATION
# ============================================================

def generate_response(
    user_question: str,
    result,
) -> str:
    """Generate a natural-language answer from verified data."""

    result_text = serialize_result(result)

    user_prompt = f"""
User question:

{user_question}

Verified IPL result:

{result_text}

Generate a complete natural-language answer to the user's question.

Make the answer informative but concise.

For a simple statistic:
- Clearly state the player or entity.
- State the exact verified value.
- Use a second sentence only when it adds useful context supported by the verified result.

For comparisons:
- Mention both subjects.
- Include all requested metrics.
- Clearly explain who leads for each metric.

For rankings:
- Introduce the ranking naturally.
- Present all verified entries clearly.
- Keep the ordering exactly as provided.

For multiple metrics:
- Include every requested metric.
- Make the answer easy to read.

Do not add facts that are not supported by the verified result.

Return only the final answer.
"""

    # ========================================================
    # GEMINI API CALL
    # ========================================================

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=150,
        ),
    )

    answer = response.text

    return clean_response(answer)