
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .sql_query_planner import generate_sql
from .sql_executor import close_connection, execute_sql, warm_up
from .response_generator import generate_response


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="IPL Copilot API",
    description="AI-powered IPL cricket analytics assistant",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    warm_up()
    logger.info("DuckDB dataset loaded during startup")


@app.on_event("shutdown")
def shutdown():
    close_connection()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "https://ipl-co-pilot.vercel.app,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST AND RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language IPL analytics question",
    )

    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    session_id: str


# ============================================================
# IN-MEMORY SESSION STORAGE
# ============================================================

# Stores recent questions for each session.
# This is temporary storage and resets when the process restarts.

sessions: dict[str, dict] = {}

MAX_HISTORY = 20


def get_or_create_session(
    session_id: Optional[str],
) -> str:
    """Return an existing session ID or create a new session."""

    if session_id and session_id in sessions:
        return session_id

    new_session_id = str(uuid4())

    sessions[new_session_id] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }

    return new_session_id


def save_question(
    session_id: str,
    question: str,
    answer: str,
) -> None:
    """Save the latest question and answer for the session."""

    session = sessions.get(session_id)

    if session is None:
        return

    session["history"].append(
        {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    session["history"] = session["history"][-MAX_HISTORY:]


def build_context_question(
    session_id: str,
    question: str,
) -> str:
    """Attach the most recent turn only for follow-up-style questions."""
    normalized = question.lower()
    is_follow_up = any(
        marker in normalized
        for marker in (
            "their ",
            "his ",
            "her ",
            "they ",
            "it ",
            "that ",
            "what about",
            "and what",
            "compare them",
        )
    )

    if not is_follow_up:
        return question

    history = sessions[session_id]["history"]
    if not history:
        return question

    previous = history[-1]["question"]
    return (
        f"Previous question: {previous}\n"
        f"Follow-up question: {question}"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "message": "IPL Copilot API is running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "IPL Copilot API",
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
):
    """
    Process an IPL analytics question.

    Flow:
    1. Validate the question.
    2. Generate a SQL query.
    3. Execute the query.
    4. Format the result into a response.
    5. Save the interaction in session history.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    session_id = get_or_create_session(
        request.session_id
    )

    try:
        # ----------------------------------------------------
        # STEP 1: GENERATE SQL
        # ----------------------------------------------------

        logger.info(
            "Processing question for session %s",
            session_id,
        )

        contextual_question = build_context_question(
            session_id,
            question,
        )
        planner_result = generate_sql(contextual_question)

        if not isinstance(planner_result, dict):
            raise RuntimeError(
                "The SQL planner returned an invalid response."
            )

        if planner_result.get("status") != "ready":
            reason = planner_result.get("reason")

            if reason:
                answer = str(reason)
            else:
                answer = (
                    "I couldn't prepare a query for that question. "
                    "Please rephrase it."
                )

            save_question(
                session_id,
                question,
                answer,
            )

            return AskResponse(
                answer=answer,
                session_id=session_id,
            )

        sql_query = planner_result.get("sql")

        if not isinstance(sql_query, str) or not sql_query.strip():
            raise RuntimeError(
                "The SQL planner did not return a valid SQL query."
            )

        # ----------------------------------------------------
        # STEP 2: EXECUTE SQL
        # ----------------------------------------------------

        result = execute_sql(sql_query)

        # ----------------------------------------------------
        # STEP 3: GENERATE RESPONSE
        # ----------------------------------------------------

        resolved_question = planner_result.get(
            "resolved_question",
            question,
        )

        answer = generate_response(
            resolved_question,
            result,
            use_llm=not planner_result.get(
                "fast_path",
                False,
            ),
            # Important fix: pass detected metrics to the
            # response generator so it formats the correct value.
            metrics=planner_result.get(
                "metrics",
                [],
            ),
        )

        if not isinstance(answer, str):
            answer = str(answer)

        # ----------------------------------------------------
        # STEP 4: SAVE SESSION HISTORY
        # ----------------------------------------------------

        save_question(
            session_id,
            question,
            answer,
        )

        logger.info(
            "Question processed successfully for session %s",
            session_id,
        )

        # ----------------------------------------------------
        # STEP 5: RETURN RESPONSE
        # ----------------------------------------------------

        return AskResponse(
            answer=answer,
            session_id=session_id,
        )

    except HTTPException:
        raise

    except ValueError as error:
        logger.warning(
            "IPL query needs clarification or is invalid "
            "for session %s: %s",
            session_id,
            error,
        )
        answer = str(error)
        save_question(session_id, question, answer)
        return AskResponse(
            answer=answer,
            session_id=session_id,
        )

    except Exception as error:
        logger.exception(
            "Failed to process IPL analytics question "
            "for session %s",
            session_id,
        )

        # Keep internal exception details in server logs,
        # rather than exposing them to the frontend.
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process your question right now. "
                "Please try again."
            ),
        ) from error


# ============================================================
# SESSION HISTORY ENDPOINT
# ============================================================

@app.get("/sessions/{session_id}")
def get_session_history(
    session_id: str,
):
    """Return recent question-and-answer history."""

    session = sessions.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "history": session["history"],
    }


@app.post("/session")
def create_session():
    session_id = str(uuid4())
    sessions[session_id] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }
    return {"session_id": session_id}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )
    del sessions[session_id]
    return {"deleted": True, "session_id": session_id}
