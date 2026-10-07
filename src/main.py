from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# INTERNAL MODULES
# ============================================================

from .sql_query_planner import generate_sql
from .sql_executor import execute_sql
from .response_generator import generate_response


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="IPL Copilot API",
    description="AI-powered IPL analytics assistant",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SESSION STORAGE
# ============================================================

sessions: dict[str, dict] = {}


# ============================================================
# REQUEST SCHEMA
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Natural-language IPL question",
    )

    session_id: str | None = Field(
        default=None,
        description="Optional conversation session ID",
    )


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class QuestionResponse(BaseModel):
    answer: str
    session_id: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "IPL Copilot API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ipl-copilot",
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest,
):
    question = request.question.strip()

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # --------------------------------------------------------
    # Create / retrieve session
    # --------------------------------------------------------

    session_id = request.session_id

    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = {
            "questions": []
        }

    # ========================================================
    # STEP 1: SQL PLANNING
    # ========================================================

    try:
        planner_result = generate_sql(
            question
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"SQL planning failed: {error}",
        ) from error

    # --------------------------------------------------------
    # Check planner result
    # --------------------------------------------------------

    if not isinstance(planner_result, dict):
        raise HTTPException(
            status_code=500,
            detail="SQL planner returned an invalid response.",
        )

    # --------------------------------------------------------
    # Unsupported question
    # --------------------------------------------------------

    if planner_result.get("status") != "ready":

        reason = planner_result.get(
            "reason",
            "The question could not be processed.",
        )

        sessions[session_id]["questions"].append(
            {
                "question": question,
                "answer": reason,
            }
        )

        return QuestionResponse(
            answer=str(reason),
            session_id=session_id,
        )

    # ========================================================
    # STEP 2: GET SQL
    # ========================================================

    sql = planner_result.get("sql")

    if not sql:
        raise HTTPException(
            status_code=500,
            detail="SQL planner returned no SQL.",
        )

    # ========================================================
    # STEP 3: EXECUTE SQL
    # ========================================================

    try:
        result = execute_sql(
            sql
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"SQL execution failed: {error}",
        ) from error

    # ========================================================
    # STEP 4: GENERATE NATURAL-LANGUAGE RESPONSE
    # ========================================================

    try:
        answer = generate_response(
            question,
            result,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Response generation failed: {error}",
        ) from error

    # ========================================================
    # STEP 5: STORE CONVERSATION
    # ========================================================

    sessions[session_id]["questions"].append(
        {
            "question": question,
            "answer": str(answer),
        }
    )

    # Keep only the latest 20 questions
    sessions[session_id]["questions"] = (
        sessions[session_id]["questions"][-20:]
    )

    # ========================================================
    # STEP 6: RETURN RESPONSE
    # ========================================================

    return QuestionResponse(
        answer=str(answer),
        session_id=session_id,
    )


# ============================================================
# CREATE NEW SESSION
# ============================================================

@app.post("/session")
def create_session():

    session_id = str(
        uuid.uuid4()
    )

    sessions[session_id] = {
        "questions": []
    }

    return {
        "session_id": session_id
    }


# ============================================================
# CLEAR SESSION
# ============================================================

@app.delete("/session/{session_id}")
def delete_session(
    session_id: str,
):

    sessions.pop(
        session_id,
        None,
    )

    return {
        "message": "Session cleared",
        "session_id": session_id
    }