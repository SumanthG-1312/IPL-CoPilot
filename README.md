# 🏏 IPL Copilot

An AI-powered IPL analytics assistant that lets users ask
cricket-related questions in natural language and receive answers from
an IPL ball-by-ball dataset.

IPL Copilot combines **Next.js, FastAPI, Google Gemini, selected
LangChain components, SQL, and Python-based data processing** to turn
natural-language questions into data-driven answers.

------------------------------------------------------------------------

## 🚀 Overview

IPL Copilot is designed to make IPL statistics easier to access.

Instead of requiring users to write SQL queries or search through
predefined statistics, the user can simply ask a question such as:

> **Who scored the most runs in IPL?**

The system interprets the question, generates a structured query plan
and SQL query, executes it against the IPL data, and converts the result
into a natural-language answer.

### Core Flow

``` text
User Question
      ↓
Next.js Frontend
      ↓
FastAPI Backend
      ↓
Gemini + Selected LangChain Components
      ↓
Structured Query Planning
      ↓
SQL Query
      ↓
SQL Executor
      ↓
IPL Dataset
      ↓
Query Result
      ↓
Gemini + Selected LangChain Components
      ↓
Natural Language Answer
      ↓
Next.js Frontend
```

------------------------------------------------------------------------

## ✨ Features

-   Natural-language IPL analytics
-   LLM-powered SQL query generation
-   Structured LLM output
-   SQL-based retrieval from IPL data
-   Natural-language response generation
-   Conversation session management
-   Next.js chat interface
-   FastAPI REST API
-   Health-check endpoint
-   Interactive API documentation through FastAPI
-   Frontend/backend communication through REST APIs

------------------------------------------------------------------------

## 🏗️ System Architecture

``` text
┌──────────────────────────────┐
│            User              │
│  Natural Language Question   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Next.js Frontend       │
│      React + TypeScript      │
└──────────────┬───────────────┘
               │
               │ POST /ask
               ▼
┌──────────────────────────────┐
│       FastAPI Backend        │
│       Request Handling       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    SQL Query Planner         │
│                              │
│ Gemini + selected LangChain  │
│ components + structured      │
│ output                       │
└──────────────┬───────────────┘
               │
               │ Generated SQL
               ▼
┌──────────────────────────────┐
│        SQL Executor          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        IPL Dataset           │
│     Ball-by-ball data        │
└──────────────┬───────────────┘
               │
               │ Query Result
               ▼
┌──────────────────────────────┐
│     Response Generator       │
│                              │
│ Gemini + selected LangChain  │
│ components                   │
└──────────────┬───────────────┘
               │
               │ Final Answer
               ▼
┌──────────────────────────────┐
│       Next.js Frontend       │
└──────────────────────────────┘
```

------------------------------------------------------------------------

## 🔄 How IPL Copilot Works

### 1. User Query

The user enters an IPL-related question through the Next.js interface.

Example:

``` text
Who scored the most runs?
```

### 2. Frontend Request

The Next.js frontend sends the question to the FastAPI backend using the
`/ask` REST endpoint.

``` http
POST /ask
```

The request contains the question and, when available, the current
session ID.

### 3. FastAPI Request Handling

FastAPI receives the request and:

-   Validates the question
-   Creates or reuses a session ID
-   Sends the question to the SQL query planner

### 4. Query Planning

The SQL query planner uses **Google Gemini** with selected **LangChain
components** to understand the user's question.

The LLM produces structured information that can be used to construct
the required SQL query.

Structured output helps make the model response predictable instead of
relying on unrestricted free-form text.

### 5. SQL Execution

The generated SQL query is passed to the SQL executor.

The executor retrieves the required information from the IPL dataset.

### 6. Response Generation

The query result is passed to the response-generation layer.

Gemini converts the retrieved result into a concise natural-language
answer.

Example:

``` text
Virat Kohli is the highest run scorer with X runs.
```

### 7. Response to Frontend

FastAPI returns the final answer and session ID to the Next.js frontend.

The frontend displays the response in the chat interface.

### 8. Session Management

IPL Copilot maintains a session ID for a conversation.

The backend stores recent questions and answers associated with that
session, allowing the application to maintain conversation context at
the application level.

------------------------------------------------------------------------

## 🧠 Role of LangChain

IPL Copilot does **not** depend on the entire LangChain ecosystem.

Only selected LangChain capabilities are used where they provide value.

### LangChain is used for:

-   LLM integration
-   Structured output
-   Prompt/LLM orchestration

The main analytics workflow is implemented directly using:

-   Python
-   FastAPI
-   SQL
-   Dataset processing
-   Custom query-planning and execution logic

This keeps the architecture relatively lightweight while still using
LangChain where it is useful.

------------------------------------------------------------------------

## 🤖 Role of Google Gemini

Google Gemini is used as the application's LLM.

It is responsible for tasks such as:

1.  Understanding natural-language IPL questions
2.  Producing structured query-planning information
3.  Generating the SQL required to retrieve the requested information
4.  Converting SQL results into natural-language responses

The actual statistics are retrieved from the project's dataset rather
than relying solely on the LLM's internal knowledge.

------------------------------------------------------------------------

## 📊 Data Layer

The project uses IPL ball-by-ball data.

The dataset contains information such as:

-   Match information
-   Innings
-   Overs
-   Ball numbers
-   Batters
-   Bowlers
-   Runs
-   Extras
-   Wickets
-   Players dismissed
-   Batting teams
-   Fielders involved

The project also contains a cleaned version of the dataset used during
processing.

------------------------------------------------------------------------

## 🛠️ Technology Stack

### Frontend

-   Next.js
-   React
-   TypeScript
-   Tailwind CSS
-   shadcn/ui
-   Lucide React
-   Motion

### Backend

-   Python
-   FastAPI
-   Pydantic
-   SQL

### AI / LLM

-   Google Gemini
-   LangChain

### Data Processing

-   Pandas
-   IPL ball-by-ball dataset

### Development

-   Git
-   GitHub
-   REST API

------------------------------------------------------------------------

## 📁 Project Structure

``` text
IPL-CoPilot/
│
├── data/
│   ├── ipl_ball_by_ball.csv
│   └── ipl_ball_by_ball_cleaned.csv
│
├── src/
│   ├── data_preperation.py
│   ├── entity_resolver.py
│   ├── main.py
│   ├── metric_engine.py
│   ├── query_schema.py
│   ├── response_generator.py
│   ├── schema_inspector.py
│   ├── sql_executor.py
│   └── sql_query_planner.py
│
├── tests/
│
├── frontend/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   └── ui/
│   ├── lib/
│   ├── public/
│   ├── components.json
│   ├── next.config.ts
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.mjs
│   └── tsconfig.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## 🔌 API Endpoints

  Method     Endpoint                  Description
  ---------- ------------------------- ---------------------------
  `GET`      `/`                       Returns API status
  `GET`      `/health`                 Health check
  `POST`     `/ask`                    Processes an IPL question
  `POST`     `/session`                Creates a new session
  `DELETE`   `/session/{session_id}`   Clears a session

### Example `/ask` Request

``` json
{
  "question": "Who scored the most runs?",
  "session_id": null
}
```

### Example Response

``` json
{
  "answer": "The highest run scorer is ...",
  "session_id": "your-session-id"
}
```

------------------------------------------------------------------------

## 💬 Example Questions

IPL Copilot can be used for questions such as:

``` text
Who scored the most runs?

Who took the most wickets?

Who hit the most sixes?

How many runs did a particular player score?

Which team scored the most runs?

What is the total number of runs scored by a player?

Who has the highest number of fours?
```

The exact questions supported depend on the available dataset schema and
query-planning logic.

------------------------------------------------------------------------

## ⚙️ Local Setup

### Prerequisites

Make sure you have:

-   Python 3.x
-   Node.js
-   npm
-   A Google Gemini API key
-   Git

------------------------------------------------------------------------

## 🔧 Backend Setup

From the project root:

``` bash
python -m venv myvenv
```

### Windows

``` powershell
.\myvenv\Scripts\Activate.ps1
```

Install Python dependencies:

``` bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

``` env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Start the FastAPI server:

``` bash
python -m uvicorn src.main:app --reload
```

Backend:

``` text
http://127.0.0.1:8000
```

FastAPI documentation:

``` text
http://127.0.0.1:8000/docs
```

Health check:

``` text
http://127.0.0.1:8000/health
```

------------------------------------------------------------------------

## 🎨 Frontend Setup

Open a new terminal and navigate to the frontend:

``` bash
cd frontend
```

Install dependencies:

``` bash
npm install
```

Create:

``` text
frontend/.env.local
```

Add:

``` env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Start the development server:

``` bash
npm run dev
```

Frontend:

``` text
http://localhost:3000
```

------------------------------------------------------------------------

## 🔐 Environment Variables

### Backend

``` env
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

### Frontend

``` env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

> Never commit `.env` or `.env.local` files containing API keys or other
> secrets.

------------------------------------------------------------------------

## 🧪 Testing the Application

After starting both servers:

### 1. Open the frontend

``` text
http://localhost:3000
```

### 2. Ask a question

``` text
Top run scorers
```

### 3. Verify the API request

The browser should send:

``` text
POST http://127.0.0.1:8000/ask
```

with a successful response.

### 4. Test conversation sessions

Ask a question and then ask a related follow-up question while using the
same conversation.

### 5. Test New Chat

Starting a new chat should create a new conversation state.

------------------------------------------------------------------------

## 🎯 Problem Solved

Traditional IPL statistics systems generally require users to:

-   Search predefined statistics
-   Navigate dashboards
-   Know which metric they need
-   Understand the underlying data structure

IPL Copilot provides a natural-language interface instead.

The user does not need to know SQL or the internal dataset structure.

For example:

``` text
User:
"Who scored the most runs?"
```

is transformed into a data retrieval operation that queries the actual
IPL data.

This makes IPL analytics more accessible while keeping the final answer
grounded in the project's dataset.

------------------------------------------------------------------------

## 💡 Key Design Decision

A major design decision was to avoid asking the LLM to directly answer
statistical questions from its own knowledge.

Instead:

``` text
Question
   ↓
LLM interprets question
   ↓
SQL generated
   ↓
Actual IPL data queried
   ↓
Result returned
   ↓
LLM explains result
```

This approach separates:

**Reasoning / Query Planning**

from

**Data Retrieval**

and

**Response Generation**

This allows the application to use the dataset as the source of truth
for supported statistical queries.

------------------------------------------------------------------------

## 🧩 Main Backend Components

### `main.py`

The FastAPI application entry point.

Responsibilities include:

-   API configuration
-   CORS
-   Request validation
-   Session management
-   Connecting the query-planning, execution, and response-generation
    stages

### `sql_query_planner.py`

Responsible for interpreting the natural-language question and
generating the SQL/query-planning output using the LLM.

### `query_schema.py`

Defines structured schemas used for predictable query-planning output.

### `sql_executor.py`

Responsible for executing the generated SQL query and returning the
result.

### `response_generator.py`

Uses the LLM to convert retrieved results into a natural-language
response.

### `entity_resolver.py`

Handles entity-related resolution required by the query workflow.

### `metric_engine.py`

Contains metric-related analytics logic.

### `schema_inspector.py`

Provides schema-related information required during query planning and
processing.

### `data_preperation.py`

Handles IPL dataset preparation and processing.

------------------------------------------------------------------------

## ⚠️ Limitations

-   The application is currently focused on the available IPL dataset.
-   The quality of generated SQL depends partly on the LLM's
    interpretation of the question.
-   Unsupported or ambiguous questions may not produce the expected
    result.
-   Gemini API availability and usage limits can affect the application.
-   Current session storage is in-memory.
-   Production deployment would require persistent session storage and
    production-grade database configuration.
-   The current data is not a live IPL data source.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Add more IPL seasons and datasets
-   Add live IPL data sources
-   Improve query validation
-   Add player-vs-player comparisons
-   Add team comparisons
-   Add advanced statistical analysis
-   Add charts and visual analytics
-   Add persistent conversation history
-   Add authentication
-   Add production database infrastructure
-   Deploy frontend and backend
-   Improve handling of ambiguous natural-language queries

------------------------------------------------------------------------

## 📌 Project Status

**Current Status:** Working local prototype

The current implementation includes:

-   Working Next.js frontend
-   Working FastAPI backend
-   Frontend-to-backend REST integration
-   Gemini-based query planning
-   SQL execution
-   LLM-based response generation
-   Session handling
-   IPL dataset integration

------------------------------------------------------------------------

## 👨‍💻 Author

### Sumanth Gajjela

AI / ML & GenAI Developer

GitHub:

**https://github.com/SumanthG-1312**

------------------------------------------------------------------------

## 📄 License

This project is intended for educational, learning, and portfolio
purposes.
