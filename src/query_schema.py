from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# ENUM / TYPE ALIASES
# ============================================================

QueryIntent = Literal[
    "statistic",
    "comparison",
    "ranking",
    "lookup",
    "multi_metric",
    "unknown",
]

EntityType = Literal[
    "player",
    "team",
    "match",
    "unknown",
]

EntityRole = Literal[
    "batter",
    "bowler",
    "fielder",
    "team",
    "unknown",
]

FilterOperator = Literal[
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "contains",
    "between",
]

SortDirection = Literal[
    "asc",
    "desc",
]

ParseStatus = Literal[
    "ready",
    "clarification_required",
    "error",
]

DataNeed = Literal[
    "structured",
    "rag",
    "live",
    "unknown",
]


# ============================================================
# BASE MODEL
# ============================================================

class IPLBaseModel(BaseModel):
    """
    Base model for IPL Copilot schemas.

    Extra fields are ignored so the schema remains compatible
    with older parser outputs and future extensions.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
    )


# ============================================================
# SUBJECT
# ============================================================

class Subject(IPLBaseModel):
    """
    Represents an entity mentioned in the user's question.

    Example:

    player = "Virat Kohli"
    role   = "batter"
    """

    entity_type: EntityType
    value: str
    role: EntityRole = "unknown"


# ============================================================
# SEMANTIC FILTER
# ============================================================

class SemanticFilter(IPLBaseModel):
    """
    Represents an explicitly requested filter.

    Example:

    team = "Kolkata Knight Riders"
    innings = 2
    """

    attribute: str
    operator: FilterOperator = "eq"
    value: Any


# ============================================================
# METRIC REQUEST
# ============================================================

class MetricRequest(IPLBaseModel):
    """
    Represents a semantic metric request.

    Examples:

    runs
    strike_rate
    wickets
    economy
    """

    name: str
    subject_role: EntityRole = "unknown"


# ============================================================
# METRIC FILTER / HAVING CONDITION
# ============================================================

class MetricFilter(IPLBaseModel):
    """
    Optional filter applied to a calculated metric.

    Example:

    runs > 1000
    strike_rate >= 130
    """

    metric: str
    operator: FilterOperator = "eq"
    value: Any


# ============================================================
# GROUP BY
# ============================================================

class GroupBy(IPLBaseModel):
    """
    Represents a grouping dimension.

    Examples:

    batter
    bowler
    BattingTeam
    ID
    innings
    """

    attribute: str


# ============================================================
# SORT
# ============================================================

class SortSpec(IPLBaseModel):
    """
    Represents sorting of query results.
    """

    attribute: str
    direction: SortDirection = "desc"


# ============================================================
# SCOPE
# ============================================================

class Scope(IPLBaseModel):
    """
    Represents optional match / innings / over scope.

    Examples:

    match_id = 1312200
    innings = 2
    over_start = 9
    over_end = 19
    """

    match_id: int | None = None
    innings: int | None = None
    over_start: float | None = None
    over_end: float | None = None
    season: int | None = None


# ============================================================
# COMPARISON
# ============================================================

class ComparisonSpec(IPLBaseModel):
    """
    Represents a comparison between two or more subjects.

    Example:

    Compare Kohli and Rohit on runs and sixes.
    """

    subjects: list[Subject] = Field(default_factory=list)
    metrics: list[MetricRequest] = Field(default_factory=list)


# ============================================================
# SEMANTIC QUERY
# ============================================================

class SemanticQuery(IPLBaseModel):
    """
    Structured representation of what the user wants.

    This model describes WHAT should be retrieved/calculated,
    not HOW the database should calculate it.
    """

    intent: QueryIntent = "unknown"

    subjects: list[Subject] = Field(
        default_factory=list
    )

    metrics: list[MetricRequest] = Field(
        default_factory=list
    )

    filters: list[SemanticFilter] = Field(
        default_factory=list
    )

    having: list[MetricFilter] = Field(
        default_factory=list
    )

    group_by: list[GroupBy] = Field(
        default_factory=list
    )

    sort: SortSpec | None = None

    limit: int | None = None

    scope: Scope = Field(
        default_factory=Scope
    )

    comparison: ComparisonSpec | None = None


# ============================================================
# AMBIGUITY
# ============================================================

class Ambiguity(IPLBaseModel):
    """
    Represents an entity that could not be uniquely resolved.

    Example:

    User entered:
        Kohli

    Candidates:
        T Kohli
        V Kohli
    """

    entity_type: EntityType
    user_value: str
    candidates: list[str] = Field(
        default_factory=list
    )


# ============================================================
# QUERY ANALYSIS
# ============================================================

class QueryAnalysis(IPLBaseModel):
    """
    Final semantic interpretation of a user's question.

    The query may be:

    ready
    clarification_required
    error
    """

    status: ParseStatus = "ready"

    query: SemanticQuery | None = None

    ambiguities: list[Any] = Field(
        default_factory=list
    )

    data_need: DataNeed = "unknown"

    clarification_question: str | None = None