import pandas as pd
from rapidfuzz import fuzz, process

from .query_schema import (
    QueryAnalysis,
    Subject,
    Ambiguity,
)


DATA_PATH = "data/ipl_ball_by_ball_cleaned.csv"


TEAM_ALIASES = {
    "rcb": "Royal Challengers Bangalore",
    "mi": "Mumbai Indians",
    "csk": "Chennai Super Kings",
    "kkr": "Kolkata Knight Riders",
    "dc": "Delhi Capitals",
    "srh": "Sunrisers Hyderabad",
    "rr": "Rajasthan Royals",
    "pbks": "Punjab Kings",
    "kxip": "Kings XI Punjab",
    "gt": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
    "rps": "Rising Pune Supergiant",
    "rpsg": "Rising Pune Supergiant",
}


PLAYER_ALIASES = {
    "rohit": "RG Sharma",
    "rohit sharma": "RG Sharma",
    "gautam gambhir": "G Gambhir",
    "gautham gambhir": "G Gambhir",
    "virat kohli": "V Kohli",
    "ms dhoni": "MS Dhoni",
    "mahendra singh dhoni": "MS Dhoni",
    "sachin": "SR Tendulkar",
    "sachin tendulkar": "SR Tendulkar",
}


def normalize(value: str) -> str:
    value = str(value).lower().strip()

    value = value.replace(".", " ")
    value = value.replace("-", " ")

    return " ".join(value.split())


def load_entities():
    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
        na_filter=False,
    )

    players = sorted(
        set(
            df["batter"].tolist()
            + df["bowler"].tolist()
            + df["non-striker"].tolist()
        )
    )

    teams = sorted(
        set(
            df["BattingTeam"].tolist()
        )
    )

    return players, teams


players, teams = load_entities()


def resolve_entity(
    user_value: str,
    entity_type: str,
) -> dict:

    if entity_type == "player":
        candidates = players

    elif entity_type == "team":
        candidates = teams

    else:
        return {
            "status": "not_supported",
            "resolved_value": None,
            "candidates": [],
        }

    user_normalized = normalize(
        user_value
    )

    # Team aliases.
    if entity_type == "team":

        alias = TEAM_ALIASES.get(
            user_normalized
        )

        if alias and alias in candidates:
            return {
                "status": "resolved",
                "resolved_value": alias,
                "score": 100,
                "candidates": [],
            }

    # Player aliases.
    if entity_type == "player":

        alias = PLAYER_ALIASES.get(
            user_normalized
        )

        if alias and alias in candidates:
            return {
                "status": "resolved",
                "resolved_value": alias,
                "score": 100,
                "candidates": [],
            }

    # Exact match.
    for candidate in candidates:

        if (
            normalize(candidate)
            == user_normalized
        ):
            return {
                "status": "resolved",
                "resolved_value": candidate,
                "score": 100,
                "candidates": [],
            }

    # Initial + surname.
    user_parts = user_normalized.split()

    if len(user_parts) >= 2:

        user_initial = user_parts[0][0]
        user_surname = user_parts[-1]

        initial_matches = []

        for candidate in candidates:

            candidate_parts = normalize(
                candidate
            ).split()

            if len(candidate_parts) < 2:
                continue

            candidate_initial = (
                candidate_parts[0][0]
            )

            candidate_surname = (
                candidate_parts[-1]
            )

            if (
                candidate_initial
                == user_initial
                and candidate_surname
                == user_surname
            ):
                initial_matches.append(
                    candidate
                )

        if len(initial_matches) == 1:

            return {
                "status": "resolved",
                "resolved_value": (
                    initial_matches[0]
                ),
                "score": 100,
                "candidates": [],
            }

        if len(initial_matches) > 1:

            return {
                "status": "ambiguous",
                "resolved_value": None,
                "score": 100,
                "candidates": [
                    (
                        candidate,
                        100,
                    )
                    for candidate in initial_matches
                ],
            }

    # Unique first-name match.
    if entity_type == "player":

        first_name_matches = []

        for candidate in candidates:

            parts = normalize(
                candidate
            ).split()

            if not parts:
                continue

            if (
                parts[0]
                == user_normalized
            ):
                first_name_matches.append(
                    candidate
                )

        if len(first_name_matches) == 1:

            return {
                "status": "resolved",
                "resolved_value": (
                    first_name_matches[0]
                ),
                "score": 98,
                "candidates": [],
            }

        if len(first_name_matches) > 1:

            return {
                "status": "ambiguous",
                "resolved_value": None,
                "score": 98,
                "candidates": [
                    (
                        candidate,
                        98,
                    )
                    for candidate in first_name_matches
                ],
            }

    # Partial matching.
    partial_matches = []

    for candidate in candidates:

        candidate_normalized = normalize(
            candidate
        )

        if (
            user_normalized
            in candidate_normalized
            or candidate_normalized
            in user_normalized
        ):
            partial_matches.append(
                candidate
            )

    if len(partial_matches) == 1:

        return {
            "status": "resolved",
            "resolved_value": (
                partial_matches[0]
            ),
            "score": 95,
            "candidates": [],
        }

    if len(partial_matches) > 1:

        return {
            "status": "ambiguous",
            "resolved_value": None,
            "score": 95,
            "candidates": [
                (
                    candidate,
                    95,
                )
                for candidate in partial_matches
            ],
        }

    # Fuzzy matching.
    normalized_candidates = {
        normalize(candidate): candidate
        for candidate in candidates
    }

    matches = process.extract(
        user_normalized,
        normalized_candidates.keys(),
        scorer=fuzz.WRatio,
        limit=5,
    )

    matches = [
        match
        for match in matches
        if match[1] >= 70
    ]

    if not matches:

        return {
            "status": "not_found",
            "resolved_value": None,
            "score": 0,
            "candidates": [],
        }

    # Ambiguity detection.
    if len(matches) > 1:

        best_score = matches[0][1]
        second_score = matches[1][1]

        if (
            best_score - second_score
            < 5
        ):

            return {
                "status": "ambiguous",
                "resolved_value": None,
                "score": best_score,
                "candidates": [
                    (
                        normalized_candidates[
                            match[0]
                        ],
                        match[1],
                    )
                    for match in matches
                ],
            }

    best_match = matches[0]

    return {
        "status": "resolved",
        "resolved_value": (
            normalized_candidates[
                best_match[0]
            ]
        ),
        "score": best_match[1],
        "candidates": [],
    }


def resolve_subject(
    subject: Subject,
) -> dict:

    if subject.entity_type not in {
        "player",
        "team",
    }:
        return {
            "status": "not_supported",
            "subject": subject,
            "candidates": [],
        }

    result = resolve_entity(
        str(subject.value),
        subject.entity_type,
    )

    if result["status"] == "resolved":

        resolved_subject = (
            subject.model_copy(
                deep=True
            )
        )

        resolved_subject.value = (
            result["resolved_value"]
        )

        return {
            "status": "resolved",
            "subject": resolved_subject,
            "candidates": [],
        }

    return {
        "status": result["status"],
        "subject": subject,
        "candidates": result.get(
            "candidates",
            [],
        ),
    }


def add_ambiguity(
    result: QueryAnalysis,
    entity_type: str,
    user_value: str,
    candidates: list,
) -> None:

    candidate_names = [
        candidate[0]
        for candidate in candidates
    ]

    for ambiguity in result.ambiguities:

        if (
            ambiguity.entity_type
            == entity_type
            and ambiguity.user_value
            == user_value
        ):
            return

    result.ambiguities.append(
        Ambiguity(
            entity_type=entity_type,
            user_value=user_value,
            candidates=candidate_names,
        )
    )


def resolve_query_analysis(
    analysis: QueryAnalysis,
) -> QueryAnalysis:

    result = analysis.model_copy(
        deep=True
    )

    if result.query is None:
        return result

    query = result.query

    # Resolve normal subjects.
    resolved_subjects = []

    for subject in query.subjects:

        resolution = resolve_subject(
            subject
        )

        if resolution["status"] == "resolved":

            resolved_subjects.append(
                resolution["subject"]
            )

        elif (
            resolution["status"]
            == "ambiguous"
        ):

            result.status = (
                "clarification_required"
            )

            add_ambiguity(
                result,
                subject.entity_type,
                str(subject.value),
                resolution["candidates"],
            )

            resolved_subjects.append(
                subject
            )

        else:

            result.status = (
                "clarification_required"
            )

            add_ambiguity(
                result,
                subject.entity_type,
                str(subject.value),
                [],
            )

            resolved_subjects.append(
                subject
            )

    query.subjects = resolved_subjects

    # Resolve comparison subjects.
    if query.comparison:

        resolved_comparison_subjects = []

        for subject in (
            query.comparison.subjects
        ):

            resolution = resolve_subject(
                subject
            )

            if (
                resolution["status"]
                == "resolved"
            ):

                resolved_comparison_subjects.append(
                    resolution["subject"]
                )

            elif (
                resolution["status"]
                == "ambiguous"
            ):

                result.status = (
                    "clarification_required"
                )

                add_ambiguity(
                    result,
                    subject.entity_type,
                    str(subject.value),
                    resolution["candidates"],
                )

                # Keep the ambiguous subject.
                resolved_comparison_subjects.append(
                    subject
                )

            else:

                result.status = (
                    "clarification_required"
                )

                add_ambiguity(
                    result,
                    subject.entity_type,
                    str(subject.value),
                    [],
                )

                # Keep the unresolved subject.
                resolved_comparison_subjects.append(
                    subject
                )

        query.comparison.subjects = (
            resolved_comparison_subjects
        )

    # Resolve entity filters.
    for filter_item in query.filters:

        if (
            filter_item.attribute
            not in {
                "player",
                "batter",
                "bowler",
                "team",
            }
        ):
            continue

        entity_type = (
            "team"
            if filter_item.attribute
            == "team"
            else "player"
        )

        original_value = str(
            filter_item.value
        )

        resolution = resolve_entity(
            original_value,
            entity_type,
        )

        if (
            resolution["status"]
            == "resolved"
        ):

            filter_item.value = (
                resolution["resolved_value"]
            )

        elif (
            resolution["status"]
            == "ambiguous"
        ):

            result.status = (
                "clarification_required"
            )

            add_ambiguity(
                result,
                entity_type,
                original_value,
                resolution["candidates"],
            )

        else:

            result.status = (
                "clarification_required"
            )

            add_ambiguity(
                result,
                entity_type,
                original_value,
                [],
            )

    # Generate clarification message.
    if (
        result.status
        == "clarification_required"
        and result.ambiguities
    ):

        ambiguity = result.ambiguities[0]

        if ambiguity.candidates:

            candidates_text = ", ".join(
                ambiguity.candidates
            )

            result.clarification_question = (
                f"Which "
                f"{ambiguity.entity_type} "
                f"do you mean by "
                f"'{ambiguity.user_value}'? "
                f"Candidates: "
                f"{candidates_text}."
            )

        else:

            result.clarification_question = (
                f"I couldn't find a matching "
                f"{ambiguity.entity_type} "
                f"for "
                f"'{ambiguity.user_value}'."
            )

    return result


def resolve_query_plan(
    query,
) -> QueryAnalysis:

    if isinstance(
        query,
        QueryAnalysis,
    ):
        return resolve_query_analysis(
            query
        )

    raise TypeError(
        "resolve_query_plan expects "
        "a QueryAnalysis object."
    )


if __name__ == "__main__":

    print(
        "IPL Copilot - Entity Resolver"
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        user_value = input(
            "\nEnter player/team name: "
        ).strip()

        if user_value.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        entity_type = input(
            "Entity type (player/team): "
        ).strip().lower()

        result = resolve_entity(
            user_value,
            entity_type,
        )

        print("\nResult:")
        print(result)