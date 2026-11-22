
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
configured_dataset = os.getenv("IPL_DATASET_PATH")
DATASET_PATH = (
    str(PROJECT_ROOT / configured_dataset)
    if configured_dataset and not Path(configured_dataset).is_absolute()
    else configured_dataset or str(
        PROJECT_ROOT / "data" / "ipl_ball_by_ball_cleaned.csv"
    )
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value: str) -> str:
    """Normalize text for reliable entity matching."""
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = value.replace("’", "'")
    value = re.sub(r"['']", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return re.sub(r"\s+", " ", value).strip()


def compact(value: str) -> str:
    """Remove spaces from normalized text."""
    return normalize(value).replace(" ", "")


def is_valid_value(value) -> bool:
    """Check whether a dataset value is meaningful."""
    if value is None:
        return False

    return str(value).strip().lower() not in {
        "",
        "nan",
        "none",
        "null",
        "nat",
    }


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"IPL dataset not found: {DATASET_PATH}. "
            "Set IPL_DATASET_PATH to the correct CSV path."
        )

    return pd.read_csv(DATASET_PATH, low_memory=False)


df = load_dataset()


# ============================================================
# EXTRACT CANONICAL PLAYERS
# ============================================================

def collect_players(dataframe: pd.DataFrame) -> list[str]:
    columns = [
        "batter",
        "bowler",
        "non-striker",
        "player_out",
    ]

    values = set()

    for column in columns:
        if column not in dataframe.columns:
            continue

        for value in dataframe[column].dropna().astype(str):
            value = value.strip()

            if is_valid_value(value):
                values.add(value)

    return sorted(values, key=normalize)


players = collect_players(df)

NORMALIZED_PLAYERS = {
    normalize(player): player
    for player in players
}


# ============================================================
# EXTRACT CANONICAL TEAMS
# ============================================================

def collect_teams(dataframe: pd.DataFrame) -> list[str]:
    if "BattingTeam" not in dataframe.columns:
        return []

    values = {
        str(value).strip()
        for value in dataframe["BattingTeam"].dropna()
        if is_valid_value(value)
    }

    return sorted(values, key=normalize)


teams = collect_teams(df)

NORMALIZED_TEAMS = {
    normalize(team): team
    for team in teams
}


# ============================================================
# PLAYER ALIASES
# ============================================================

def generate_name_aliases(name: str) -> set[str]:
    """
    Generate aliases dynamically from a stored player name.

    Examples:
        AB de Villiers -> ab de villiers, abdevilliers, abd
        V Kohli        -> v kohli, vk
        RG Sharma      -> rg sharma, rgsharma, rgs
    """
    normalized = normalize(name)

    if not normalized:
        return set()

    parts = normalized.split()

    aliases = {
        normalized,
        compact(normalized),
    }

    # Multiword name components.
    for start in range(len(parts)):
        for end in range(start + 2, len(parts) + 1):
            phrase = " ".join(parts[start:end])

            if len(phrase) >= 3:
                aliases.add(phrase)

    # Initials of all name components.
    initials = "".join(
        part[0] for part in parts if part
    )

    if len(initials) >= 2:
        aliases.add(initials)

    if len(parts) >= 2:
        # First stored token + initials of remaining tokens.
        aliases.add(
            parts[0]
            + "".join(part[0] for part in parts[1:])
        )

        # First stored token + last initial.
        aliases.add(parts[0] + parts[-1][0])

    return {
        normalize(alias)
        for alias in aliases
        if normalize(alias)
    }


def build_player_alias_index():
    index = defaultdict(set)

    for player in players:
        for alias in generate_name_aliases(player):
            index[alias].add(player)

    return index


PLAYER_ALIAS_CANDIDATES = build_player_alias_index()


# ============================================================
# KNOWN FULL-NAME ALIASES
# ============================================================

# Dataset names may use initials instead of full first names.
# These aliases bridge known differences between query names
# and dataset names. Each mapping is checked against the CSV.

KNOWN_PLAYER_ALIASES = {
    "rohit sharma": "RG Sharma",
    "gautam gambhir": "G Gambhir",
    "abd": "AB de Villiers",
}


def apply_known_player_aliases() -> None:
    for alias, canonical_name in KNOWN_PLAYER_ALIASES.items():
        normalized_alias = normalize(alias)

        if canonical_name in players:
            PLAYER_ALIAS_CANDIDATES[normalized_alias] = {
                canonical_name
            }


apply_known_player_aliases()


# Only aliases with one candidate are safe for general matching.
PLAYER_ALIASES = {
    alias: next(iter(candidates))
    for alias, candidates in PLAYER_ALIAS_CANDIDATES.items()
    if len(candidates) == 1
}


# ============================================================
# TEAM ALIASES
# ============================================================

def generate_team_aliases(name: str) -> set[str]:
    normalized = normalize(name)

    if not normalized:
        return set()

    parts = normalized.split()

    aliases = {
        normalized,
        compact(normalized),
    }

    for part in parts:
        if len(part) >= 3:
            aliases.add(part)

    initials = "".join(
        part[0] for part in parts if part
    )

    if len(initials) >= 2:
        aliases.add(initials)

    return {
        normalize(alias)
        for alias in aliases
        if normalize(alias)
    }


def build_team_alias_index():
    index = defaultdict(set)

    for team in teams:
        for alias in generate_team_aliases(team):
            index[alias].add(team)

    return index


TEAM_ALIAS_CANDIDATES = build_team_alias_index()

TEAM_ALIASES = {
    alias: next(iter(candidates))
    for alias, candidates in TEAM_ALIAS_CANDIDATES.items()
    if len(candidates) == 1
}


# ============================================================
# EXACT MATCH HELPERS
# ============================================================

def exact_player_match(value: str):
    normalized = normalize(value)

    if not normalized:
        return None

    if normalized in NORMALIZED_PLAYERS:
        return NORMALIZED_PLAYERS[normalized]

    candidates = PLAYER_ALIAS_CANDIDATES.get(
        normalized, set()
    )

    if len(candidates) == 1:
        return next(iter(candidates))

    return None


def exact_team_match(value: str):
    normalized = normalize(value)

    if not normalized:
        return None

    if normalized in NORMALIZED_TEAMS:
        return NORMALIZED_TEAMS[normalized]

    candidates = TEAM_ALIAS_CANDIDATES.get(
        normalized, set()
    )

    if len(candidates) == 1:
        return next(iter(candidates))

    return None


# ============================================================
# FULL-NAME VARIANT MATCHING
# ============================================================

def resolve_name_variant(value: str):
    """
    Resolve a full name to an abbreviated dataset name when
    the surname and first initial identify exactly one player.

    Example:
        Gautam Gambhir -> G Gambhir
        Rohit Sharma   -> RG Sharma, if unique

    Returns None when the result is ambiguous.
    """
    query_parts = normalize(value).split()

    if len(query_parts) < 2:
        return None

    query_surname = query_parts[-1]
    query_first_initial = query_parts[0][0]

    candidates = []

    for player in players:
        player_parts = normalize(player).split()

        if len(player_parts) < 2:
            continue

        if player_parts[-1] != query_surname:
            continue

        if player_parts[0][0] != query_first_initial:
            continue

        candidates.append(player)

    if len(candidates) == 1:
        return candidates[0]

    return None


# ============================================================
# FUZZY MATCHING
# ============================================================

def fuzzy_resolve(
    value: str,
    candidates: list[str],
):
    normalized = normalize(value)

    if not normalized or not candidates:
        return {
            "input": value,
            "resolved": None,
            "resolved_value": None,
            "method": "not_found",
            "status": "not_found",
        }

    scored = []

    for candidate in candidates:
        score = SequenceMatcher(
            None,
            normalized,
            normalize(candidate),
        ).ratio()

        scored.append((score, candidate))

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best_candidate = scored[0]

    second_score = (
        scored[1][0]
        if len(scored) > 1
        else 0.0
    )

    threshold = (
        0.92
        if len(normalized) <= 4
        else 0.88
    )

    if best_score < threshold:
        return {
            "input": value,
            "resolved": None,
            "resolved_value": None,
            "method": "fuzzy",
            "score": round(best_score, 4),
            "status": "not_found",
        }

    # Refuse to guess when multiple candidates are too close.
    if best_score - second_score < 0.05:
        ambiguous_candidates = [
            candidate
            for score, candidate in scored[:5]
            if best_score - score < 0.05
        ]

        return {
            "input": value,
            "resolved": None,
            "resolved_value": None,
            "method": "fuzzy_ambiguous",
            "score": round(best_score, 4),
            "candidates": ambiguous_candidates,
            "status": "ambiguous",
        }

    return {
        "input": value,
        "resolved": best_candidate,
        "resolved_value": best_candidate,
        "method": "fuzzy",
        "score": round(best_score, 4),
        "status": "resolved",
    }


# ============================================================
# ENTITY RESOLUTION
# ============================================================

def resolve_entity(value: str, entity_type: str):
    """
    Resolve a player or team name.

    Result fields are kept compatible with the SQL planner:
        input, resolved, resolved_value, method, status
    """
    normalized = normalize(value)

    if not normalized:
        return {
            "input": value,
            "resolved": None,
            "resolved_value": None,
            "method": "empty",
            "status": "not_found",
        }

    entity_type = normalize(entity_type)

    # --------------------------------------------------------
    # PLAYER
    # --------------------------------------------------------
    if entity_type == "player":

        # Exact canonical dataset name.
        if normalized in NORMALIZED_PLAYERS:
            player = NORMALIZED_PLAYERS[normalized]

            return {
                "input": value,
                "resolved": player,
                "resolved_value": player,
                "method": "exact",
                "score": 1.0,
                "status": "resolved",
            }

        # Explicit or dynamically generated alias.
        candidates = PLAYER_ALIAS_CANDIDATES.get(
            normalized, set()
        )

        if len(candidates) == 1:
            player = next(iter(candidates))

            return {
                "input": value,
                "resolved": player,
                "resolved_value": player,
                "method": "alias",
                "score": 1.0,
                "status": "resolved",
            }

        if len(candidates) > 1:
            return {
                "input": value,
                "resolved": None,
                "resolved_value": None,
                "method": "ambiguous_alias",
                "score": 1.0,
                "candidates": sorted(candidates),
                "status": "ambiguous",
            }

        # Match full names to abbreviated stored names.
        variant = resolve_name_variant(value)

        if variant:
            return {
                "input": value,
                "resolved": variant,
                "resolved_value": variant,
                "method": "name_variant",
                "score": 1.0,
                "status": "resolved",
            }

        # Fuzzy matching is a last resort.
        return fuzzy_resolve(value, players)

    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------
    if entity_type == "team":

        if normalized in NORMALIZED_TEAMS:
            team = NORMALIZED_TEAMS[normalized]

            return {
                "input": value,
                "resolved": team,
                "resolved_value": team,
                "method": "exact",
                "score": 1.0,
                "status": "resolved",
            }

        candidates = TEAM_ALIAS_CANDIDATES.get(
            normalized, set()
        )

        if len(candidates) == 1:
            team = next(iter(candidates))

            return {
                "input": value,
                "resolved": team,
                "resolved_value": team,
                "method": "alias",
                "score": 1.0,
                "status": "resolved",
            }

        if len(candidates) > 1:
            return {
                "input": value,
                "resolved": None,
                "resolved_value": None,
                "method": "ambiguous_alias",
                "score": 1.0,
                "candidates": sorted(candidates),
                "status": "ambiguous",
            }

        return fuzzy_resolve(value, teams)

    return {
        "input": value,
        "resolved": None,
        "resolved_value": None,
        "method": "unsupported_entity_type",
        "status": "unsupported",
    }


# ============================================================
# FIND PLAYER MENTIONS IN QUESTIONS
# ============================================================

def find_player_mentions(question: str):
    """
    Find player aliases inside a natural-language question.

    This function deliberately avoids fuzzy matching arbitrary
    question fragments. It only accepts known aliases and
    verified full-name variants.
    """
    normalized_question = normalize(question)

    if not normalized_question:
        return []

    words = normalized_question.split()
    mentions = []
    occupied_ranges = []

    def overlaps(start: int, end: int) -> bool:
        return any(
            start < existing_end and end > existing_start
            for existing_start, existing_end in occupied_ranges
        )

    def add_mention(
        start: int,
        end: int,
        input_value: str,
        canonical_player: str,
    ) -> None:
        if overlaps(start, end):
            return

        occupied_ranges.append((start, end))

        mentions.append({
            "input": input_value,
            "value": input_value,
            "resolved": canonical_player,
            "resolved_value": canonical_player,
            "status": "resolved",
        })

    # First pass: known aliases, longest first.
    aliases = sorted(
        PLAYER_ALIAS_CANDIDATES.keys(),
        key=lambda alias: (
            len(alias.split()),
            len(alias),
        ),
        reverse=True,
    )

    for alias in aliases:
        candidates = PLAYER_ALIAS_CANDIDATES[alias]

        if len(candidates) != 1:
            continue

        alias_words = alias.split()
        phrase_length = len(alias_words)

        for start in range(
            len(words) - phrase_length + 1
        ):
            end = start + phrase_length

            if overlaps(start, end):
                continue

            phrase = " ".join(words[start:end])

            if phrase != alias:
                continue

            add_mention(
                start,
                end,
                phrase,
                next(iter(candidates)),
            )

    # Second pass: check two-word full-name variants only.
    # Never fuzzy-resolve long question fragments.
    for phrase_length in [2]:
        for start in range(
            len(words) - phrase_length + 1
        ):
            end = start + phrase_length

            if overlaps(start, end):
                continue

            phrase = " ".join(words[start:end])
            result = resolve_entity(phrase, "player")

            if (
                result["status"] == "resolved"
                and result["method"] == "name_variant"
            ):
                add_mention(
                    start,
                    end,
                    phrase,
                    result["resolved"],
                )

    # Return mentions in the order they appeared in the question.
    mentions.sort(
        key=lambda item: next(
            (
                index
                for index in range(len(words))
                if " ".join(
                    words[
                        index:index
                        + len(normalize(item["input"]).split())
                    ]
                ) == normalize(item["input"])
            ),
            len(words),
        )
    )

    # Remove duplicate input/canonical pairs.
    unique_mentions = []
    seen = set()

    for mention in mentions:
        key = (
            normalize(mention["input"]),
            mention["resolved"],
        )

        if key not in seen:
            seen.add(key)
            unique_mentions.append(mention)

    return unique_mentions


# ============================================================
# DEBUG HELPERS
# ============================================================

def get_player_aliases_for_debug(
    player_name: str,
) -> list[str]:
    """Return dynamically generated aliases for a player."""
    return sorted(generate_name_aliases(player_name))


# ============================================================
# CLI TESTS
# ============================================================

if __name__ == "__main__":
    print("Dataset:", DATASET_PATH)
    print("Players:", len(players))
    print("Teams:", len(teams))

    test_names = [
        "Rohit Sharma",
        "Gautam Gambhir",
        "RG Sharma",
        "G Gambhir",
        "ABD",
        "AB de Villiers",
    ]

    for name in test_names:
        print(f"\n{name}:")
        print(resolve_entity(name, "player"))

    print("\nTeam: KKR")
    print(resolve_entity("KKR", "team"))

    question = (
        "How many runs did Rohit Sharma and ABD score?"
    )

    print("\nQuestion:", question)
    print(find_player_mentions(question))
