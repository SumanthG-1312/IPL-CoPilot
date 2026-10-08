import os
import re
from difflib import SequenceMatcher
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

DATASET_PATH = os.getenv(
    "IPL_DATASET_PATH",
    "data/ipl_ball_by_ball_cleaned.csv",
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value: str) -> str:
    """
    Normalize text for matching.
    """
    if value is None:
        return ""

    value = str(value).strip().lower()

    value = value.replace("’", "'")

    value = re.sub(
        r"['’]",
        "",
        value,
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def compact(value: str) -> str:
    """
    Remove spaces for compact aliases.
    """
    return normalize(value).replace(" ", "")


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    return pd.read_csv(
        DATASET_PATH,
        low_memory=False,
    )


df = load_dataset()


# ============================================================
# EXTRACT PLAYERS
# ============================================================

def collect_players(dataframe: pd.DataFrame):
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

        series = (
            dataframe[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        for value in series:

            if not value:
                continue

            if value.lower() in {
                "nan",
                "none",
                "null",
            }:
                continue

            values.add(value)

    return sorted(
        values,
        key=lambda value: normalize(value),
    )


players = collect_players(df)


# ============================================================
# EXTRACT TEAMS
# ============================================================

def collect_teams(dataframe: pd.DataFrame):

    if "BattingTeam" not in dataframe.columns:
        return []

    values = (
        dataframe["BattingTeam"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(
        {
            value
            for value in values
            if value
        },
        key=lambda value: normalize(value),
    )


teams = collect_teams(df)


# ============================================================
# DYNAMIC NAME ALIAS GENERATION
# ============================================================

def generate_name_aliases(name: str):
    """
    Generate aliases dynamically from the stored dataset name.

    Example:

        AB de Villiers

    produces aliases such as:

        ab de villiers
        abdevilliers
        abd
        adv
        abv
        villiers
        de villiers

    No player names are hardcoded.
    """

    normalized = normalize(name)

    if not normalized:
        return set()

    parts = normalized.split()

    aliases = set()

    # --------------------------------------------------------
    # Full name
    # --------------------------------------------------------

    aliases.add(normalized)

    # --------------------------------------------------------
    # Compact full name
    # --------------------------------------------------------

    aliases.add(
        normalized.replace(" ", "")
    )

    # --------------------------------------------------------
    # Individual name components
    # --------------------------------------------------------

    for part in parts:

        if len(part) >= 2:
            aliases.add(part)

    # --------------------------------------------------------
    # Consecutive name combinations
    # --------------------------------------------------------

    for start in range(len(parts)):

        for end in range(
            start + 2,
            len(parts) + 1,
        ):

            phrase = " ".join(
                parts[start:end]
            )

            if len(phrase) >= 3:
                aliases.add(phrase)

    # --------------------------------------------------------
    # Initials
    #
    # Example:
    # AB de Villiers -> adv
    # Virat Kohli    -> vk
    # --------------------------------------------------------

    initials = "".join(
        part[0]
        for part in parts
        if part
    )

    if len(initials) >= 2:
        aliases.add(initials)

    # --------------------------------------------------------
    # First stored token + initials of remaining tokens
    #
    # This is important for:
    #
    # AB de Villiers
    # -> AB + D
    # -> ABD
    #
    # --------------------------------------------------------

    if len(parts) >= 2:

        first_plus_remaining_initials = (
            parts[0]
            + "".join(
                part[0]
                for part in parts[1:]
                if part
            )
        )

        if len(
            first_plus_remaining_initials
        ) >= 2:

            aliases.add(
                first_plus_remaining_initials
            )

    # --------------------------------------------------------
    # First token + last initial
    # --------------------------------------------------------

    if len(parts) >= 2:

        first_plus_last_initial = (
            parts[0]
            + parts[-1][0]
        )

        if len(
            first_plus_last_initial
        ) >= 2:

            aliases.add(
                first_plus_last_initial
            )

    return {
        normalize(alias)
        for alias in aliases
        if normalize(alias)
    }


# ============================================================
# BUILD PLAYER ALIAS INDEX
# ============================================================

def build_player_alias_index():

    alias_candidates = defaultdict(set)

    for player in players:

        for alias in generate_name_aliases(
            player
        ):

            alias_candidates[alias].add(
                player
            )

    return alias_candidates


PLAYER_ALIAS_CANDIDATES = (
    build_player_alias_index()
)


# Unique aliases only.
#
# Example:
#
# "abd" -> {"AB de Villiers"}
#
# becomes:
#
# PLAYER_ALIASES["abd"] = "AB de Villiers"
#

PLAYER_ALIASES = {}

for alias, candidates in (
    PLAYER_ALIAS_CANDIDATES.items()
):

    if len(candidates) == 1:

        PLAYER_ALIASES[alias] = next(
            iter(candidates)
        )


# ============================================================
# BUILD TEAM ALIAS INDEX
# ============================================================

def generate_team_aliases(name: str):

    normalized = normalize(name)

    if not normalized:
        return set()

    parts = normalized.split()

    aliases = {
        normalized,
        normalized.replace(" ", ""),
    }

    # Individual meaningful words
    for part in parts:

        if len(part) >= 2:
            aliases.add(part)

    # Initials
    initials = "".join(
        part[0]
        for part in parts
        if part
    )

    if len(initials) >= 2:
        aliases.add(initials)

    return {
        normalize(alias)
        for alias in aliases
        if normalize(alias)
    }


def build_team_alias_index():

    alias_candidates = defaultdict(set)

    for team in teams:

        for alias in generate_team_aliases(
            team
        ):

            alias_candidates[alias].add(
                team
            )

    return alias_candidates


TEAM_ALIAS_CANDIDATES = (
    build_team_alias_index()
)


TEAM_ALIASES = {}

for alias, candidates in (
    TEAM_ALIAS_CANDIDATES.items()
):

    if len(candidates) == 1:

        TEAM_ALIASES[alias] = next(
            iter(candidates)
        )


# ============================================================
# EXACT MATCH HELPERS
# ============================================================

def exact_player_match(value: str):

    normalized = normalize(value)

    if not normalized:
        return None

    # Canonical player name
    for player in players:

        if normalize(player) == normalized:

            return player

    # Dynamic alias
    return PLAYER_ALIASES.get(
        normalized
    )


def exact_team_match(value: str):

    normalized = normalize(value)

    if not normalized:
        return None

    # Canonical team name
    for team in teams:

        if normalize(team) == normalized:

            return team

    # Dynamic alias
    return TEAM_ALIASES.get(
        normalized
    )


# ============================================================
# ENTITY RESOLUTION
# ============================================================

def resolve_entity(
    value: str,
    entity_type: str,
):

    normalized_value = normalize(value)

    if not normalized_value:

        return {
            "input": value,
            "resolved": None,
            "resolved_value": None,
            "method": "empty",
            "status": "not_found",
        }

    # ========================================================
    # PLAYER
    # ========================================================

    if entity_type == "player":

        # ----------------------------------------------------
        # Exact canonical name
        # ----------------------------------------------------

        for player in players:

            if normalize(player) == normalized_value:

                return {
                    "input": value,
                    "resolved": player,
                    "resolved_value": player,
                    "method": "exact",
                    "score": 1.0,
                    "status": "resolved",
                }

        # ----------------------------------------------------
        # Exact dynamic alias
        # ----------------------------------------------------

        candidates = (
            PLAYER_ALIAS_CANDIDATES.get(
                normalized_value,
                set(),
            )
        )

        if len(candidates) == 1:

            player = next(
                iter(candidates)
            )

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
                "candidates": sorted(
                    candidates
                ),
                "status": "ambiguous",
            }

        # ----------------------------------------------------
        # Fuzzy matching
        # ----------------------------------------------------

        scored = []

        for player in players:

            player_normalized = normalize(
                player
            )

            score = SequenceMatcher(
                None,
                normalized_value,
                player_normalized,
            ).ratio()

            scored.append(
                (
                    score,
                    player,
                )
            )

        scored.sort(
            reverse=True,
            key=lambda item: item[0],
        )

        if not scored:

            return {
                "input": value,
                "resolved": None,
                "resolved_value": None,
                "method": "not_found",
                "status": "not_found",
            }

        best_score, best_player = (
            scored[0]
        )

        second_score = (
            scored[1][0]
            if len(scored) > 1
            else 0
        )

        # More conservative thresholds.
        threshold = (
            0.92
            if len(normalized_value) <= 4
            else 0.88
        )

        if best_score < threshold:

            return {
                "input": value,
                "resolved": None,
                "resolved_value": None,
                "method": "fuzzy",
                "score": best_score,
                "status": "not_found",
            }

        # Avoid unsafe fuzzy resolutions.
        if (
            best_score - second_score
            < 0.05
        ):

            ambiguous_candidates = [
                player
                for score, player in scored[:5]
                if (
                    best_score - score
                ) < 0.05
            ]

            return {
                "input": value,
                "resolved": None,
                "resolved_value": None,
                "method": "fuzzy_ambiguous",
                "score": best_score,
                "candidates": ambiguous_candidates,
                "status": "ambiguous",
            }

        return {
            "input": value,
            "resolved": best_player,
            "resolved_value": best_player,
            "method": "fuzzy",
            "score": best_score,
            "status": "resolved",
        }

    # ========================================================
    # TEAM
    # ========================================================

    if entity_type == "team":

        # Exact canonical
        for team in teams:

            if normalize(team) == normalized_value:

                return {
                    "input": value,
                    "resolved": team,
                    "resolved_value": team,
                    "method": "exact",
                    "score": 1.0,
                    "status": "resolved",
                }

        # Exact alias
        candidates = (
            TEAM_ALIAS_CANDIDATES.get(
                normalized_value,
                set(),
            )
        )

        if len(candidates) == 1:

            team = next(
                iter(candidates)
            )

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
                "candidates": sorted(
                    candidates
                ),
                "status": "ambiguous",
            }

        # Fuzzy team matching

        scored = []

        for team in teams:

            team_normalized = normalize(
                team
            )

            score = SequenceMatcher(
                None,
                normalized_value,
                team_normalized,
            ).ratio()

            scored.append(
                (
                    score,
                    team,
                )
            )

        scored.sort(
            reverse=True,
            key=lambda item: item[0],
        )

        if not scored:

            return {
                "input": value,
                "resolved": None,
                "resolved_value": None,
                "method": "not_found",
                "status": "not_found",
            }

        best_score, best_team = scored[0]

        second_score = (
            scored[1][0]
            if len(scored) > 1
            else 0
        )

        threshold = 0.88

        if best_score < threshold:

            return {
                "input": value,
                "resolved": None,
                "resolved_value": None,
                "method": "fuzzy",
                "score": best_score,
                "status": "not_found",
            }

        if (
            best_score - second_score
            < 0.05
        ):

            candidates = [
                team
                for score, team in scored[:5]
                if (
                    best_score - score
                ) < 0.05
            ]

            return {
                "input": value,
                "resolved": None,
                "resolved_value": None,
                "method": "fuzzy_ambiguous",
                "score": best_score,
                "candidates": candidates,
                "status": "ambiguous",
            }

        return {
            "input": value,
            "resolved": best_team,
            "resolved_value": best_team,
            "method": "fuzzy",
            "score": best_score,
            "status": "resolved",
        }

    # ========================================================
    # UNSUPPORTED ENTITY TYPE
    # ========================================================

    return {
        "input": value,
        "resolved": None,
        "resolved_value": None,
        "method": "unsupported_entity_type",
        "status": "unsupported",
    }


# ============================================================
# FIND PLAYER MENTIONS
# ============================================================

def find_player_mentions(question: str):
    """
    Find dynamically generated player aliases
    inside a natural-language question.
    """

    normalized_question = normalize(
        question
    )

    if not normalized_question:
        return []

    mentions = []

    # Longest aliases first.
    #
    # This prevents a short alias from stealing
    # a longer player name.
    aliases = sorted(
        PLAYER_ALIAS_CANDIDATES.keys(),
        key=lambda alias: (
            len(alias),
            alias.count(" "),
        ),
        reverse=True,
    )

    occupied_ranges = []

    for alias in aliases:

        pattern = re.compile(
            rf"\b{re.escape(alias)}\b",
            re.IGNORECASE,
        )

        for match in pattern.finditer(
            normalized_question
        ):

            start = match.start()
            end = match.end()

            # Skip overlapping shorter aliases.
            overlap = False

            for existing_start, existing_end in occupied_ranges:

                if (
                    start < existing_end
                    and end > existing_start
                ):
                    overlap = True
                    break

            if overlap:
                continue

            candidates = (
                PLAYER_ALIAS_CANDIDATES.get(
                    alias,
                    set(),
                )
            )

            if len(candidates) != 1:
                continue

            canonical_player = next(
                iter(candidates)
            )

            mentions.append(
                {
                    "input": match.group(0),
                    "value": match.group(0),
                    "resolved": canonical_player,
                    "resolved_value": canonical_player,
                    "status": "resolved",
                }
            )

            occupied_ranges.append(
                (start, end)
            )

    # Longest match first.
    mentions.sort(
        key=lambda item: len(
            item["input"]
        ),
        reverse=True,
    )

    return mentions


# ============================================================
# DEBUG INFORMATION
# ============================================================

def get_player_aliases_for_debug(
    player_name: str
):
    """
    Useful for testing generated aliases.
    """

    return sorted(
        generate_name_aliases(
            player_name
        )
    )


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Players:",
        len(players),
    )

    print(
        "Teams:",
        len(teams),
    )

    print(
        "\nABD:",
        PLAYER_ALIASES.get("abd"),
    )

    print(
        "\nAB de Villiers aliases:"
    )

    print(
        get_player_aliases_for_debug(
            "AB de Villiers"
        )
    )

    print(
        "\nQuestion mentions:"
    )

    print(
        find_player_mentions(
            "How many runs did ABD score?"
        )
    )

    print(
        "\nDirect resolution:"
    )

    print(
        resolve_entity(
            "ABD",
            "player",
        )
    )