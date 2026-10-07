from query_parser import parse_query
from entity_resolver import resolve_query_plan
from query_executor import execute_query


# Display possible entity matches
def display_options(options):

    print("\nI found multiple possible matches:\n")

    for index, option in enumerate(
        options,
        start=1
    ):
        print(f"{index}. {option}")


# Get the user's selected option
def get_user_choice(options):

    while True:

        choice = input(
            "\nEnter the number of the correct "
            "player/team: "
        ).strip()

        try:

            choice_number = int(choice)

            if 1 <= choice_number <= len(options):
                return options[choice_number - 1]

            print(
                f"Please enter a number between "
                f"1 and {len(options)}."
            )

        except ValueError:

            print("Please enter a valid number.")


# Apply the selected entity to the QueryAnalysis
def apply_entity_choice(
    analysis,
    ambiguity,
    selected_value
):

    updated = analysis.model_copy(
        deep=True
    )

    query = updated.query

    if query is None:
        return updated

    target = str(
        ambiguity.user_value
    ).strip().lower()

    for subject in query.subjects:

        if (
            subject.entity_type
            == ambiguity.entity_type
            and str(subject.value).strip().lower()
            == target
        ):
            subject.value = selected_value

    if query.comparison is not None:

        for subject in query.comparison.subjects:

            if (
                subject.entity_type
                == ambiguity.entity_type
                and str(subject.value).strip().lower()
                == target
            ):
                subject.value = selected_value

    for filter_item in query.filters:

        value = filter_item.value

        if (
            isinstance(value, str)
            and value.strip().lower() == target
        ):
            filter_item.value = selected_value

        elif isinstance(value, list):

            filter_item.value = [
                selected_value
                if (
                    isinstance(item, str)
                    and item.strip().lower() == target
                )
                else item
                for item in value
            ]

    updated.ambiguities = []
    updated.clarification_question = None
    updated.status = "ready"

    return resolve_query_plan(
        updated
    )


# Run the IPL Copilot
def main():

    print("IPL Copilot")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print("\nExiting IPL Copilot...")
            break

        if not question:
            continue

        try:

            # Parse the user question
            query_plan = parse_query(
                question
            )

            print("\nQuery Plan:")
            print(query_plan)

            # Resolve entities
            resolution = resolve_query_plan(
                query_plan
            )

            if resolution.status == "ready":

                resolved_plan = resolution.query

                if resolved_plan is None:

                    print(
                        "\nQuery Error:"
                        " No executable query was created."
                    )

                    continue

            elif (
                resolution.status
                == "clarification_required"
            ):

                if not resolution.ambiguities:

                    print(
                        "\nClarification:"
                    )

                    print(
                        resolution.clarification_question
                        or "I need more information."
                    )

                    continue

                ambiguity = (
                    resolution.ambiguities[0]
                )

                options = ambiguity.candidates

                if not options:

                    print(
                        "\nClarification:"
                    )

                    print(
                        resolution.clarification_question
                        or (
                            "I couldn't identify "
                            "the requested entity."
                        )
                    )

                    continue

                print(
                    f"\nI found multiple possible "
                    f"matches for "
                    f"'{ambiguity.user_value}':"
                )

                display_options(
                    options
                )

                selected_value = get_user_choice(
                    options
                )

                print(
                    f"\nYou selected: "
                    f"{selected_value}"
                )

                resolution = apply_entity_choice(
                    resolution,
                    ambiguity,
                    selected_value
                )

                if resolution.status != "ready":

                    print(
                        "\nClarification:"
                    )

                    print(
                        resolution.clarification_question
                        or "The selected entity could not be resolved."
                    )

                    continue

                resolved_plan = resolution.query

                if resolved_plan is None:

                    print(
                        "\nQuery Error:"
                        " No executable query was created."
                    )

                    continue

            else:

                print(
                    "\nQuery Error:"
                )

                print(
                    resolution.reason
                    or "This query is not supported."
                )

                continue

            print(
                "\nResolved Query Plan:"
            )

            print(
                resolved_plan
            )

            # Execute the query
            result = execute_query(
                resolved_plan
            )

            # Display the result
            print("\nResult:")
            print(result)

        except ValueError as e:

            print("\nQuery Error:")
            print(e)

        except Exception as e:

            print("\nUnexpected Error:")
            print(e)


if __name__ == "__main__":
    main()