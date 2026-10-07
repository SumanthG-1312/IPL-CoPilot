from sql_query_planner import generate_sql
from sql_executor import execute_sql


def main():

    print("IPL Copilot - End-to-End SQL Pipeline")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:
            break

        if not question:
            continue

        try:

            planner_result = generate_sql(
                question
            )

            print("\nResolved Question:")
            print(
                planner_result.get(
                    "resolved_question",
                    question
                )
            )

            if planner_result["status"] != "ready":

                print("\nPlanner:")
                print(
                    planner_result.get(
                        "reason",
                        "Question is unsupported."
                    )
                )

                continue

            sql = planner_result["sql"]

            print("\nGenerated SQL:")
            print(sql)

            result = execute_sql(
                sql
            )

            print("\nResult:")
            print(
                result.to_string(
                    index=False
                )
            )

        except Exception as error:

            print("\nPipeline Error:")
            print(error)


if __name__ == "__main__":
    main()