
import pandas as pd


def create_augmented_dataframe(df):

    try:

        new_rows = []

        for _, row in df.iterrows():

            rule_text = row["rule"]

            new_rows.append(
                {
                    "merged_text":
                    rule_text + " " + row["body"],

                    "rule_violation": row["rule_violation"]
                }
            )
            new_rows.append(
                {
                    "merged_text":
                    rule_text + " " + row["positive_example_1"],

                    "rule_violation":
                    1
                }
            )

            new_rows.append(
                {
                    "merged_text":
                    rule_text + " " + row["positive_example_2"],

                    "rule_violation":
                    1
                }
            )

            new_rows.append(
                {
                    "merged_text":
                    rule_text + " " + row["negative_example_1"],

                    "rule_violation":
                    0
                }
            )

            new_rows.append(
                {
                    "merged_text":
                    rule_text + " " + row["negative_example_2"],

                    "rule_violation":
                    0
                }
            )

        merged_df = pd.DataFrame(new_rows)

        return merged_df

    except Exception as e:
        print(f"Augmentation error : {e}")
        raise


def main():
    pass


if __name__ == "__main__":
    main()
