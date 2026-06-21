import numpy as np
import pandas as pd
import re
import os
from preprocess import preprocess_dataframe
from data_augment import create_augmented_dataframe
from data_collection import load_data

def create_test_dataframe(df_sub):
    try:
        new_rows = []

        for _, row in df_sub.iterrows():

            rule_text = row["rule"]

            new_rows.append({
                    "merged_text": rule_text + " " + row["body"],
                     "row_id": row["row_id"]
                }  
            )

        merged_df_sub = pd.DataFrame(new_rows)

        return merged_df_sub

    except Exception as e:
        print(f"Error while creating test dataframe : {e}")
        raise


def save_dataframe(df, path):

    try:

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        df.to_csv(
            path,
            index=False
        )

        print(f"Saved file to {path}")

    except Exception as e:

        print(f"Saving error : {e}")

        raise


def main():

    try:

        train_path = "jigsaw_toxic_comment_classification_train.csv"

        test_path = "jigsaw_toxic_comment_classification_test.csv"

        print("Loading train data...")
        train_df = load_data(train_path)

        print("Loading test data...")
        test_df = load_data(test_path)

        print("Preprocessing train data...")
        train_df = preprocess_dataframe(train_df)

        print("Preprocessing test data...")
        test_df = preprocess_dataframe(test_df)

        print("Creating augmented train dataframe...")
        merged_train_df = create_augmented_dataframe(
            train_df
        )

        print("Creating merged test dataframe...")
        merged_test_df = create_test_dataframe(
            test_df
        )

        save_dataframe(
            merged_train_df,
            "data/processed/merged_train.csv"
        )
        save_dataframe(merged_test_df,
            "data/processed/merged_test.csv"
        )

        print("\nData preparation completed successfully.")

        print(
            "Train shape :",
            merged_train_df.shape
        )

        print(
            "Test shape :",
            merged_test_df.shape
        )

    except Exception as e:

        print(
            f"Data preparation pipeline failed : {e}"
        )

        raise


if __name__ == "__main__":

    main()

