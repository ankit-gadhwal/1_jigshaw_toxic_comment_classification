
import pandas as pd


def load_data(path):
    """
    Load csv file into dataframe.
    """
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}")
        return df

    except Exception as e:
        print(f"Error while loading data : {e}")
        raise


def main():

    train_path = "data/raw/jigsaw_toxic_comment_classification_train.csv"

    test_path = "data/raw/jigsaw_toxic_comment_classification_test.csv"


    try:
        train_df = load_data(train_path)
        test_df = load_data(test_path)

        print("Train shape :", train_df.shape)
        print("Test shape :", test_df.shape)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()

