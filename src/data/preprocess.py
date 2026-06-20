
import re


def minimal_preprocess(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    text = url_pattern.sub("LINK", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df):

    columns = [
        "body",
        "rule",
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2"
    ]

    try:

        for col in columns:

            df[col] = (
                df[col]
                .astype(str)
                .apply(minimal_preprocess)
            )

        return df

    except Exception as e:
        print(f"Preprocessing error : {e}")
        raise


def main():
    pass


if __name__ == "__main__":
    main()
