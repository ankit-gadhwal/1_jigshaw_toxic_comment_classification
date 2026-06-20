from sklearn.model_selection import train_test_split


def split_glove_data(X, y):

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_valid, y_train, y_valid


def split_tfidf_data(X_tfidf, y):

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_tfidf,
        y,
        stratify=y,
        test_size=0.15,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.15,
        stratify=y_train_val,
        random_state=42
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def main():
    pass


if __name__ == "__main__":
    main()