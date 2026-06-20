import pandas as pd
import numpy as np
import pickle

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def data_load(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Processed dataframe loaded successfully.")
        return df
    except Exception as e:
        raise Exception(f"Error while loading data : {e}")

def create_tfidf_features(merged_df,max_features=10000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform( merged_df["merged_text"])
    y = merged_df["rule_violation"].values
    return (X_tfidf,y,vectorizer)

def split_tfidf_data(X_tfidf,y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_tfidf,y,stratify=y,test_size=0.15,random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,y_train_val,test_size=0.15, stratify=y_train_val,random_state=42
    )

    return (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test

    )


def save_vectorizer(vectorizer,path):

    try:

        with open(path, "wb") as f:

            pickle.dump(vectorizer,f)

        print("Vectorizer saved successfully.")

    except Exception as e:

        raise Exception(f"Error while saving vectorizer : {e}")

def main():

    try:

        print("Loading processed dataframe...")
        data_path = "data/processed/merged_train.csv"
        merged_df = data_load(data_path)
        print("Creating TF-IDF features...")
        X_tfidf, y, vectorizer = create_tfidf_features(merged_df)
        print("Splitting train-validation-test...")

        (X_train,X_val,X_test,y_train,y_val,y_test) = split_tfidf_data(X_tfidf, y)
        base_path = ("data/processed/")
        save_npz(base_path + "X_train_tfidf.npz",X_train)
        save_npz(base_path + "X_val_tfidf.npz",X_val)
        save_npz(base_path + "X_test_tfidf.npz",X_test)
        np.save(base_path + "y_train.npy",y_train)
        np.save( base_path + "y_val.npy",y_val)
        np.save(base_path + "y_test.npy",y_test)
        save_vectorizer(vectorizer,base_path + "vectorizer.pkl")
        print("TF-IDF artifacts saved successfully.")
        print("X_train shape :",X_train.shape)
        print("X_val shape :",X_val.shape)
        print("X_test shape :",X_test.shape)

    except Exception as e:
       raise Exception(f"Error in tfidf feature pipeline : {e}")
        
if __name__ == "__main__":
    main()
