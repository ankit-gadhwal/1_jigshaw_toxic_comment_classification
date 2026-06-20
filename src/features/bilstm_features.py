import pandas as pd
import numpy as np
import yaml
import pickle
from sklearn.model_selection import train_test_split
def load_dataframe(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(
            f"Error loading dataframe : {e}"
        )
def load_params(filepath):
  try:
    with open(filepath,"r") as file:
      params = yaml.safe_load(file)
    return params["bilstm_feature"]["max_len"],params["bilstm_feature"]["dim"]
  except Exception as e:
    raise Exception(f"error occured while parameter loading {e}")
def load_glove_embeddings(filepath):
    try:
        with open(filepath, "rb") as f:
            glove_embeddings = pickle.load(f)
        return glove_embeddings
    except Exception as e:
        raise Exception(
            f"Error loading glove embeddings : {e}"
        )

def text_to_embedded_seq(text,embeddings_index,max_len,dim):
    try:
        words = text.lower().split()[:max_len]
        seq_embeds = [
            embeddings_index.get(w,np.zeros(dim))
            for w in words
        ]
        seq_embeds += [np.zeros(dim)] * (max_len - len(seq_embeds))
        return np.array(
            seq_embeds
        )
    except Exception as e:
        raise Exception(
            f"Error creating sequence : {e}"
        )

def create_sequence_features(texts,glove_embeddings,max_len,dim):
    try:
        X = np.array([text_to_embedded_seq(text,glove_embeddings,max_len,dim)for text in texts])
        return X
    except Exception as e:
        raise Exception(f"Error creating sequence features : {e}")

def split_data(texts,labels):
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            texts,labels,test_size=0.15,stratify=labels,random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,y_train_val,test_size=0.15,
            stratify=y_train_val,random_state=42
        )
        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )
    except Exception as e:
        raise Exception(f"Error splitting data : {e}")

def save_dataset(
        X_train_seq,X_val_seq,X_test_seq,y_train,
        y_val,y_test,base_path):
    try:
        np.save(
            base_path +
            "X_train_seq.npy",
            X_train_seq
        )
        np.save(
            base_path +
            "X_val_seq.npy",
            X_val_seq
        )
        np.save(
            base_path +
            "X_test_seq.npy",
            X_test_seq
        )
        np.save(
            base_path +
            "y_train_seq.npy",
            y_train
        )
        np.save(
            base_path +
            "y_val_seq.npy",
            y_val
        )

        np.save(
            base_path +
            "y_test_seq.npy",
            y_test
        )

    except Exception as e:

        raise Exception(
            f"Error saving dataset : {e}"
        )


def main():
  try:
    data_path = "data/processed/merged_train.csv"
    merged_df = load_dataframe(data_path)
    glove_path = "data/processed/glove_embeddings.pkl"
    glove_embeddings = load_glove_embeddings(glove_path)
    texts = merged_df["merged_text"].values
    labels = merged_df["rule_violation"].values
    params_path = "params.yaml"
    max_len,dim = load_params(params_path)
    (X_train_text,X_val_text,X_test_text,y_train,y_val,y_test) = split_data(texts,labels)

    X_train_seq = create_sequence_features(
    X_train_text,
    glove_embeddings,max_len,dim
    )

    X_val_seq = create_sequence_features(
    X_val_text,
    glove_embeddings,max_len,dim
    )

    X_test_seq = create_sequence_features(
    X_test_text,
    glove_embeddings,max_len,dim
    )
    base_path = "data/processed/"
    save_dataset(
        X_train_seq,X_val_seq,X_test_seq,
        y_train,y_val,y_test,base_path)
  except Exception as e:
    raise Exception(f"error occured while bilstm feature {e}")
if __name__ == "__main__":
  main()