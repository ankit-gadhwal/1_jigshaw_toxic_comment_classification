import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import yaml
def data_load(filepath:str)-> pd.DataFrame:
  try:
    return pd.read_csv(filepath)
  except Exception as e:
    raise Exception(f"error occured while data loading {e}")

def load_params(filepath):
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return params["glove_features"]["embedding_dim"],params["glove_features"]["test_sizes"]
    except Exception as e:
      raise Exception(f"error occur while parameter loading {e}")
def split_glove_data(X, y,test_sz):

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_sz,
        random_state=42
    )

    return X_train, X_valid, y_train, y_valid
  
def save_embeddings(
        glove_embeddings,
        filepath
):

    try:

        with open(
                filepath,
                "wb"
        ) as f:

            pickle.dump(
                glove_embeddings,
                f
            )

    except Exception as e:

        raise Exception(
            f"Error saving model : {e}"
        )

def load_glove_embeddings(filepath):

    embeddings_index = {}

    try:

        with open(filepath, "r", encoding="utf8") as f:

            for line in f:

                values = line.split()

                word = values[0]

                vector = np.asarray(
                    values[1:],
                    dtype="float32"
                )

                embeddings_index[word] = vector

        return embeddings_index

    except Exception as e:
        print(f"GloVe loading error : {e}")
        raise


def text_to_avg_embedding(
        text,
        embeddings,
        dim
):

    words = text.split()

    valid_vectors = [
        embeddings[word]
        for word in words
        if word in embeddings
    ]

    if len(valid_vectors) == 0:
        return np.zeros(dim)

    avg_vector = np.mean(
        valid_vectors,
        axis=0
    )

    return avg_vector


def create_glove_features(
        merged_df,
        embeddings,
        embedding_dim
):

    X_embeddings = np.vstack(

        merged_df["merged_text"].apply(
            lambda x:
            text_to_avg_embedding(x,embeddings,embedding_dim)).values)
    
    y = merged_df["rule_violation"].values

    return X_embeddings, y


def main(): 
  try: 
    params_path = "params.yaml"
    embedding_dim,test_sz = load_params(params_path)
    glove_path = ( "glove.6B.100d.txt" ) 
    print("Loading processed data...")
    data_path = "data/processed/merged_train.csv" 
    merged_df = data_load(data_path) 
    print("Loading GloVe embeddings...") 
    glove_embeddings = load_glove_embeddings(glove_path) 
    path_embedding = "data/processed/"
    save_embeddings(glove_embeddings,path_embedding + "glove_embeddings.pkl")
    print("Creating GloVe features...") 
    X_embeddings, y = create_glove_features( merged_df, glove_embeddings, embedding_dim )
    X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = split_glove_data(X_embeddings,y,test_sz)
    np.save("data/processed/X_train_emb.npy", X_train_emb) 
    np.save("data/processed/y_train_emb.npy", y_train_emb)
    np.save("data/processed/X_valid_emb.npy", X_valid_emb) 
    np.save("data/processed/y_valid_emb.npy", y_valid_emb)  
    print("Features saved successfully.") 
    print("Shape of X : ", X_embeddings.shape) 
    print("Shape of y : ", y.shape)
  except Exception as e: 
    raise Exception( f"Error in glove feature pipeline : {e}" )

if __name__ == "__main__":
    main()