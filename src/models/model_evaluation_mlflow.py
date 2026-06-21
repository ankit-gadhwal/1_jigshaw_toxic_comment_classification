import os
import pandas as pd
import numpy as np
from pandas.io import json
import pickle
import json
import os
import torch
from torch.utils.data import DataLoader,Dataset
from src.models.BILSTM_model import BiLSTM_Attention
import yaml
import dagshub
import mlflow
from mlflow.models import infer_signature
import mlflow.xgboost

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name = "1_jigshaw_toxic_comment_classification"

os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

def get_or_create_experiment_id(experiment_name):

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"New experiment created: {experiment_id}")
        return experiment_id
    print(f"Existing experiment found: "f"{experiment.experiment_id}")
    return experiment.experiment_id

def load_params(filepath):
  try:
    with open(filepath,'r') as f:
      params = yaml.safe_load(f)
    return (
            params["bilstm"]["embedding_dim"],
            params["bilstm"]["hidden_dim"],
            params["bilstm"]["n_layers"],
            params["bilstm"]["dropout"],
            params["bilstm"]["batch_size"],
            params["bilstm_feature"]["max_len"],
            params["bilstm_feature"]["dim"]
        )
  except Exception as e:
    raise Exception(f"error occur while loading parameter {e}")


def load_test_data(filepath):
  try:
    test_df = pd.read_csv(filepath)
    return test_df
  except Exception as e:
    raise Exception(f"error while loading test data : {e}")

def load_vectorizer(filepath):
  try:
    with open(filepath,"rb") as f:
      vectorizer = pickle.load(f)
    return vectorizer
  except Exception as e:
    raise Exception(f"Error loading vectorizer : {e}")
  
def create_tfidf_features(vectorizer,merged_df_sub):
  try:
    X_test_tfidf = vectorizer.transform(merged_df_sub["merged_text"])
    return X_test_tfidf
  except Exception as e:
    raise Exception(f"Error creating tfidf features : {e}")

def load_model(filepath):
  try:
    with open(filepath,"rb") as f:
      model = pickle.load(f)
    return model
  except Exception as e:
    raise Exception(f"Error loading model : {e}")

def load_glove_embeddings(filepath):
  try:
    with open(filepath,"rb") as f:
      glove_embeddings = pickle.load(f)
    return glove_embeddings

  except Exception as e:
    raise Exception(f"Error loading glove embeddings: {e}")

def text_to_embedded_seq(text,embeddings_index,max_len,dim):
  words = text.lower().split()[:max_len]
  seq_embeds = [embeddings_index.get(word,np.zeros(dim)) for word in words]
  seq_embeds += [np.zeros(dim)]*(max_len - len(seq_embeds))
  return np.array(seq_embeds)

def create_sequence_features(texts,glove_embeddings,max_len,dim):
  try:
    X_seq = np.array([text_to_embedded_seq(text,glove_embeddings,max_len,dim) for text in texts])
    return X_seq
  except Exception as e:
    raise Exception(f"Error creating sequence features : {e}")

class SequenceDataset(Dataset):
  def __init__(self,X):
    self.X = X
  def __len__(self):
    return len(self.X)
  def __getitem__(self,idx):
    return torch.tensor(self.X[idx],dtype=torch.float32)

def create_dataloader(X,batch_size):
  try:
    dataset = SequenceDataset(X)
    loader = DataLoader(dataset,batch_size)
    return loader
  except Exception as e:
    raise Exception(f"Error creating loader : {e}")

def get_ml_probabilities(rf,xgb_model,nb,X_test_tfidf):
  try:
    probs_rf = rf.predict_proba(X_test_tfidf)[:,1]
    probs_xgb = xgb_model.predict_proba(X_test_tfidf)[:,1]
    probs_nb = nb.predict_proba(X_test_tfidf)[:,1]
    return (probs_rf,probs_xgb,probs_nb)
  except Exception as e:
    raise Exception(f"error occur while generating Ml probabilities : {e}")

def load_bilstm_model(device,embedding_dim,hidden_dim,n_layers,dropout):
  try:
    model = BiLSTM_Attention(embedding_dim,hidden_dim,n_layers,dropout)
    model.load_state_dict(torch.load("model/bilstm_model.pth",map_location = device))
    model.to(device)
    model.eval()
    return model
  except Exception as e:
    raise Exception(f"Error loading BiLSTM : {e}")

def get_bilstm_probabilities(model,loader,device):
  try:
    probs = []
    with torch.no_grad():
      for inputs in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs.extend(outputs.cpu().numpy())
    return np.array(probs)
  except Exception as e:
    raise Exception(f"Error generating BiLSTM probabilities : {e}")

def create_meta_features(probs_rf,probs_xgb,probs_nb,probs_bilstm):
    try:
        return np.column_stack([probs_rf,probs_xgb,probs_nb,probs_bilstm])
    except Exception as e:
        raise Exception(f"Error creating meta features : {e}")

def predict(ensemble_model,X_meta_test):
    try:
        pred_probs = ensemble_model.predict_proba(X_meta_test)[:,1]
        pred_labels = (pred_probs >= 0.5).astype(int)
        return pred_probs,pred_labels
    except Exception as e:
        raise Exception(f"Error generating predictions : {e}")

def save_predictions(comments,probs,labels,filepath):
  try:
    os.makedirs(os.path.dirname(filepath),exist_ok = True)
    result = []
    for comment,prob,label in zip(comments,probs,labels):
      result.append({"comment": comment,
        "probability":float(prob),
                    "prediction":int(label),"comment_type":"toxic"
                    if label == 1 else "non_toxic",
                    "confidence":"high" 
                    if prob > 0.9 or prob < 0.1 
                    else "medium"})
    with open(filepath,"w") as f:
      json.dump(result,f,indent = 4)
  except Exception as e:
    raise Exception(f"Error saving predictions : {e}")

def main():
    try:
        experimentId = get_or_create_experiment_id("Jigsaw Toxic Comment Evaluation")
        with mlflow.start_run(experiment_id = experimentId) as run:
          (embedding_dim,hidden_dim,n_layers,dropout,batch_size,max_len,dim) = load_params("params.yaml")

          # log parameters
          mlflow.log_param("embedding_dim",embedding_dim)

          mlflow.log_param("hidden_dim",hidden_dim)

          mlflow.log_param("n_layers",n_layers)

          mlflow.log_param("dropout",dropout)

          mlflow.log_param("batch_size",batch_size)

          mlflow.log_param("max_len",max_len)
          # Load test dataframe
          merged_df_sub = load_test_data("data/processed/merged_test.csv")
          # Create TF-IDF features
          vectorizer = load_vectorizer("data/processed/vectorizer.pkl")

          X_test_tfidf = create_tfidf_features(vectorizer,merged_df_sub)

          # Load classical ML models
          rf = load_model("model/random_forest.pkl")

          xgb_model = load_model("model/xgboost.pkl")

          nb = load_model("model/naive_bayes.pkl")

          # Get RF, XGB and NB probabilities
          (probs_rf,probs_xgb,probs_nb) = get_ml_probabilities(rf,xgb_model,nb,X_test_tfidf)

          # Load glove embeddings
          glove_embeddings = load_glove_embeddings("data/processed/glove_embeddings.pkl")

          # Create sequence features
          texts = merged_df_sub["merged_text"].values

          X_test_seq = create_sequence_features(texts,glove_embeddings,max_len=100,dim=100)

          # Create dataloader
          test_loader = create_dataloader(X_test_seq,batch_size)

          # Device
          device = torch.device("cuda"if torch.cuda.is_available()else "cpu")

          # Load BiLSTM model
          bilstm_model = load_bilstm_model(device,embedding_dim,hidden_dim,n_layers,dropout)

          # Get BiLSTM probabilities
          probs_bilstm = get_bilstm_probabilities(bilstm_model,test_loader,device)

          # Create meta features
          X_meta_test = create_meta_features(probs_rf,probs_xgb,probs_nb,probs_bilstm)

          # Load ensemble model
          ensemble_model = load_model("model/meta_ensemble.pkl")

          # Generate predictions
          pred_probs, pred_labels = predict(ensemble_model,X_meta_test)
  
          prediction_path = ("reports/predictions/test_predictions_mlflow.json")

          # Save results
          save_predictions(merged_df_sub["merged_text"],pred_probs,pred_labels,prediction_path)
          # metrics
          num_toxic = int(np.sum(pred_labels))

          num_non_toxic = int(len(pred_labels) -num_toxic)

          avg_probability = float(np.mean(pred_probs))

          toxic_ratio = float(num_toxic /len(pred_labels))

          mlflow.log_metric("num_toxic",num_toxic)

          mlflow.log_metric("num_non_toxic",num_non_toxic)

          mlflow.log_metric("avg_probability",avg_probability)

          mlflow.log_metric("toxic_ratio",toxic_ratio)
          
           # log prediction file
          mlflow.log_artifact(prediction_path)

            # log models
          
          mlflow.xgboost.log_model(xgb_model,artifact_path="xgboost")
          
          mlflow.sklearn.log_model(rf,artifact_path="random_forest")

          mlflow.sklearn.log_model(nb,artifact_path="naive_bayes")

          mlflow.pytorch.log_model(pytorch_model=bilstm_model,artifact_path="bilstm",serialization_format="pickle")

          signature = infer_signature(X_meta_test,ensemble_model.predict(X_meta_test))

          mlflow.sklearn.log_model(ensemble_model,artifact_path="meta_ensemble",signature=signature)

          run_info = {"run_id":run.info.run_id,
                "model_name":"meta_ensemble"}

          os.makedirs("reports",exist_ok=True)

          with open("reports/run_info.json","w") as file:
            json.dump(run_info,file,indent=4)
          mlflow.log_artifact("reports/run_info.json")

          print("RUN ID =",run.info.run_id)
  
          print("Predictions generated successfully.")

    except Exception as e:
        raise Exception(f"Error in evaluation pipeline : {e}")

if __name__ == "__main__":

    main()