import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import torch
import torch.nn
from torch.utils.data import Dataset,DataLoader
import pickle
from BILSTM_model import BiLSTM_Attention
from sklearn.linear_model import LogisticRegression
import json 
from sklearn.metrics import f1_score, classification_report
import os
import yaml

def load_params(filepath):
  try:
    with open(filepath,'r') as f:
      params = yaml.safe_load(f)
    return (
            params["bilstm"]["embedding_dim"],
            params["bilstm"]["hidden_dim"],
            params["bilstm"]["n_layers"],
            params["bilstm"]["dropout"],
            params["bilstm"]["batch_size"]
        )
  except Exception as e:
    raise Exception(f"error occured while parameter loading {e}")
def load_tfidf_data(base_path):
  try:
    X_val_tfidf = load_npz(base_path + "X_val_tfidf.npz")
    X_test_tfidf = load_npz(base_path + "X_test_tfidf.npz")
    y_val = np.load(base_path + "y_val.npy")
    y_test = np.load(base_path + "y_test.npy")
    return (X_val_tfidf,X_test_tfidf,y_val,y_test)

  except Exception as e:
    raise Exception(f"Error loading tfidf data : {e}")

def load_bilstm_data(base_path):
  try:
    X_val_seq = np.load(base_path + "X_val_seq.npy")
    X_test_seq = np.load(base_path + "X_test_seq.npy")
    y_val_seq = np.load(base_path + "y_val_seq.npy")
    y_test_seq = np.load(base_path + "y_test_seq.npy")
    return (X_val_seq,X_test_seq,y_val_seq,y_test_seq)
  except Exception as e:
    raise Exception(f"error occured while loading sequence data : {e}")

class SequenceDataset(Dataset):
  def __init__(self,X,y):
    self.X = X
    self.y = y
  def __len__(self):
    return len(self.y)
  
  def __getitem__(self, idx):
    return(torch.tensor(self.X[idx],dtype=torch.float32),
    torch.tensor(self.y[idx],dtype=torch.float32))


def create_dataloader(X,y,batch_size):
  try:
    dataset = SequenceDataset(X,y)
    loader = DataLoader(dataset,batch_size = batch_size)
    return loader
  except Exception as e:
    raise Exception(f"error occur in dataloader: {e}")
  
def load_ml_models(filepath):
  try:
    with open(filepath,"rb") as f:
      load_model = pickle.load(f)
    return load_model
  except Exception as e:
    raise Exception(f"error occur while loading model {e}")

def load_bilstm_model(device,embedding_dim,hidden_dim,n_layers,dropout):
  try:
    model = BiLSTM_Attention(embedding_dim,hidden_dim,n_layers,dropout)
    model.load_state_dict(torch.load("model/bilstm_model.pth",map_location = device))
    model.to(device)
    model.eval()
    return model
  
  except Exception as e:
    raise Exception(f"Error loading BiLSTM: {e}")

def get_device():
    try:
        return torch.device("cuda"if torch.cuda.is_available()
            else "cpu")
    except Exception as e:
        raise Exception(f"Error getting device : {e}")
  
def get_ml_probabilities(rf,Xgb_model,nb,X):
  try:
    probs_rf = rf.predict_proba(X)[:,1]
    probs_xgb = Xgb_model.predict_proba(X)[:,1]
    probs_nb = nb.predict_proba(X)[:,1]
    return (probs_rf,probs_xgb,probs_nb)
  
  except Exception as e:
    raise Exception(f"Error generating ML probabilities : {e}")

def get_bilstm_probabilities(model,loader,device):
  try:
    probs = []
    with torch.no_grad():
      for inputs,_ in loader:
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

def train_meta_model(X_meta,y):
  try:
    meta_clf = LogisticRegression()
    meta_clf.fit(X_meta,y)
    return meta_clf
  
  except Exception as e:
    raise Exception(f"Error training meta model: {e}")

def evaluate_ensemble(meta_model,X_meta_test,y_test):
  try:
    preds = meta_model.predict(X_meta_test)
    f1 = f1_score(y_test,preds)
    report = classification_report(y_test,preds,output_dict = True)
    return (f1,report)
  
  except Exception as e:
    raise Exception(f"Error evaluating ensemble {e}")

def save_model(model_path,model):
  try:
    with open(model_path,"wb") as f:
      pickle.dump(model,f)
  except Exception as e:
    raise Exception(f"error occur while saving model: {e}")

def save_metrics(filepath,metrics):
  try:
    os.makedirs(os.path.dirname(filepath),exist_ok = True)
    with open(filepath,"w") as f:
      json.dump(metrics,f,indent = 4)
  except Exception as e:
    raise Exception(f"Error occur while saving json: {e}")

def main():
    try:
        (embedding_dim,hidden_dim,n_layers,dropout,batch_size) = load_params("params.yaml")

        (X_val_tfidf,X_test_tfidf,y_val,y_test) = load_tfidf_data("data/processed/")

        (X_val_seq,X_test_seq,y_val_seq,y_test_seq) = load_bilstm_data("data/processed/")
        val_loader = create_dataloader(X_val_seq,y_val_seq,batch_size)
        test_loader = create_dataloader(X_test_seq,y_test_seq,batch_size)
        rf = load_ml_models("model/random_forest.pkl")
        xgb_model = load_ml_models("model/xgboost.pkl")
        nb = load_ml_models("model/naive_bayes.pkl")

        device = get_device()

        bilstm_model = load_bilstm_model(device,embedding_dim,hidden_dim,n_layers,dropout)

        # Validation probabilities

        (probs_rf_val,probs_xgb_val,probs_nb_val) = get_ml_probabilities(rf,xgb_model,nb,X_val_tfidf)

        probs_bilstm_val = get_bilstm_probabilities(bilstm_model,val_loader,device)

        X_meta_val = create_meta_features(probs_rf_val,probs_xgb_val,probs_nb_val,probs_bilstm_val)

        meta_model = train_meta_model(X_meta_val,y_val)

        # Test probabilities

        (probs_rf_test,probs_xgb_test,probs_nb_test) = get_ml_probabilities(rf,xgb_model,nb,X_test_tfidf)

        probs_bilstm_test = get_bilstm_probabilities(bilstm_model,test_loader,device)

        X_meta_test = create_meta_features(probs_rf_test,probs_xgb_test,probs_nb_test,probs_bilstm_test)

        f1, report = evaluate_ensemble(meta_model,X_meta_test,y_test)

        save_model("model/meta_ensemble.pkl",meta_model)
        save_metrics("reports/metrics/ensemble_metrics.json",
            {"f1_score": float(f1),"classification_report": report})

        print(f"Ensemble F1 Score : {f1:.4f}")

    except Exception as e:
        raise Exception(f"Error in ensemble pipeline : {e}")

if __name__ == "__main__":
    main()