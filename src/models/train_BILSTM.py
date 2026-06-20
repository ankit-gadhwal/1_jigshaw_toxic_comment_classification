import pandas as pd
import numpy as np
import yaml
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import json
from BILSTM_model import BiLSTM_Attention
def load_params(filepath):
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return (
            params["bilstm"]["embedding_dim"],
            params["bilstm"]["hidden_dim"],
            params["bilstm"]["n_layers"],
            params["bilstm"]["dropout"],
            params["bilstm"]["learning_rate"],
            params["bilstm"]["epochs"],
            params["bilstm"]["batch_size"]
        )
    except Exception as e:
        raise Exception(
            f"Error loading params : {e}"
        )

def load_data(base_path):
    try:
        X_train = np.load(
            base_path + "X_train_seq.npy"
        )
        X_val = np.load(
            base_path + "X_val_seq.npy"
        )
        X_test = np.load(
            base_path + "X_test_seq.npy"
        )
        y_train = np.load(
            base_path + "y_train_seq.npy"
        )

        y_val = np.load(
            base_path + "y_val_seq.npy"
        )
        y_test = np.load(
            base_path + "y_test_seq.npy"
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
        raise Exception(f"Error loading sequence data : {e}")

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx],dtype=torch.float32),
            torch.tensor(self.y[idx],dtype=torch.float32))

def create_datasets(X_train,X_val,X_test,y_train,y_val,y_test):
    try:
        train_dataset = SequenceDataset(
            X_train,
            y_train
        )
        val_dataset = SequenceDataset(
            X_val,
            y_val
        )
        test_dataset = SequenceDataset(X_test,y_test)
        return (
            train_dataset,
            val_dataset,
            test_dataset
        )
    except Exception as e:
        raise Exception(f"Error creating datasets : {e}")

def create_dataloaders(train_dataset,val_dataset,test_dataset,batch_size):
    try:
        train_loader = DataLoader(train_dataset,batch_size=batch_size,
            shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=batch_size)
        test_loader = DataLoader(test_dataset,batch_size=batch_size)
        return (train_loader,val_loader,test_loader)
    except Exception as e:
        raise Exception(f"Error creating dataloaders : {e}")

def get_device():
    try:
        return torch.device("cuda"if torch.cuda.is_available()
            else "cpu")
    except Exception as e:
        raise Exception(f"Error getting device : {e}")

def create_model(device,embedding_dim,hidden_dim,n_layers,dropout):
    try:
        model = BiLSTM_Attention(embedding_dim,hidden_dim,n_layers,dropout).to(device)
        return model
    except Exception as e:
        raise Exception(f"Error creating model : {e}")
  
def create_optimizer(model,lr):
    try:
        return torch.optim.Adam(model.parameters(),lr=lr)
    except Exception as e:
        raise Exception(f"Error creating optimizer : {e}")

def create_criterion():
    try:
        return nn.BCELoss()
    except Exception as e:
        raise Exception(f"Error creating criterion : {e}")

def train_one_epoch(model,train_loader,optimizer,criterion,device):
    try:
        model.train()
        for xs, ys in train_loader:
            xs = xs.to(device)
            ys = ys.to(device)
            optimizer.zero_grad()
            outputs = model(xs)
            loss = criterion(outputs,ys)
            loss.backward()
            optimizer.step()
    except Exception as e:
        raise Exception(f"Error during training : {e}")

def validate_model(model,val_loader,device):
    try:
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for xs, ys in val_loader:
                xs = xs.to(device)
                outputs = model(xs)
                preds = (outputs > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(ys.numpy())
        return f1_score(val_true,val_preds)
    except Exception as e:
        raise Exception(f"Error validating model : {e}")

def train_model(model,train_loader,
        val_loader,optimizer,criterion,
        device,epochs):
    try:
        best_f1 = 0
        for epoch in range(epochs):
            train_one_epoch(model,train_loader,optimizer,criterion,device)
            f1 = validate_model(model,val_loader,device)
            print(f"Epoch {epoch+1}/{epochs} "f"Validation F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
        return model, best_f1
    except Exception as e:
        raise Exception(f"Error during model training : {e}")

def save_model(model,filepath):
    try:
        torch.save(model.state_dict(),filepath)
    except Exception as e:
        raise Exception(f"Error saving model : {e}")

def save_metrics(metrics,filepath):
    try:
        with open(filepath,"w") as f:
            json.dump(metrics,f,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics : {e}")

def main():
    try:
        (embedding_dim,hidden_dim,n_layers,dropout,
            lr,epochs,batch_size
        ) = load_params("params.yaml")

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test

        ) = load_data("data/processed/")

        (
            train_dataset,
            val_dataset,
            test_dataset

        ) = create_datasets(X_train,X_val,X_test,
        y_train,y_val,y_test)

        (
            train_loader,val_loader,test_loader
        ) = create_dataloaders(train_dataset,val_dataset,test_dataset,batch_size)
        
        device = get_device()

        model = create_model(device,embedding_dim,hidden_dim,n_layers,
            dropout)
        optimizer = create_optimizer(model,lr)

        criterion = create_criterion()

        model, best_f1 = train_model(
            model,train_loader,
            val_loader,optimizer,
            criterion,device,epochs)

        save_model(model,"model/bilstm_model.pth")
        save_metrics({"validation_f1":float(best_f1)},
            "reports/metrics/bilstm_metrics.json")
    except Exception as e:
        raise Exception(f"Error in BiLSTM pipeline : {e}")

if __name__ == "__main__":

    main()