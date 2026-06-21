import unittest
import mlflow
from mlflow.tracking import MlflowClient
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score
from src.models.model_evalvation import (load_params,load_vectorizer,load_glove_embeddings,
    create_tfidf_features,create_sequence_features,create_dataloader,
    get_ml_probabilities,get_bilstm_probabilities,create_meta_features,)

dagshub_token = os.getenv("JTC_CLASSIFICATION")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name = "1_jigshaw_toxic_comment_classification"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

model_name = "meta_ensemble"

class TestModelLoading(unittest.TestCase):
    """unit test class to verify MLflow model loading from the Staging stage"""

    def test_model_in_staging(self):
        """Test if the model exists in the staging stage"""

        # Initialize the mlflow client to interact with mlflow server
        client = MlflowClient()

       #  Retrive the latest versions of the models in the 'Staging' stage
        versions = client.get_latest_versions(model_name,stages=["Staging"])

       # Assert that at least one version of the model exists in the 'Staging' stage.
       #  If no versions are found,it will raise an error
        self.assertGreater(len(versions),0,"No model found in the 'Staging' stage")

    def test_model_loading(self):
        """Test if the model can be loaded properly from the Staging stage."""
        
        # Retrive the latest versions of the models in the 'Staging' stage
        versions = client.get_latest_versions(model_name,stages=["Staging"])

        # Initialize the Mlflow client again to interact with the server
        client = MlflowClient()

        # If no versions are found,fails the test and skip the model loading part
        if not versions:
            self.fail("No model found in the 'staging' stage,skipping model loading test.")

            # get the version details of the latest model in the 'Staging' stage
        latest_version = versions[0].version
        run_id = versions[0].run_id
            
        try:
                # try to load the model from the specified path
            rf = mlflow.sklearn.load_model(f"runs:/{run_id}/random_forest")

            xgb_model = mlflow.sklearn.load_model(f"runs:/{run_id}/xgboost")

            nb = mlflow.sklearn.load_model(f"runs:/{run_id}/naive_bayes")

            bilstm_model = mlflow.pytorch.load_model(f"runs:/{run_id}/bilstm")
            ensemble_model = mlflow.sklearn.load_model(f"runs:/{run_id}/meta_ensemble")
        except Exception as e:
            # If loading the modals fails,fail the test and output the error message
            self.fail(f"Failed to load the model:{e}")
            
        self.assertIsNotNone(rf)
        self.assertIsNotNone(xgb_model)
        self.assertIsNotNone(nb)
        self.assertIsNotNone(bilstm_model)
        self.assertIsNotNone(ensemble_model)

    def test_model_performance(self):
        """Test the performance of the model on test data."""
        client = MlflowClient()    
        
        versions = client.get_latest_versions(model_name,stages=["Staging"])

        if not versions:
            self.fail("No model found in Staging")

        run_id = versions[0].run_id
        rf = mlflow.sklearn.load_model(f"runs:/{run_id}/random_forest")
        
        xgb_model = mlflow.sklearn.load_model(f"runs:/{run_id}/xgboost")

        nb = mlflow.sklearn.load_model(f"runs:/{run_id}/naive_bayes")

        bilstm_model = mlflow.pytorch.load_model(f"runs:/{run_id}/bilstm")

        ensemble_model = mlflow.sklearn.load_model(f"runs:/{run_id}/meta_ensemble")

        (embedding_dim,hidden_dim,n_layers,dropout,batch_size,max_len,dim) = load_params("params.yaml")

        test_df = pd.read_csv("data/processed/merged_test.csv")

        vectorizer = load_vectorizer("data/processed/vectorizer.pkl")

        glove_embeddings = load_glove_embeddings("data/processed/glove_embeddings.pkl")

        X_test_tfidf = create_tfidf_features(vectorizer,test_df)

        (probs_rf,probs_xgb,probs_nb) = get_ml_probabilities(rf,xgb_model,nb,X_test_tfidf)

        texts = test_df["merged_text"].values

        X_test_seq = create_sequence_features(texts,glove_embeddings,max_len,dim)

        test_loader = create_dataloader(X_test_seq,batch_size)

        device = torch.device("cuda"if torch.cuda.is_available()else "cpu")

        probs_bilstm = get_bilstm_probabilities(bilstm_model,test_loader,device)

        X_meta_test = create_meta_features(probs_rf,probs_xgb,probs_nb,probs_bilstm)

        y_probs = ensemble_model.predict_proba(X_meta_test)[:, 1]

        y_pred = (y_probs >= 0.5).astype(int)

        y_true = test_df["rule_violation"].values

        f1 = f1_score(y_true,y_pred)
        

        print(f"F1 Score = {f1:.4f}")

        self.assertGreater(f1,0.75,f"F1 score too low ({f1:.4f})")
        
if __name__ == "__main__":
    unittest.main()