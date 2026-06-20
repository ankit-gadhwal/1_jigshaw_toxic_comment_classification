import json
import pickle
import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import os
import yaml
def load_data(base_path):
    try:
        X_train = load_npz(
            base_path + "X_train_tfidf.npz"
        )
        X_val = load_npz(
            base_path + "X_val_tfidf.npz"
        )
        y_train = np.load(
            base_path + "y_train.npy"
        )
        y_val = np.load(
            base_path + "y_val.npy"
        )
        return (
            X_train,
            X_val,
            y_train,
            y_val
        )
    except Exception as e:
        raise Exception(
            f"Error loading data : {e}"
        )

def load_params(filepath):
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return params["train_rfc_xgb_nb"]["rfc_n_estimators"]
    except Exception as e:
      raise Exception(f"error occur while parameter loading {e}")


def train_random_forest(
        X_train,y_train,n_estimator):
    try:
        rf = RandomForestClassifier(
            random_state=42,
            n_estimators=n_estimator
        )
        rf.fit(
            X_train,
            y_train
        )
        return rf

    except Exception as e:
      raise Exception(
            f"Error training Random Forest : {e}"
        )

def train_xgboost(
        X_train,y_train):
    try:
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(
            X_train,
            y_train
        )
        return model

    except Exception as e:

        raise Exception(
            f"Error training XGBoost : {e}"
        )

def train_naive_bayes(
        X_train,
        y_train
):
    try:
        model = MultinomialNB()
        model.fit(
            X_train,
            y_train
        )
        return model
    except Exception as e:
        raise Exception(
            f"Error training Naive Bayes : {e}"
        )

def evaluate_model(
        model,
        X_val,
        y_val
):

    try:

        y_pred = model.predict(
            X_val
        )

        f1 = f1_score(
            y_val,
            y_pred
        )

        report = classification_report(
            y_val,
            y_pred,
            output_dict=True
        )

        return f1, report

    except Exception as e:

        raise Exception(
            f"Error evaluating model : {e}"
        )


def save_model(
        model,
        filepath
):

    try:
        with open(filepath,"wb") as f:
            pickle.dump(model,f)
    except Exception as e:
        raise Exception(f"Error saving model : {e}")


def save_metrics(
        metrics,
        filepath
):

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath,"w") as f:
          json.dump(metrics,f,indent=4)
    except Exception as e:
      raise Exception(f"Error saving metrics : {e}")
def main():
  try:
    base_path = ("data/processed/")
    (X_train,
      X_val,
      y_train,
      y_val
      ) = load_data(base_path)
    params_path = "params.yaml"
    n_estimator = load_params(params_path)
    model_path = ("model/")
    metric_path = ("reports/metrics/")
        # Random Forest
    rf_model = train_random_forest(X_train,y_train,n_estimator)
    rf_f1, rf_report = evaluate_model( rf_model,X_val,y_val)
    save_model(rf_model,
          model_path +
          "random_forest.pkl"
        )
    save_metrics({"f1_score": rf_f1,"classification_report":rf_report},
          metric_path +
          "rf_metrics.json"
        )

        # XGBoost
    xgb_model = train_xgboost(
          X_train,
          y_train
        )
    xgb_f1, xgb_report = evaluate_model(
        xgb_model,
        X_val,
        y_val
    )
    save_model(
         xgb_model,
         model_path +
         "xgboost.pkl"
        )

    save_metrics({
            "f1_score": xgb_f1,
            "classification_report":
            xgb_report
        },
        metric_path +
        "xgb_metrics.json"
        )
        # Naive Bayes
    nb_model = train_naive_bayes(
        X_train,
        y_train
        )
    nb_f1, nb_report = evaluate_model(
        nb_model,X_val,y_val)
    save_model(
        nb_model,
        model_path +
        "naive_bayes.pkl"
        )
    save_metrics({
            "f1_score": nb_f1,
            "classification_report":nb_report},metric_path + "nb_metrics.json")
    print("Training completed successfully.")
  except Exception as e:
    raise Exception(f"Error in model training pipeline : {e}")
if __name__ == "__main__":
    main()
