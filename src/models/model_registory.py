import mlflow
import dagshub
import numpy as np
import pandas as pd
import os
import json
from mlflow.tracking import MlflowClient
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name = "1_jigshaw_toxic_comment_classification"

os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

mlflow.set_experiment("Final_model")
reports_path = "reports/run_info.json"
with open(reports_path,'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id']
model_name = run_info['model_name']

client = MlflowClient()

model_uri = f"runs:/{run_id}/model"
run = client.get_run(run_id)
print(run.data.tags)
print(run.info.run_id)
print(run.info.artifact_uri)
arts = client.list_artifacts(run_id)
print("Number of artifacts:",len(arts))
for a in arts:
    print(a.path)
    # register the model
reg = mlflow.register_model(model_uri,model_name)
model_version = reg.version 
new_stage = "Staging"

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions = True
)

print(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")