import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import mlflow.lightgbm
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import(
    accuracy_score,recall_score,classification_report,
    f1_score,roc_auc_score,precision_score
)
from lightgbm import LGBMClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data
from src.models.tune import tune_model

def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    mlruns_path = args.mlflow_uri or f"sqlite:///{project_root}/mlflow.db"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model","lightgbm")
        mlflow.log_param("threshold",args.threshold)
        mlflow.log_param("test_size",args.test_size)

        ## Stage 1-- loading data
        print("loading data")
        df = load_data(args.input)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        ### data validation step
        # print(" Validating data quality with Great Expectations...")
        # is_valid, failed = validate_telco_data(df)
        # mlflow.log_metric("data_quality_pass", int(is_valid)) 

        # if not is_valid:
        #     import json
        #     mlflow.log_text(json.dumps(failed,indent=2),artifact_file="failed_expectations.json")
        #     raise ValueError(f" Data quality check failed. Issues: {failed}")
        # else:
        #     print("Data validation passed. Logged to MLflow.")

        ## stage 2 -- preprocess data
        print("Preprocessing data")
        df = preprocess_data(df)

        processed_path = os.path.join(project_root,"data","processed","telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path),exist_ok=True)
        df.to_csv(processed_path,index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        ## stage 3-- building features
        print("Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column {target} not found in data")
        
        df_encoded = build_features(df,target_col=target)
        for c in df_encoded.select_dtypes(include=["bool"]).columns:
            df_encoded[c] = df_encoded[c].astype(int)
        print(f"Feature engineering completed: {df_encoded.shape[1]} features")

        import json,joblib
        artifacts_dir = os.path.join(project_root,"artificats")
        os.makedirs(artifacts_dir,exist_ok=True)

        feature_cols = list(df_encoded.drop(columns=[target]).columns)

        with open(os.path.join(artifacts_dir,"feature_columns.json"),"w") as f:
            json.dump(feature_cols,f)
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        ## Stage 4: train-test split
        print("Splitting data")
        X= df_encoded.drop(columns=[target])
        y = df_encoded[target]
        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=args.test_size,stratify=y,random_state=42
        )
        print(f" Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
        print()
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        print("Training Lightgbm model")
        best_params = tune_model(X=X_train,y=y_train,)
        model = LGBMClassifier(**best_params,scale_pos_weight=scale_pos_weight,random_state=42)

        ## training model
        start_time = time.time()
        model.fit(X_train,y_train)
        train_time = time.time() - start_time
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"Model trained in {train_time:.2f} seconds")

        ##evaluating model
        print("Evaluating model")
        start_time = time.time()
        proba = model.predict_proba(X_test)[:,1]
        y_pred = (proba>=args.threshold).astype(int)
        pred_time = time.time()-start_time
        mlflow.log_metric("pred_time", pred_time)

        ### log evaluation metrics to MLflow

        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_pred)

        mlflow.log_metric("precision",precision)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("f1",f1)
        mlflow.log_metric("roc_auc",roc_auc)

        print(f" Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        ### stage 7: model serialization and logging
        mlflow.lightgbm.log_model(model,name='model')
        print("Model saved to MLflow for serving pipleline")
        # === Final Performance Summary ===
        print(f"\n  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))
        print(df.dtypes)
        print(df.columns.tolist())

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with Lightgbm + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)
# python scripts/run_pipeline.py \                                            
#     --input data/raw/Telco-Customer-Churn.csv \
#     --target Churn

#mlflow ui --backend-store-uri sqlite:///mlflow.db