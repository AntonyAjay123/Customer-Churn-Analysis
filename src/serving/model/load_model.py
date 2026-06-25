import mlflow
import mlflow.lightgbm
import joblib


mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


# model = mlflow.lightgbm.load_model(
#     "runs:/35cb3ff7763649dcbe4d135b26c34816/model"
# )
model = joblib.load("artificats/model.pkl")