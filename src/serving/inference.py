import pandas as pd
import os
import json

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # src/serving
        "..",                       # src
        ".."                        # project root
    )
)

FEATURE_COLUMNS_PATH = os.path.join(
    PROJECT_ROOT,
    "artificats",
    "feature_columns.json"
)
with open(FEATURE_COLUMNS_PATH) as f:
    feature_columns = json.load(f)



def preprocess_input(df):

    df = df.copy()


    # same as training preprocess
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])


    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


    df = df.fillna(0)


    # binary encoding
    df = df.replace({
        "Yes":1,
        "No":0,
        "Male":1,
        "Female":0
    })


    # one hot encode remaining categorical columns
    df = pd.get_dummies(
        df,
        drop_first=True
    )


    # VERY IMPORTANT
    # match training columns

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )


    return df