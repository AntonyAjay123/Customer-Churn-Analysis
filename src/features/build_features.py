import pandas as pd
def get_binary_columns(df:pd.DataFrame,columns:list):
    res=[]
    for col in columns:
        unique_vals = df[col].unique()
        if len(unique_vals)==2:
            res.append(col)
        else:
            continue
    return res

def encode_binary_features(s: pd.Series)->pd.DataFrame:
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # === DETERMINISTIC BINARY MAPPINGS ===
    # CRITICAL: These exact mappings are hardcoded in serving pipeline
    
    # Yes/No mapping (most common pattern in telecom data)
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
        
    # Gender mapping (demographic feature)
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # === GENERIC BINARY MAPPING ===
    # For any other 2-category feature, use stable alphabetical ordering
    if len(vals) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding
    return s

def build_features(df:pd.DataFrame,target_col:str='Churn')->pd.DataFrame:
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")
    obj_cols = [c for c in df.select_dtypes(include=['object']).columns]
    # print("object cols are",obj_cols)
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    print(f" found {len(obj_cols)} categorical and {len(numeric_cols)} numeric cols")
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    print("multi cols are",multi_cols)
    for c in binary_cols:
        # original_dtype = df[c].dtype
        df[c] = encode_binary_features(df[c].astype(str))
    print(f"Encoded {len(binary_cols)}binary features to 0/1")
    df = pd.get_dummies(df,columns=multi_cols,drop_first=True)
    print(f"One hot encoded {len(multi_cols)}multicategorical features")
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"Converted {len(bool_cols)}boolean columns to int")
    #convert nullable integers to standard integers
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)
    print(f"Feature enginnering complete: {df.shape[1]} final features")
    print("feature engineered post",df.columns)
    return df