import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
def preprocess_data(df: pd.DataFrame,target_col:str = 'Churn') ->pd.DataFrame:
    df.columns = df.columns.str.strip()
    if 'customerID' in df.columns:
        df=df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)
    df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)
    return df



# def vif(df:pd.DataFrame)->pd.DataFrame:
#     corr_matrix = df.corr(numeric_only=True)
#     churn_corr = corr_matrix[['Churn']].sort_values(by='Churn',ascending=False)
#     ## collapsing redundant features before running VIF
#     df['No_internet_service'] = (
#     df['StreamingTV_No internet service'] |
#     df['OnlineSecurity_No internet service'] |
#     df['OnlineBackup_No internet service'] |
#     df['DeviceProtection_No internet service'] |
#     df['StreamingMovies_No internet service'] |
#     df['TechSupport_No internet service']
#     ).astype(int)
#     drop_cols = [col for col in df.columns if 'No internet service' in col]
#     df = df.drop(columns=drop_cols)
#     if 'MultipleLines_No phone service' in df.columns:
#         df['No_phone_service']=df['MultipleLines_No phone service'].astype(int)
#         df = df.drop(columns=['MultipleLines_No phone service'])

#     X = df.drop(columns=['Churn'])
#     bool_cols = X.select_dtypes(include='bool').columns
#     X[bool_cols] = X[bool_cols].astype(int)

#     X= X.replace([np.inf,-np.inf],np.nan)
#     X=X.dropna()
#     vif_data = pd.DataFrame()
#     vif_data['feature'] = X.columns
#     vif_data['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
#     vif_data = vif_data.sort_values(by='VIF',ascending=False)