import pandas as pd

def encode_features(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df