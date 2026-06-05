import pandas as pd
from preprocess import load_data


def create_time_features(df):

    df["is_weekend"] = df["day"].apply(lambda x: 1 if x % 7 in [0, 6] else 0)
    return df


def create_session_features(df):
    
    session_stats = df.groupby("session_id").agg(
        clicks=("order", "count"),
        total_price=("price", "sum"),
        avg_price=("price", "mean")
    ).reset_index()

    df = df.merge(session_stats, on="session_id", how="left")
    return df


def run_feature_engineering(path):
    df = load_data(path)

    df = create_time_features(df)
    df = create_session_features(df)

    return df


if __name__ == "__main__":
    df = run_feature_engineering("data/train.csv")
    print(df.head())
    print("Shape after FE:", df.shape)