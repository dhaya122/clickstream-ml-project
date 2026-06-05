import pandas as pd

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

train_df, test_df = load_data(
    r"C:\Users\dhaya\Documents\clickstream_project\data\train.csv",
    r"C:\Users\dhaya\Documents\clickstream_project\data\test.csv"
)

print(train_df.head())
print(test_df.head())