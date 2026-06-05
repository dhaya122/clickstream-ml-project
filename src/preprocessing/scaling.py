from sklearn.preprocessing import StandardScaler

def scale_features(train_df, test_df):
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    return train_scaled, test_scaled, scaler