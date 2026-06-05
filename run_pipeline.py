from preprocess import load_data, split_data, build_preprocessor, transform_data


df = load_data("data/train.csv")


X_train, X_test, y_train, y_test = split_data(df, "page")


preprocessor = build_preprocessor(df, "page")


X_train_p, X_test_p = transform_data(preprocessor, X_train, X_test)


print("Original shape:", X_train.shape)
print("Processed shape:", X_train_p.shape)