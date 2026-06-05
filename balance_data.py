import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from preprocess import load_data, build_preprocessor, transform_data


def balance_with_smote(path, target_col):

    df = load_data(path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    preprocessor = build_preprocessor(df, target_col)

    
    X_train_p, X_test_p = transform_data(preprocessor, X_train, X_test)

    print("Before SMOTE:")
    print(y_train.value_counts())

    
    X_train_p = X_train_p.toarray()

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_p, y_train)

    print("\nAfter SMOTE:")
    print(y_train_res.value_counts())

    return X_train_res, X_test_p, y_train_res, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = balance_with_smote(
        "data/train.csv",
        target_col="page"
    )