from preprocess import load_data, split_data, build_preprocessor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
import pandas as pd


df = load_data("data/train.csv")

X_train, X_test, y_train, y_test = split_data(df, "page")

print("Data split done")


le = LabelEncoder()

y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)


preprocessor = build_preprocessor(X_train)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)


smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train_p, y_train_enc)

print("\nAfter SMOTE:")
print(pd.Series(y_train_res).value_counts())


def evaluate_cv(model, X, y, name):
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    print(f"\n{name} CV Scores:", scores)
    print(f"{name} Mean CV Accuracy:", scores.mean())


lr = LogisticRegression(max_iter=300, solver="lbfgs")
lr.fit(X_train_res, y_train_res)

lr_pred = lr.predict(X_test_p)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test_enc, lr_pred))
print(confusion_matrix(y_test_enc, lr_pred))
print(classification_report(y_test_enc, lr_pred))

evaluate_cv(lr, X_train_res, y_train_res, "Logistic Regression")


rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_res, y_train_res)

rf_pred = rf.predict(X_test_p)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test_enc, rf_pred))
print(confusion_matrix(y_test_enc, rf_pred))
print(classification_report(y_test_enc, rf_pred))

evaluate_cv(rf, X_train_res, y_train_res, "Random Forest")


print("\n=== Feature Importance (Top 10) ===")

fi = pd.DataFrame({
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print(fi.head(10))


xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train_res, y_train_res)

xgb_pred = xgb.predict(X_test_p)

print("\n=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test_enc, xgb_pred))
print(confusion_matrix(y_test_enc, xgb_pred))
print(classification_report(y_test_enc, xgb_pred))

evaluate_cv(xgb, X_train_res, y_train_res, "XGBoost")


results = {
    "Logistic Regression": accuracy_score(y_test_enc, lr_pred),
    "Random Forest": accuracy_score(y_test_enc, rf_pred),
    "XGBoost": accuracy_score(y_test_enc, xgb_pred)
}

print("\n=== MODEL COMPARISON ===")
for k, v in results.items():
    print(k, ":", round(v, 4))