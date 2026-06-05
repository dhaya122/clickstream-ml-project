from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from preprocess import load_data, split_data, build_preprocessor



df = load_data("data/train.csv")

X_train, X_test, y_train, y_test = split_data(df, "page")

print("Data split done")



preprocessor = build_preprocessor(X_train)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)



smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train_p, y_train)

print("After SMOTE:", y_train_res.value_counts())



lr = LogisticRegression(max_iter=300, solver="lbfgs")
lr.fit(X_train_res, y_train_res)

lr_pred = lr.predict(X_test_p)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))



rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_res, y_train_res)

rf_pred = rf.predict(X_test_p)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))