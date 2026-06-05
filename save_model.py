import joblib
from preprocess import load_data, split_data, build_preprocessor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

df = load_data("data/train.csv")

X_train, X_test, y_train, y_test = split_data(df, "page")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

preprocessor = build_preprocessor(X_train)

X_train_p = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_p, y_train_enc)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_res, y_train_res)

joblib.dump(model, "model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model saved successfully")