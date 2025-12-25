import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import os

data_path = "Kriteria 2/diabetes_preprocessing/diabetes.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.sklearn.autolog()

clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Simpan model ke folder lokal
mlflow.sklearn.save_model(
    sk_model=clf,
    path="mlflow_model_local",
    input_example=X_test.iloc[:1]
)

print(f"Accuracy : {acc:.4f}")
print(f"Recall   : {recall:.4f}")
