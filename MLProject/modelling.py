import mlflow
import mlflow.sklearn
import pandas as pd
import os
import sys
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data_path = "diabetes_preprocessing/diabetes.csv"

if not os.path.exists(data_path):
    data_path = "Kriteria 2/diabetes_preprocessing/diabetes.csv"

df = pd.read_csv(data_path)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("Diabetes_Production_Model")

with mlflow.start_run(run_name="final_model_deployment"):
  
    mlflow.sklearn.autolog()

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)


    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example=X_test.iloc[:1]
    )

    print(f"--- Evaluasi Model ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"----------------------")