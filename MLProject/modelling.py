import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

BASE_PATH = "MLProject/diabetes_preprocessing"

X_train_path = f"{BASE_PATH}/X_train_scaled.csv"
X_test_path = f"{BASE_PATH}/X_test_scaled.csv"
y_train_path = f"{BASE_PATH}/y_train.csv"
y_test_path = f"{BASE_PATH}/y_test.csv"

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).squeeze()  # jadi Series
y_test = pd.read_csv(y_test_path).squeeze()

mlflow.sklearn.autolog()

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.save_model(
        sk_model=model,
        path="mlflow_model_local",
        input_example=X_test.iloc[:1]
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
