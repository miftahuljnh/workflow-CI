import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

current_dir = os.path.dirname(os.path.abspath(__file__))

possible_path_1 = os.path.join(current_dir, "diabetes_preprocessing")
possible_path_2 = os.path.join(current_dir, "MLProject", "diabetes_preprocessing")

if os.path.exists(possible_path_1):
    BASE_PATH = possible_path_1
elif os.path.exists(possible_path_2):
    BASE_PATH = possible_path_2
else:
    # Fallback jika dijalankan dari root folder project
    BASE_PATH = "MLProject/diabetes_preprocessing"

print(f"INFO: Menggunakan base path: {BASE_PATH}")

# Definisi Path File
X_train_path = os.path.join(BASE_PATH, "X_train_scaled.csv")
X_test_path = os.path.join(BASE_PATH, "X_test_scaled.csv")
y_train_path = os.path.join(BASE_PATH, "y_train.csv")
y_test_path = os.path.join(BASE_PATH, "y_test.csv")

try:
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()
except Exception as e:
    print(f"ERROR: Gagal membaca file CSV. Pastikan file ada di {BASE_PATH}")
    raise e

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

    # Log Metrics secara manual (selain dari autolog)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("recall", recall)

    # Simpan Model
    # Gunakan folder yang berbeda untuk setiap run agar tidak konflik
    model_save_path = "mlflow_model_local"
    if os.path.exists(model_save_path):
        import shutil
        shutil.rmtree(model_save_path) # Hapus folder lama jika sudah ada

    mlflow.sklearn.save_model(
        sk_model=model,
        path=model_save_path,
        input_example=X_test.iloc[:1]
    )

    print("-" * 30)
    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("-" * 30)
 
