"""
Lab 16 - Buoc 4.4: Train + Inference benchmark cho LightGBM tren CPU node.
Chay tren Compute Node sau khi da tai dataset ve ~/ml-benchmark/creditcard.csv

Usage:
    python3 benchmark.py
"""

import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = "creditcard.csv"
RESULT_PATH = "benchmark_result.json"

results = {}

# 1. Load dataset va tach train/test
print("Loading dataset...")
t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
load_time = time.perf_counter() - t0
results["data_load_time_sec"] = load_time
results["train_rows"] = len(X_train)
results["test_rows"] = len(X_test)
print(f"Data loaded in {load_time:.3f}s | train={len(X_train)} test={len(X_test)}")

# 2. Huan luyen LGBMClassifier
print("Training LGBMClassifier...")
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)

t0 = time.perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
)
train_time = time.perf_counter() - t0
results["train_time_sec"] = train_time
results["best_iteration"] = int(getattr(model, "best_iteration_", model.n_estimators))
print(f"Training done in {train_time:.3f}s | best_iteration={results['best_iteration']}")

# 3. Danh gia model tren tap test
print("Evaluating model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

results["auc_roc"] = float(roc_auc_score(y_test, y_proba))
results["accuracy"] = float(accuracy_score(y_test, y_pred))
results["f1_score"] = float(f1_score(y_test, y_pred))
results["precision"] = float(precision_score(y_test, y_pred))
results["recall"] = float(recall_score(y_test, y_pred))

print(f"AUC-ROC:   {results['auc_roc']:.4f}")
print(f"Accuracy:  {results['accuracy']:.4f}")
print(f"F1-Score:  {results['f1_score']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall:    {results['recall']:.4f}")

# 4. Inference latency (1 dong)
print("Measuring inference latency (1 row)...")
single_row = X_test.iloc[[0]]
n_latency_runs = 100
latencies = []
for _ in range(n_latency_runs):
    t0 = time.perf_counter()
    model.predict(single_row)
    latencies.append(time.perf_counter() - t0)
avg_latency_ms = (sum(latencies) / n_latency_runs) * 1000
results["inference_latency_ms_per_row"] = avg_latency_ms
print(f"Avg inference latency (1 row): {avg_latency_ms:.3f} ms")

# 5. Inference throughput (1000 dong)
print("Measuring inference throughput (1000 rows)...")
batch = X_test.iloc[: min(1000, len(X_test))]
t0 = time.perf_counter()
model.predict(batch)
batch_time = time.perf_counter() - t0
throughput = len(batch) / batch_time
results["inference_batch_rows"] = len(batch)
results["inference_batch_time_sec"] = batch_time
results["inference_throughput_rows_per_sec"] = throughput
print(f"Throughput: {throughput:.1f} rows/sec ({len(batch)} rows in {batch_time:.4f}s)")

# 6. Ghi ket qua ra file JSON
with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {RESULT_PATH}")
