# File: Flask_API.py
from __future__ import annotations

import os
import json
import time
import pendulum
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from flask import Flask, redirect, render_template, jsonify

# Configuration
WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://airflow-webserver:8080")
AF_USER = os.getenv("AIRFLOW_USERNAME", os.getenv("_AIRFLOW_WWW_USER_USERNAME", "airflow"))
AF_PASS = os.getenv("AIRFLOW_PASSWORD", os.getenv("_AIRFLOW_WWW_USER_PASSWORD", "airflow"))
TARGET_DAG_ID = os.getenv("TARGET_DAG_ID", "Airflow_Lab2")
MODEL_DIR = "/opt/airflow/model"

# Default args
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# Flask app
app = Flask(__name__, template_folder="templates")


def get_latest_run_info():
    """
    Query Airflow stable REST API (/api/v2) using Basic Auth and load metrics.
    """
    url = f"{WEBSERVER}/api/v2/dags/{TARGET_DAG_ID}/dagRuns?order_by=-logical_date&limit=1"
    
    try:
        r = requests.get(url, auth=(AF_USER, AF_PASS), timeout=5)
    except Exception as e:
        return False, {"note": f"Exception calling Airflow API: {e}", "metrics": {}}

    if r.status_code != 200:
        snippet = r.text[:200].replace("\n", " ")
        return False, {"note": f"API status {r.status_code}: {snippet}", "metrics": {}}

    runs = r.json().get("dag_runs", [])
    if not runs:
        return False, {"note": "No DagRuns found yet.", "metrics": {}}

    run = runs[0]
    state = run.get("state")
    
    # Load model metrics if available
    metrics = {}
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
    
    info = {
        "state": state,
        "run_id": run.get("dag_run_id"),
        "logical_date": run.get("logical_date"),
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
        "note": "",
        "metrics": metrics,
    }
    
    return state == "success", info


@app.route("/")
def index():
    ok, _ = get_latest_run_info()
    return redirect("/success" if ok else "/failure")


@app.route("/success")
def success():
    ok, info = get_latest_run_info()
    return render_template("success.html", **info)


@app.route("/failure")
def failure():
    ok, info = get_latest_run_info()
    return render_template("failure.html", **info)


@app.route("/metrics")
def metrics():
    """Endpoint to view model metrics as JSON"""
    _, info = get_latest_run_info()
    return jsonify(info.get("metrics", {}))


@app.route("/health")
def health():
    return "ok", 200


def start_flask_app():
    """
    Run Flask dev server in-process; task intentionally blocks to keep API alive.
    Disable reloader to avoid forking inside Airflow worker.
    """
    print("Starting Flask on 0.0.0.0:5555 ...", flush=True)
    app.run(host="0.0.0.0", port=5555, use_reloader=False)
    # If app.run ever returns, keep the task alive:
    while True:
        time.sleep(60)


# DAG
flask_api_dag = DAG(
    dag_id="Airflow_Lab2_Flask",
    default_args=default_args,
    description="DAG to manage Flask API lifecycle",
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["Flask_Api"],
    max_active_runs=1,
)

start_flask_API = PythonOperator(
    task_id="start_Flask_API",
    python_callable=start_flask_app,
    dag=flask_api_dag,
)

start_flask_API

if __name__ == "__main__":
    start_flask_API.cli()