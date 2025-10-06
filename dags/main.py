"""
Modified Airflow Lab 2 - Wine Quality Classification
Using Random Forest instead of Logistic Regression
"""
from __future__ import annotations
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    evaluate_model,
    load_model,
)

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# Main DAG
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="Wine Quality Classification using Random Forest",
    schedule="@daily",
    catchup=False,
    tags=["wine_quality", "random_forest", "classification"],
    owner_links={"Student": "https://github.com/YashKhare20/MLOps-Lab2.git"},
    max_active_runs=1,
)

# Task 1: Owner identification
owner_task = BashOperator(
    task_id="identify_owner",
    bash_command='echo "Pipeline Owner: Yash Khare"',
    owner="Student",
    dag=dag,
)

# Task 2: Load wine quality data
load_data_task = PythonOperator(
    task_id="load_wine_data",
    python_callable=load_data,
    dag=dag,
)

# Task 3: Preprocess data (scaling, encoding)
data_preprocessing_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 4: Separate data outputs
separate_data_outputs_task = PythonOperator(
    task_id="separate_train_test",
    python_callable=separate_data_outputs,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# Task 5: Build and save Random Forest model
build_save_model_task = PythonOperator(
    task_id="train_random_forest",
    python_callable=build_model,
    op_args=[separate_data_outputs_task.output, "wine_rf_model.pkl"],
    dag=dag,
)

# Task 6: Evaluate model performance
evaluate_model_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    op_args=[separate_data_outputs_task.output, "wine_rf_model.pkl"],
    dag=dag,
)

# Task 7: Load and test model
load_model_task = PythonOperator(
    task_id="load_and_test_model",
    python_callable=load_model,
    op_args=[separate_data_outputs_task.output, "wine_rf_model.pkl"],
    dag=dag,
)

# Task 8: Send email notification
send_email = EmailOperator(
    task_id="send_success_email",
    to="yashkharess20@gmail.com",
    subject="Wine Quality Model Training Complete",
    html_content="<p>The Random Forest model has been successfully trained and evaluated.</p>",
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Task 9: Trigger Flask API DAG
trigger_dag_task = TriggerDagRunOperator(
    task_id="trigger_flask_dashboard",
    trigger_dag_id="Airflow_Lab2_Flask",
    conf={"message": "Model training completed"},
    reset_dag_run=False,
    wait_for_completion=False,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# Define task dependencies
(
    owner_task
    >> load_data_task
    >> data_preprocessing_task
    >> separate_data_outputs_task
    >> build_save_model_task
    >> evaluate_model_task
    >> load_model_task
    >> [send_email, trigger_dag_task]
)