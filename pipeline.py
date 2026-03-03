"""
MLOps Churn Prediction Pipeline — Vertex AI
End-to-end Kubeflow pipeline: Extract → Engineer → Train → Evaluate → Register → Deploy
Conditional deployment gate: only deploys if AUC >= threshold
"""

from kfp import dsl, compiler
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform


# ═══════════════════════════════════════════════════════════════════
# Component 1: Extract Data from BigQuery
# ═══════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-bigquery", "pandas", "db-dtypes", "pyarrow"]
)
def extract_data(
    project_id: str,
    output_dataset: Output[Dataset]
):
    from google.cloud import bigquery
    import pandas as pd
    client = bigquery.Client(project=project_id)
    query = (
        "SELECT c.customer_id AS customer_id, c.region AS region, "
        "DATE_DIFF(DATE '2024-12-31', MAX(f.order_date), DAY) AS days_since_last_purchase, "
        "COUNT(DISTINCT f.order_id) AS purchase_frequency, "
        "ROUND(SUM(f.total_amount), 2) AS total_spend, "
        "ROUND(AVG(f.total_amount), 2) AS avg_order_value, "
        "COUNT(DISTINCT f.product_id) AS unique_products_bought, "
        "COUNT(DISTINCT f.region) AS regions_purchased_from, "
        "ROUND(COALESCE(STDDEV(f.total_amount), 0), 2) AS spend_variability, "
        "DATE_DIFF(DATE '2024-12-31', c.signup_date, DAY) AS customer_tenure_days, "
        "COUNT(DISTINCT f.category) AS unique_categories, "
        "CASE WHEN DATE_DIFF(DATE '2024-12-31', MAX(f.order_date), DAY) > 90 THEN 1 ELSE 0 END AS churned "
        f"FROM `{project_id}.retail_gold.dim_customer` c "
        f"JOIN `{project_id}.retail_gold.fct_daily_sales` f ON c.customer_id = f.customer_id "
        "GROUP BY c.customer_id, c.region, c.signup_date"
    )
    df = client.query(query).to_dataframe()
    df.to_csv(output_dataset.path, index=False)
    print(f"Extracted {len(df)} rows")


# ═══════════════════════════════════════════════════════════════════
# Component 2: Feature Engineering + Train/Test Split
# ═══════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def feature_engineering(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import json

    df = pd.read_csv(input_dataset.path)
    feature_cols = [
        'purchase_frequency', 'total_spend', 'avg_order_value',
        'unique_products_bought', 'regions_purchased_from',
        'spend_variability', 'customer_tenure_days', 'unique_categories'
    ]
    X = df[feature_cols]
    y = df['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv(train_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")


# ═══════════════════════════════════════════════════════════════════
# Component 3: Train XGBoost Model
# ═══════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "xgboost", "scikit-learn"]
)
def train_model(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model]
):
    import pandas as pd
    import xgboost as xgb

    train_df = pd.read_csv(train_dataset.path)
    feature_cols = [c for c in train_df.columns if c != 'churned']
    X_train = train_df[feature_cols]
    y_train = train_df['churned']

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        objective='binary:logistic', eval_metric='auc', random_state=42
    )
    model.fit(X_train, y_train, verbose=False)

    model_path = model_artifact.path + ".bst"
    model.save_model(model_path)
    model_artifact.metadata['framework'] = 'xgboost'
    model_artifact.metadata['model_path'] = model_path
    print(f"Model trained and saved")


# ═══════════════════════════════════════════════════════════════════
# Component 4: Evaluate Model + Deployment Gate
# ═══════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "xgboost", "scikit-learn"]
)
def evaluate_model(
    test_dataset: Input[Dataset],
    model_artifact: Input[Model],
    metrics: Output[Metrics],
    auc_threshold: float = 0.75
) -> bool:
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score
    )

    test_df = pd.read_csv(test_dataset.path)
    feature_cols = [c for c in test_df.columns if c != 'churned']
    X_test = test_df[feature_cols]
    y_test = test_df['churned']

    model = xgb.XGBClassifier()
    model.load_model(model_artifact.metadata['model_path'])

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    metrics.log_metric("accuracy", accuracy_score(y_test, y_pred))
    metrics.log_metric("auc_roc", auc)
    metrics.log_metric("precision", precision_score(y_test, y_pred))
    metrics.log_metric("recall", recall_score(y_test, y_pred))
    metrics.log_metric("f1_score", f1_score(y_test, y_pred))

    passed = auc >= auc_threshold
    metrics.log_metric("deployment_gate", "PASS" if passed else "FAIL")
    print(f"AUC: {auc:.4f} | {'PASS' if passed else 'FAIL'}")
    return passed


# ═══════════════════════════════════════════════════════════════════
# Component 5: Register Model in Vertex AI Model Registry
# ═══════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"]
)
def register_model(
    project_id: str,
    model_artifact: Input[Model],
    model_name: str = "churn-prediction-pipeline"
) -> str:
    from google.cloud import aiplatform, storage

    aiplatform.init(project=project_id, location='us-central1')

    storage_client = storage.Client()
    bucket = storage_client.bucket(project_id)
    blob = bucket.blob("pipeline-models/model.bst")
    blob.upload_from_filename(model_artifact.metadata['model_path'])

    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=f"gs://{project_id}/pipeline-models/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
        description="Pipeline-deployed churn prediction model",
    )
    print(f"Model registered: {model.resource_name}")
    return model.resource_name


# ═══════════════════════════════════════════════════════════════════
# Component 6: Deploy Model to Endpoint
# ═══════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model(
    project_id: str,
    model_resource_name: str
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location='us-central1')

    model = aiplatform.Model(model_resource_name)
    endpoint = aiplatform.Endpoint.create(
        display_name="churn-pipeline-endpoint"
    )

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="churn-pipeline-v1",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
    )
    print(f"Deployed to: {endpoint.resource_name}")


# ═══════════════════════════════════════════════════════════════════
# Pipeline DAG Definition
# ═══════════════════════════════════════════════════════════════════
@dsl.pipeline(
    name="churn-prediction-pipeline",
    description="End-to-end MLOps churn pipeline with conditional deployment gate"
)
def churn_pipeline(
    project_id: str = "mlops-project-489016",
    auc_threshold: float = 0.75
):
    extract_task = extract_data(project_id=project_id)

    feature_task = feature_engineering(
        input_dataset=extract_task.outputs["output_dataset"]
    )

    train_task = train_model(
        train_dataset=feature_task.outputs["train_dataset"]
    )

    eval_task = evaluate_model(
        test_dataset=feature_task.outputs["test_dataset"],
        model_artifact=train_task.outputs["model_artifact"],
        auc_threshold=auc_threshold
    )

    with dsl.Condition(eval_task.outputs["Output"] == True, name="deploy-gate"):
        register_task = register_model(
            project_id=project_id,
            model_artifact=train_task.outputs["model_artifact"]
        )
        deploy_task = deploy_model(
            project_id=project_id,
            model_resource_name=register_task.outputs["Output"]
        )


# ═══════════════════════════════════════════════════════════════════
# Compile + Submit
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time

    # Compile pipeline to YAML
    compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path="churn_pipeline.yaml"
    )
    print("Pipeline compiled to churn_pipeline.yaml")

    # Submit to Vertex AI
    aiplatform.init(project='mlops-project-489016', location='us-central1')

    job = aiplatform.PipelineJob(
        display_name="churn-prediction-run",
        template_path="churn_pipeline.yaml",
        pipeline_root="gs://mlops-project-489016/pipeline-root/",
        job_id=f"churn-pipeline-{int(time.time())}",
        parameter_values={
            "project_id": "mlops-project-489016",
            "auc_threshold": 0.75
        },
        enable_caching=False
    )
    job.submit()
    print(f"Pipeline submitted: {job.resource_name}")
