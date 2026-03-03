# Architecture-  MLOps Churn Prediction Pipeline

## System Context

This system automates the lifecycle of a customer churn prediction model on Google Cloud Platform. It transforms raw retail transaction data in BigQuery into a production-deployed ML model behind an HTTPS endpoint, with automated quality gates and drift monitoring. The architecture is designed for repeatability-  every pipeline run produces a fully auditable, versioned model artifact.

## Infrastructure Layer

**Compute:** All training and pipeline execution runs on Vertex AI managed infrastructure. There are no persistent VMs or Kubernetes clusters to operate. Each Kubeflow component spins up a container on demand, executes, and tears down. This means cost accrues only during execution, not idle time.

**Storage:** Three storage systems serve distinct roles. BigQuery holds the source-of-truth retail data (dimension and fact tables) and the feature engineering view. Cloud Storage holds intermediate pipeline artifacts (train/test splits, model binaries, scaler parameters) and the compiled pipeline YAML. The Vertex AI Model Registry holds registered model versions with metadata, serving container references, and deployment history.

**Networking:** The endpoint serves predictions over HTTPS with TLS termination handled by Google's edge infrastructure. No VPC configuration is required for the standard deployment path. In a production environment, Private Service Connect would restrict endpoint access to internal networks.

## Data Flow

```
BigQuery (retail_gold)
  ├── dim_customer (2,000 rows: customer_id, region, signup_date, segment)
  └── fct_daily_sales (201K rows: order_id, customer_id, product_id, category, total_amount, order_date)
          │
          ▼
  v_customer_churn_features (SQL view: aggregates transactions into 10 customer-level features)
          │
          ▼
  Pipeline: Extract Data (BigQuery → CSV on GCS)
          │
          ▼
  Pipeline: Feature Engineering (StandardScaler, stratified 80/20 split)
          │
          ├── train_dataset (1,600 rows, 8 features, scaled)
          └── test_dataset (400 rows, 8 features, scaled)
                    │
                    ▼
  Pipeline: Train Model (XGBoost → model.bst on GCS)
                    │
                    ▼
  Pipeline: Evaluate Model (AUC, accuracy, precision, recall, F1)
                    │
              ┌─────┴─────┐
        AUC ≥ 0.75    AUC < 0.75
              │              │
              ▼              ▼
  Register Model       Pipeline ends.
  (Model Registry)     No deployment.
              │
              ▼
  Deploy to Endpoint
  (HTTPS prediction serving)
              │
              ▼
  Model Monitoring
  (drift detection, 0.3 threshold, hourly)
```

## Pipeline Components-  Design Decisions

### Extract Data
Runs a single BigQuery SQL query that joins dim_customer and fct_daily_sales, computes aggregate features (RFM metrics, behavioral signals), and writes the result as a CSV to GCS. The query runs in BigQuery's serverless compute, so there is no data movement until the result set materializes.

**Design choice:** The feature engineering SQL lives inside the pipeline component rather than as a persisted BigQuery view. This ensures the pipeline is self-contained-  the feature logic is versioned with the pipeline code, not managed separately in BigQuery. A production evolution would be to push this logic to Vertex AI Feature Store for real-time feature serving.

### Feature Engineering
Applies StandardScaler normalization and produces a stratified train/test split. Scaler parameters (mean, scale per feature) are stored as artifact metadata so they can be retrieved for inference-time preprocessing.

**Design choice:** StandardScaler is applied despite XGBoost being tree-based (and thus scale-invariant) because the preprocessing step is architecturally important-  it demonstrates that the pipeline treats preprocessing as a first-class, versioned component. In a multi-model setup, you would swap this component for model-specific preprocessing without touching the rest of the DAG.

### Train Model
Trains an XGBoost binary classifier with fixed hyperparameters and saves the model in native `.bst` format. The `.bst` format is required by Vertex AI's pre-built XGBoost serving container-  joblib format causes version compatibility issues between the training environment and the serving container.

**Design choice:** Hyperparameters are hardcoded rather than tuned because the focus is on the MLOps infrastructure, not model optimization. In production, this component would call Vertex AI Vizier for Bayesian hyperparameter search, with the best parameters logged to the evaluation metrics artifact.

### Evaluate Model
Computes five classification metrics and applies a configurable deployment gate. The gate threshold (AUC ≥ 0.75) is a pipeline parameter, not a hardcoded value. Returns a boolean that controls the conditional branch in the DAG.

**Design choice:** The gate compares against a static threshold rather than the currently deployed model. A production enhancement would query the Model Registry for the deployed model's metrics and only promote if the new model is strictly better. This prevents lateral moves that add deployment risk without performance gain.

### Register Model (Conditional)
Uploads the model artifact to GCS and registers it in the Vertex AI Model Registry with a pre-built XGBoost serving container reference. Only executes if the evaluation gate passes.

**Design choice:** The Model Registry serves as the single source of truth for model lineage. Each registration creates a new version, not a new model resource. This means the deployment history is a linear sequence of versions that can be queried, compared, and rolled back.

### Deploy to Endpoint (Conditional)
Creates an HTTPS endpoint and deploys the registered model with a single replica on n1-standard-2 compute. Only executes if the evaluation gate passes.

**Design choice:** The pipeline creates a new endpoint per run rather than updating an existing one. This is a deliberate simplification for the portfolio build. In production, the deploy component would target an existing endpoint and use traffic splitting to implement canary deployment-  route 10% of traffic to the new model, monitor, then shift to 100%.

## Model Monitoring

Prediction drift detection is configured via gcloud CLI with 0.3 Jensen-Shannon divergence thresholds on all 8 input features. The monitor runs on an hourly schedule and sends email alerts when any feature's distribution diverges from the training baseline.

**What it catches:** Seasonal shifts (holiday spending patterns), data quality regressions (null values in a feature), upstream schema changes (a feature gets renamed or dropped), and population drift (the customer mix changes over time).

**What it does not catch:** Label drift (actual churn rate changes) or concept drift (the relationship between features and churn changes). These require a feedback loop that compares predictions to actual outcomes, which is a Phase 2 enhancement.

## Security Model

The pipeline runs under the Compute Engine default service account with the following IAM roles: BigQuery Data Viewer (read source data), BigQuery Job User (execute queries), Storage Object Admin (read/write model artifacts), Vertex AI User (create pipeline runs, register models, deploy endpoints). In a production environment, a dedicated service account with least-privilege roles would replace the default account, and model artifacts would be encrypted with a customer-managed encryption key (CMEK).

## Cost Architecture

The system is designed for burst usage-  expensive resources (endpoints, notebooks) exist only during active development or testing sessions. Pipeline runs themselves cost $0.03 per run. The dominant cost driver is endpoint compute at $0.10/hour. The teardown script ensures no orphaned resources accumulate charges.

| Resource | Cost Model | Control |
|----------|-----------|---------|
| Pipeline runs | $0.03/run | Pay per use |
| Custom training containers | $0.10/hr per component | Auto-terminate |
| Endpoints | $0.10/hr while deployed | Manual undeploy / teardown script |
| Workbench notebooks | $0.07/hr while running | Manual stop |
| BigQuery | Free tier (< 1TB/month) | No action needed |
| Cloud Storage | $0.02/GB/month | Negligible |
| Model Monitoring | $0.01/hr per feature | Tied to endpoint lifecycle |

## Evolution Path

**Near-term:** Scheduled pipeline execution via Cloud Scheduler (weekly retraining), Vertex AI Feature Store for real-time feature serving, A/B testing with endpoint traffic splitting.

**Medium-term:** Terraform for infrastructure-as-code, Cloud Build CI/CD for pipeline YAML deployment, Great Expectations for data validation, feedback loop for actual-vs-predicted comparison.

**Long-term:** Multi-model serving (ensemble of XGBoost + logistic regression), custom prediction container with embedded preprocessing, streaming feature computation via Dataflow for real-time churn signals.
