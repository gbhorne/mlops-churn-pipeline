# Building a Production MLOps Pipeline on Vertex AI-  From Notebook to Automated Deployment

*How I built an end-to-end churn prediction system with Kubeflow Pipelines, conditional deployment gates, and drift monitoring-  for under $2.*

---

Most ML projects die in notebooks. The model works, the metrics look good, and then it sits there-  never deployed, never monitored, never retrained. The gap between "model trained" and "model in production" is where MLOps lives, and it's where most data engineers and ML practitioners struggle.

I built this project to bridge that gap. Not with a toy example, but with a complete system: synthetic retail data in BigQuery, feature engineering, XGBoost classification, a six-stage Kubeflow pipeline on Vertex AI, conditional deployment gates, real-time endpoint serving, and prediction drift monitoring. The whole thing cost less than $2 to build.

Here's how I did it, what broke along the way, and the architectural decisions that matter.

## The Problem: Churn Prediction as an MLOps Vehicle

I chose churn prediction deliberately. Not because the ML is interesting-  it's a binary classifier on tabular data, which XGBoost handles in its sleep-  but because churn has the cleanest MLOps narrative. The evaluation gate is obvious (AUC above a threshold means deploy), the business case is clear (retain customers before they leave), and the monitoring story writes itself (feature distributions shift seasonally).

The ML is simple so the infrastructure can be complex. That's the point.

## Phase 1: Data Generation and the Leakage Trap

I generated synthetic retail data: 2,000 customers across four behavioral segments (high-value, regular, occasional, at-risk) with 201,000 transactions spanning 2023-2024. The data includes realistic patterns-  seasonal spikes in November and December, customer product preferences, and gradual disengagement for churned customers.

The churn label is defined as no purchase in the last 90 days. This is a standard definition, but it creates an immediate trap: if you include `days_since_last_purchase` as a feature, the model gets perfect accuracy. It's not predicting anything-  it's just learning the rule that defines the label. I caught this because the model returned 1.0 across every metric, which is never real.

After removing the leaky feature, the model settled at 0.8883 AUC with 80% accuracy. Realistic, defensible, and balanced across both classes. The feature importance showed purchase frequency as the dominant predictor at roughly 55%, which makes intuitive sense-  customers who buy rarely are more likely to stop entirely.

**Takeaway:** Always be suspicious of perfect metrics. If your model is too good, you have a data problem, not a model success.

## Phase 2: The Serving Container Trap

Model training is easy. Getting that model to serve predictions behind an HTTPS endpoint is where the friction starts.

My first deployment attempt used Vertex AI's sklearn serving container with a joblib-serialized XGBoost model. The endpoint spun up, the container loaded, and the model server immediately crashed. The error: the sklearn container doesn't know how to deserialize an XGBoost model.

My second attempt used the XGBoost serving container, but the model was trained with a newer XGBoost version that includes the `use_label_encoder` attribute. The serving container's older version didn't have that attribute. Another crash.

The fix was retraining the model without the deprecated parameter and saving in XGBoost's native `.bst` format instead of joblib. This format is compatible across XGBoost versions because it's the framework's own serialization, not Python's.

**Takeaway:** The serving container and the training environment must agree on serialization format and library version. Native model formats (.bst for XGBoost, SavedModel for TensorFlow) are more portable than Python-specific formats like joblib or pickle.

## Phase 3: The Pipeline-  Where MLOps Actually Happens

The Kubeflow pipeline has six components, each running in its own container:

**Extract Data** pulls customer features from BigQuery. One SQL query joins the customer dimension table with the transaction fact table, computes RFM metrics, and writes the result to GCS. The query runs in BigQuery's compute, so the pipeline container just executes the query and collects the result.

**Feature Engineering** applies StandardScaler and splits the data 80/20 with stratification. XGBoost is tree-based and doesn't technically need scaling, but I included it because the preprocessing step is architecturally important-  it demonstrates that the pipeline treats preprocessing as a versioned, swappable component.

**Train Model** fits an XGBoost classifier and saves the model in native `.bst` format. The hyperparameters are fixed because the focus is infrastructure, not model optimization.

**Evaluate Model** computes AUC, accuracy, precision, recall, and F1 on the test set. It returns a boolean: did the model pass the deployment threshold?

**Register Model** (conditional) uploads the model to Vertex AI Model Registry. This only executes if the evaluation gate passes.

**Deploy to Endpoint** (conditional) creates an HTTPS endpoint and deploys the model. Also conditional on the gate.

The conditional deployment gate is the architectural centerpiece. It's a `dsl.Condition` block in the pipeline DAG-  if AUC is below 0.75, the pipeline ends after evaluation. No registration, no deployment. This prevents model regression from silently reaching production. The threshold is a pipeline parameter, not a hardcoded value, so a team lead can raise it to 0.80 without touching code.

The pipeline took about 30 minutes to run end-to-end, including container provisioning for each step. Total compute cost: roughly $0.30.

### What Broke in the Pipeline

The first pipeline run failed because my BigQuery query used unqualified column names. `customer_id` exists in both the customer and transaction tables, and BigQuery correctly refused to guess which one I meant. The fix was adding table aliases: `c.customer_id` instead of `customer_id`.

But here's the subtle part: after fixing the code in my notebook and resubmitting, the pipeline still failed with the same error. Kubeflow caches component results, and since the component signature hadn't changed, it served the cached (failed) result. I had to resubmit with `enable_caching=False` to force re-execution.

**Takeaway:** Kubeflow caching is aggressive. When debugging pipeline failures, always disable caching to ensure your fix actually runs. Re-enable it once the pipeline is stable.

## Phase 4: Monitoring-  The Part Everyone Skips

A deployed model without monitoring is a liability. Feature distributions shift, upstream data pipelines break, and the model quietly degrades. You don't know until a stakeholder asks why the churn predictions stopped making sense.

I configured prediction drift detection using gcloud CLI with Jensen-Shannon divergence thresholds of 0.3 on all 8 input features. The monitor runs hourly and sends email alerts when any feature's distribution diverges significantly from the training baseline.

What this catches: seasonal shifts in spending patterns, data quality regressions, upstream schema changes, and population drift. What it doesn't catch: concept drift (the relationship between features and churn changes) or label drift (the actual churn rate shifts). Those require a feedback loop comparing predictions to actual outcomes-  a legitimate next phase.

## Cost Discipline

The biggest cost risk in Vertex AI isn't the pipeline runs ($0.03 each)-  it's the endpoints. A single n1-standard-2 endpoint costs $0.10/hour, which adds up to $72/month if you forget about it. My approach was strict: deploy, test, screenshot, undeploy within the same session. I set phone timers as safety nets and built a teardown script that nukes every billable resource in one command.

Total project cost: under $2. That buys data generation, a Workbench notebook for interactive development, multiple pipeline runs, endpoint deployments for testing, and monitoring configuration.

## What I'd Add for Production

Five things separate this from a true production system:

**Data validation.** A step between extraction and feature engineering that checks for schema changes, null spikes, and distribution anomalies before they reach the model. Great Expectations or TensorFlow Data Validation would slot in as another pipeline component.

**Hyperparameter tuning.** Replace hardcoded parameters with Vertex AI Vizier for Bayesian optimization. The pipeline architecture already supports this-  the train component would call Vizier and log the best parameters to the evaluation artifact.

**Canary deployment.** Instead of deploying to 100% traffic immediately, route 10% to the new model, monitor error rates, and gradually shift. Vertex AI Endpoints support traffic splitting natively.

**Feedback loop.** Capture actual churn outcomes and compare to predictions. This is the only way to know if the model is accurate on real data rather than just the test split.

**CI/CD.** The pipeline YAML is the deployable artifact. A Cloud Build trigger would compile, test, and submit the pipeline on code changes, with the same YAML running in staging and production.

## The Architectural Lesson

The model is the least interesting part of this project. XGBoost on 2,000 customers with 8 features is a solved problem. What matters is everything around it: the pipeline that automates retraining, the gate that prevents bad models from deploying, the registry that versions every artifact, the endpoint that serves predictions, and the monitor that catches drift.

Most ML engineers can train a model. The ones who can deploy, monitor, and maintain that model in production are the ones who build systems that last.

---

*This project is part of a portfolio demonstrating cloud data engineering and MLOps on GCP. The complete code, pipeline definitions, and architecture documentation are available on GitHub.*
