# MLOps Churn Pipeline Q&A

## System Design & Architecture

**Q: Walk me through the architecture of this pipeline. Why did you design it this way?**

The pipeline follows a six-stage DAG pattern on Vertex AI: Extract → Feature Engineering → Train → Evaluate → Register → Deploy. Each stage is a containerized Kubeflow component running on managed infrastructure, which means no persistent compute-  resources spin up per step and tear down after. I chose this over a monolithic training script because each component is independently testable, swappable, and versioned. The feature engineering step could be replaced with a Feature Store pull without touching the training logic. The evaluation step gates deployment-  if AUC drops below 0.75, the pipeline stops. This prevents model regression from silently reaching production. The architecture separates the data layer (BigQuery), the orchestration layer (Vertex AI Pipelines), the model layer (Model Registry), and the serving layer (Endpoints). That separation means each layer can scale, fail, and be debugged independently.

**Q: Why Vertex AI over a self-managed Kubeflow cluster?**

Three reasons. First, managed infrastructure - I don't want to operate a Kubernetes cluster just to run ML pipelines. Vertex AI abstracts the compute orchestration while still giving me the Kubeflow SDK for pipeline authoring, which is portable if I need to move later. Second, native integration with BigQuery and GCS - the data already lives in BigQuery from the enterprise analytics project, so the extraction step is a single query rather than a cross-cloud data transfer. Third, the Model Registry and Endpoint deployment are first-class Vertex AI concepts, not bolted-on services. The alternative was self-managed Kubeflow on GKE, which gives more control but adds operational overhead that wasn't justified for this use case. If the model needed custom serving logic or GPU inference, I'd reconsider.

**Q: How does the conditional deployment gate work, and why is it important?**

The evaluate-model component computes AUC-ROC on the held-out test set and returns a boolean. The pipeline DAG uses a `dsl.Condition` block-  if the boolean is True (AUC ≥ 0.75), the register-model and deploy-model components execute. If False, the pipeline ends after evaluation. This is critical because without it, every pipeline run deploys to production regardless of model quality. In a production setting with scheduled retraining, data distribution shifts could degrade model performance. The gate catches that. The threshold of 0.75 is configurable as a pipeline parameter, so a team lead could raise it to 0.80 without touching code. In a more mature setup, I'd also compare against the currently deployed model's metrics-  only deploy if the new model is strictly better.

**Q: You used XGBoost instead of a neural network. Why?**

The dataset has 2,000 customers with 8 tabular features. XGBoost dominates in this regime-  small-to-medium tabular data with engineered features. A neural network would overfit without aggressive regularization and wouldn't provide interpretable feature importances out of the box. XGBoost gave me 0.8883 AUC with default-ish hyperparameters in seconds of training time. The model choice also simplified serving-  Vertex AI has a pre-built XGBoost serving container, so I didn't need to build a custom prediction container. For a dataset with 10M+ rows, unstructured features, or multi-modal inputs, I'd revisit this decision.

## Data Engineering & Feature Design

**Q: How did you handle the data leakage issue with days_since_last_purchase?**

I caught it when the model returned perfect 1.0 scores across all metrics. The `days_since_last_purchase` feature is mechanically derived from the churn label-  churned is defined as no purchase in 90+ days, and `days_since_last_purchase` directly encodes that threshold. Including it means the model just learns the rule `if days > 90 then churn`, which isn't prediction-  it's tautology. I dropped the feature and retrained. AUC went from 1.0 to 0.8883, which is a realistic and defensible result. This is a common trap in churn modeling-  any feature that encodes future behavior relative to the prediction point creates leakage. In production, I'd add automated leakage detection as a pipeline step: flag any feature with correlation > 0.95 to the target.

**Q: Why StandardScaler and not other normalization approaches?**

XGBoost is tree-based and theoretically scale-invariant-  it shouldn't need feature scaling. However, I included StandardScaler for two reasons. First, it's a pipeline component that demonstrates preprocessing as a separable step, which matters architecturally-  in production you'd swap this for a Feature Store transformation. Second, the scaler parameters (mean, scale) become part of the model contract. Any client sending predictions needs to apply the same transformation. I stored the scaler parameters as pipeline artifact metadata so they're versioned alongside the model. If I were optimizing purely for XGBoost performance, I'd skip scaling. But the pipeline pattern matters more than the marginal accuracy difference.

**Q: How would you handle class imbalance if the churn rate were 5% instead of 47%?**

At 47%, the classes are nearly balanced, so no intervention was needed. At 5%, I'd take a layered approach. First, adjust the evaluation metric-  accuracy becomes meaningless at 5% churn, so I'd optimize for precision-recall AUC or F1 instead of ROC-AUC. Second, use XGBoost's `scale_pos_weight` parameter set to the ratio of negative to positive samples (19:1 at 5%). This is cheaper and more stable than SMOTE. Third, if that's insufficient, apply SMOTE or ADASYN in the feature engineering component-  importantly, only on the training split, never on the test split. The pipeline architecture supports this cleanly because feature engineering is a separate component. Fourth, adjust the deployment gate threshold and the prediction threshold-  at 5% churn, you might deploy at AUC > 0.70 and classify as churned at probability > 0.3 instead of 0.5.

## MLOps & Production Concerns

**Q: How would you implement CI/CD for this pipeline?**

The pipeline YAML is the deployable artifact, analogous to a Docker image. The CI/CD flow would be: code change triggers a Cloud Build pipeline that runs unit tests on each component, compiles the Kubeflow pipeline to YAML, submits a test run against a staging dataset, validates that evaluation metrics meet thresholds, and then stores the YAML in Artifact Registry as a versioned pipeline template. Deployment to production is a separate trigger that submits the tested YAML to the production Vertex AI environment. The key principle is that the same compiled YAML runs in staging and production-  only the pipeline parameters (project ID, dataset, thresholds) change. I'd use Terraform to manage the Vertex AI infrastructure and Cloud Build triggers.

**Q: The model is deployed. What happens next week when the data changes?**

This is where the monitoring setup matters. I configured prediction drift detection with 0.3 thresholds on all 8 features. If the statistical distribution of incoming prediction requests diverges from the training distribution by more than 0.3 (measured by Jensen-Shannon divergence), it triggers an email alert. The response workflow would be: alert fires, investigate which features drifted, determine if it's a data quality issue or a genuine distribution shift. If it's a shift, retrigger the pipeline to retrain on recent data. In a mature setup, I'd schedule the pipeline to run weekly via Cloud Scheduler, compare the new model's metrics against the deployed model, and auto-promote if the new model wins. The conditional deployment gate already supports this-  it's just a matter of comparing against the production model's AUC rather than a static threshold.

**Q: How would you scale this to handle 100M customers instead of 2,000?**

Three changes. First, the extract-data component would switch from `client.query().to_dataframe()` to a BigQuery export job that writes directly to GCS in Parquet format-  you can't pull 100M rows into a pandas dataframe. Second, the training component would move from a single-node XGBoost fit to Vertex AI's distributed training with XGBoost's distributed mode or switch to BigQuery ML for in-warehouse training. Third, the feature engineering component would move to Dataflow or Spark for distributed preprocessing, or use Vertex AI Feature Store for real-time feature serving. The pipeline DAG structure stays the same-  the components are swappable precisely because of the modular architecture. The evaluation, registry, and deployment steps don't change at all.

**Q: How do you handle model versioning and rollback?**

Every pipeline run creates a new version in the Vertex AI Model Registry. Each version carries metadata: the training dataset hash, the evaluation metrics, the pipeline run ID, and the training timestamp. If a deployed model starts underperforming, I can query the registry for the previous version's resource name and redeploy it with a single API call. The pipeline doesn't delete old versions-  they accumulate in the registry as an audit trail. In a more advanced setup, I'd implement canary deployment: deploy the new model to 10% of traffic, monitor error rates and prediction distributions, and gradually shift traffic to 100% if metrics hold. Vertex AI Endpoints support traffic splitting natively-  it's a parameter on the deploy call.

**Q: What would you add to make this truly production-ready?**

Five things. First, a data validation step between extraction and feature engineering-  Great Expectations or TensorFlow Data Validation to catch schema changes, null spikes, or distribution anomalies before they reach the model. Second, automated hyperparameter tuning using Vertex AI's Vizier service instead of hardcoded parameters. Third, A/B testing infrastructure with traffic splitting on the endpoint. Fourth, a feedback loop-  capture actual churn outcomes and compare to predictions, computing model accuracy on real data rather than just the test split. Fifth, alerting and dashboarding-  Cloud Monitoring dashboards showing prediction latency, error rates, drift metrics, and pipeline success rates, with PagerDuty integration for critical failures.

## Business Justification

**Q: What's the business case for investing in an MLOps pipeline versus running predictions in a notebook?**

A notebook requires a human to run it, validate results, and manually deploy. That's a bottleneck - the model gets stale between runs, there's no audit trail, and the human is a single point of failure. The pipeline runs on a schedule without intervention, automatically validates model quality before deployment, maintains a version history in the registry, and alerts when data drifts. The investment is maybe 40 hours of engineering time to build. The return is that the model stays current, deployments are safe, and the team doesn't need an ML engineer on-call to retrain. For a churn model specifically, even a 1% improvement in identifying at-risk customers can translate to significant revenue retention - the pipeline ensures that improvement compounds over time rather than degrading.
