# Demand planning

## 1. Overview & High-Level Objectives

* **Primary Goal**: Enable the customer to generate reliable SKU‐level demand forecasts in the cloud, with both on‐demand inference (via API) and batch forecasting (for all SKUs) on a schedule.

* **Key Pillars**:

  1. **Data Ingestion & Preprocessing**: Support multiple file formats (CSV, Parquet, Excel, JSON, etc.) from different producers; unify them into a single "master" transactions table at N-week granularity; clean, filter, and enrich.
  2. **Feature Engineering**:

     * Time features (sin, cos) to capture seasonality/cycles.
     * Categorical encoding (composition, gender, department, class, group, color, family, subclass, size, season, channel, brand, collection).
     * Outlier handling (z-score capping) and low-volume SKU elimination/configuration.
     * Meta-features (e.g., rolling averages, volatility, trend indicators).
  3. **Modeling**:

     * Dynamically construct an LSTM architecture with embedding layers for each categorical variable (so that if the customer later adds or removes categories, the network adjusts).
     * Offer two loss modes:

       1. **Point-by-Point Loss**: Standard per‐time‐step MSE/MAE on the M-point output sequence.
       2. **Flattened (Aggregated) Loss**: Sum over all forecast steps compared to “sum of actuals” to minimize total volume error.
     * Hyperparameter optimization with Optuna (e.g., number of LSTM layers, hidden size, learning rate, dropout, embedding dimensions).
     * Save per‐SKU error distributions for later confidence interval construction.
  4. **Cloud Deployment**:

     * **Batch Forecasting (Training)**: Launch on a GPU‐enabled instance or managed training service (e.g., SageMaker Training or EKS Job). At the end of each training run, produce:

       * Model artifacts (PyTorch checkpoint, tokenizer/encoder mappings, Optuna study).
       * Full‐catalog point forecasts for the next N weeks, stored in S3 or a relational store.
       * Per‐SKU error metrics (e.g., historical residuals) saved for confidence‐interval calculation.
     * **Online Inference**: Expose a RESTful endpoint (FastAPI or Flask) on a CPU‐backed container (e.g., ECS Fargate, Lambda with a larger memory allocation) that:

       * Reads the latest “preprocessed” features for a given SKU from S3 (or a feature store).
       * Runs a forward pass of the saved LSTM model (on CPU) to return an M-point forecast and associated confidence interval.
     * **Data Storage**: Raw transaction files in S3; “cleaned” master dataset in a data‐lake or data‐warehouse (e.g., AWS Redshift, Athena); model artifacts in S3; forecasts & error tables in S3 or an RDS/DynamoDB table.
     * **Scheduler & Orchestration**: Use AWS Step Functions or Managed Airflow to orchestrate daily/weekly ETL, training, and batch inference pipelines.

* **Additional “Standard” Demand‐Planning Features (Not Yet Mentioned)**:

  * **Aggregation & Hierarchy Management**: Demand planning is often done at multiple levels (e.g., store → region → country; product → family → category). We must enable roll-up forecasts at any level and disaggregate higher‐level forecasts to granular SKUs.
  * **Calendar Effects**: Built-in support for holidays, promotional events, markdowns, price changes, special campaigns. This requires either integrating a public holiday API or letting the user upload a promotions calendar.
  * **External Regressors**: Price, marketing spend, competitor actions, weather data, economic indicators. Ability for the customer to supply additional time‐series regressors and merge them in during feature engineering.
  * **Bayesian/Probabilistic Forecasting or Quantiles**: The current setup yields point forecasts + confidence intervals post hoc. It’s often useful to train models explicitly for quantile regression (e.g., pinball loss) or use Monte Carlo dropout to generate predictive distributions.
  * **Explainability / Feature Importance**: Shapley values or permutation importance at forecast time to help the user understand drivers of demand (e.g., “sales spike was driven by seasonality + promotion”).
  * **User Interface (UI) & Dashboarding**: Visual dashboards (e.g., QuickSight, Tableau, or a React‐based frontend) showing forecast vs. actuals, top outliers, SKU health metrics, forecast accuracy by category, etc.
  * **Manual Adjustments & Overrides**: End users may want to override model forecasts for specific SKUs (e.g., a one-off promotional event not seen in data). The system should let planners manually adjust and lock a forecast value.
  * **Automatic Retraining & Drift Detection**: Monitor model performance metrics; when forecast error (e.g., MAPE) drifts above a threshold, trigger an automated retraining. Detect feature drift in key regressors.
  * **Versioning & Auditability**: Track datapipeline versions, model versions, and deployment dates. Log user adjustments, retraining triggers, and data availability times for audit purposes.
  * **Notification & Alerting**: Send alerts (via email/Slack) when data ingestion fails, when forecast pipeline errors out, or when accuracy drops below a configured threshold.
  * **Security & Access Control**: Role‐based access (e.g., “planner,” “data engineer,” “admin,” “viewer”), encrypt data at rest/in transit, secure API endpoints with JWT or API keys, enforce least privilege on S3 buckets & RDS.
  * **Scalability & Cost Management**: Use auto‐scaling groups for inference containers, leverage spot instances for non‐urgent batch jobs, archive old data to Glacier, and right-size EC2/SageMaker instances.

---

## 2. Detailed Component Breakdown

### 2.1. Data Ingestion & Preprocessing

#### 2.1.1. Data Sources & Formats

* **Sources**: Multiple producers (e.g., vendor A, vendor B, vendor C) supply transaction logs in potentially different formats (CSV, XLSX, JSON, Parquet).
* **Ingestion**:

  1. **Landing Stage**: Raw files are uploaded by the user (or via an SFTP/FTP integration) into a designated “landing” S3 bucket (e.g., `s3://demand-planner-raw/producerA/2025-05-`\*).
  2. **File Polling**: A lightweight Lambda or EKS‐scheduled job periodically lists new files in the landing bucket and kicks off a preprocessing pipeline (e.g., AWS Step Functions or Airflow DAG).

#### 2.1.2. Column Selection & Initial Filtering

1. **Customer‐Specified Columns**: The pipeline reads the configuration file (e.g., a YAML or JSON manifest) that lists “useful” columns (e.g., `["SKU", "transaction_date", "quantity", "store_id", "color", "size", "description", …]`).
2. **Custom Code**:

   * Use a Python script (PySpark or pandas depending on scale) to load each file, drop unused columns in‐memory, and immediately discard any transaction rows that fail “basic” filtering (e.g., negative/zero quantity, missing SKU, or outside date range).
   * The code is modular so that the customer can plug in additional filters (e.g., exclude wholesale channels, exclude returns, etc.) via a config file.

#### 2.1.3. Format Heterogeneity & Standardization

* **Schema Unification**: Each producer’s file may use different column names (`“item_code”` vs. `"SKU"`, `"qty_sold"` vs. `"quantity"`). Maintain a “mapping table” (CSV/JSON) that defines how to rename columns into a unified schema at ingestion.
* **Missing Attributes Inheritance**:

  * Suppose Producer A’s transactions for SKU=123 lack `color` and `size` info, but Producer B’s records for the same SKU have them. The pipeline should:

    1. After loading and renaming all producers’ data into a single DataFrame, group by SKU and forward‐fill missing categorical fields (e.g., color, size, description) from the union of all producer records.
    2. A simple approach is:

       ```python
       temp = df.groupby("SKU").agg({
           "color": lambda x: x.dropna().unique().tolist(),
           "size": lambda x: x.dropna().unique().tolist(),
           ...
       })
       # Then merge back on SKU, preferring non-null values in a ‘first non-null’ fashion.
       ```

  * This ensures that every SKU has a complete set of static attributes (even if one producer omitted them).

#### 2.1.4. Time Resampling

* **Resampling Frequency**: Customer can choose “N weeks” (e.g., 1-week, 2-week, 4-week buckets).
* **Procedure**:

  1. Convert `transaction_date` → a standard datetime (ensure timezone consistency).
  2. Define the time index as week‐start dates (e.g., use Monday or configurable anchor).
  3. For each SKU × week, sum the `quantity` (after filtering/cleaning).
  4. If a given SKU has no transactions in a week, explicitly fill with zero (unless the customer wants to ignore zero‐sale weeks, but generally for time‐series, zeros matter for intermittent items).

#### 2.1.5. Categorical Encoding

* **List of Possible Categorical Variables**:

  * `["composition", "gender_description", "department_description", "class_description", "group_description", "color_description", "family_description", "subclass_description", "size", "season", "channel", "brand", "collection"]`.
* **Encoding Strategy**:

  1. **MultiLabelBinarizer**: For fields that can have multiple labels per SKU (e.g., composition might be `["Cotton", "Elastane"]`).
  2. **LabelBinarizer (One‐Hot)**: For single‐valued categorical fields (e.g., `gender_description = “Men”`).
* **Workflow**:

  1. Build a dictionary of encoders (e.g., `{"composition": MultiLabelBinarizer(...), "gender_description": LabelBinarizer(...), …}`).
  2. Fit the encoders on the **full historical dataset** during an **encoder‐fitting stage**, then save encoder objects (pickle or Torch’s `nn.Embedding` indices).
  3. During training or inference, load the saved encoder and transform each categorical field into a dense/one‐hot vector.
  4. **Dropping SKUs**: If a SKU has no categorical data at all (all missing or “Unknown”), drop it unless the customer flags it as “mandatory.”

#### 2.1.6. Outlier Handling & Low‐Volume SKU Removal

* **Outlier Definition**:

  * Compute, for each SKU, the z‐score of weekly sales. If a week’s `quantity` > (mean + z‐threshold × std), treat it as an outlier.
  * The z‐threshold (e.g., 3.0) is configurable per business logic.
* **Capping Strategy**:

  * Instead of dropping the entire SKU or row, “cap” the outlier sales to the threshold value (e.g., if z‐score>3, set quantity = mean + 3×std).
  * Log original vs. capped values to a diagnostics table for audit.
* **Low‐Volume SKU Removal**:

  * Compute total number of nonzero-sale weeks (or total quantity) per SKU.
  * If below a configured floor (e.g., fewer than 10 weeks of sales in the last 52 weeks), drop the SKU from modeling.
  * Allow the user to override: keep certain key SKUs even if they’re low volume (e.g., strategic lines, new products).

#### 2.1.7. Time Decomposition (Sin/Cos)

* **Motivation**: To capture cyclical patterns (e.g., seasonality by week of year).
* **Feature Construction**:

  1. Compute `week_of_year` or `day_of_year` from the resampled date.
  2. Add two features:

     * `time_sin = sin(2π × week_of_year / 52)`
     * `time_cos = cos(2π × week_of_year / 52)`
  3. If the customer wants multi‐seasonality (e.g., annual + quarterly), add additional pairs (e.g., `sin(2π × week_of_year / 26)`, etc.).

#### 2.1.8. Meta-Feature Engineering

* **Examples of Meta‐Features**:

  1. **Rolling Statistics**: Rolling mean, rolling standard deviation, rolling min/max over past K periods (configurable, e.g., 4-week or 12-week rolling).
  2. **Trend Indicators**: (current\_week − previous\_week) / previous\_week to capture short-term momentum.
  3. **Intermittency Metrics**: Durations of consecutive zero‐sale weeks, ratio of zero weeks.
  4. **Price/Promotion Flags** (if available): If a given week had a promotion or markdown, encode as binary.
* **Implementation**:

  * After resampling but before model‐data construction, augment each SKU-week row with these meta‐features.
  * Standardize/normalize continuous features (e.g., z-score across training set) and save scalers to S3 for inference time.

---

### 2.2. Time Series Dataset Construction

#### 2.2.1. Defining Lookback (L) & Forecast Horizon (M)

* **Lookback Window (L)**: Number of past timesteps used as input (e.g., last 52 weeks). Configurable by customer.
* **Forecast Output Length (M)**: Number of future timesteps to predict (e.g., next 13 weeks). Also configurable.

#### 2.2.2. Sliding‐Window Creation

1. For each SKU, take its resampled series of length T (after dropping low‐volume SKUs).
2. Starting from `t = L`, create input sequences:

   * `X_t = [features at t-L, t-L+1, …, t-1]`
   * `y_t = [quantity at t, t+1, …, t+M-1]`
3. Only generate windows where all L+M points exist. Optionally allow “padding” at beginning if the customer wants to keep the first few windows (less common).

#### 2.2.3. Feature Matrix & Target Matrix

* **X\_t** consists of:

  * **Numeric features**: previous weekly quantities (vector of length L), rolling means, rolling std, time\_sin/cos per timestep.
  * **Categorical embeddings**: each categorical column has been transformed into either:

    * A multi‐hot vector (if MultiLabelBinarizer), or
    * A one‐hot vector (if LabelBinarizer), or
    * A learned embedding index (if you want to feed into `nn.Embedding`).
* **y\_t** consists of the raw quantities (untransformed), which the model will predict directly (but you can also optionally apply a log1p or Box‐Cox transform to stabilize variance, then invert it at inference).

#### 2.2.4. Train/Validation/Test Split

* **SKU‐wise Partitioning**:

  * Option A: Time‐based split across all SKUs (e.g., last 13 weeks for validation, previous 13 weeks for test).
  * Option B (if user wants “cold‐start” evaluation): Reserve some SKUs as “out‐of‐sample” entirely. Less common for demand planning, but possible if new SKUs must be forecast.
* **Cross‐Validation**:

  * For time series, use a rolling‐forecast CV (e.g., train on weeks 1–T1, validate on T1+1–T1+M; then expand training window, etc.). This is embedded in the Optuna tuning process.

---

### 2.3. Model Architecture

#### 2.3.1. Dynamic Handling of Categorical Features

* **Embedding Layers**:

  * For each categorical field (e.g., `gender_description` has V\_g distinct values), create an `nn.Embedding(V_g, d_g)` where `d_g = min(50, ceil(log2(V_g)))` (or some rule of thumb).
  * The model building code inspects the fitted encoders to discover:

    ```python
    categorical_dims = { “gender_description”: V_g, “brand”: V_b, … }
    embedding_dims = { k: min(50, ceil(V/2)) for k, V in categorical_dims.items() }
    ```

  * At runtime, you concatenate all embedding outputs into a single vector of length `D_cat = ∑ d_g`.
* **Continuous Numeric Inputs**:

  * The L continuous features (including past quantities, rolling stats, time\_sin/cos) form a tensor of shape `(batch_size, L, N_cont)`.
  * You can apply a linear layer to project them into a hidden dimension before feeding to LSTM, or feed them directly if the LSTM expects `(batch_size, seq_len, input_size = N_cont + D_cat_total)`.
* **Final Architecture** (Example):

  1. **Embedding Block**:

     * Inputs: one integer index per categorical column for each timestep (if the categorical is static across time, you can tile the embedding across all L timesteps or treat it separately).
     * Output: `(batch_size, D_cat)` per SKU.
  2. **Concatenate** `(L × N_cont)` with static `D_cat` (either by repeating `D_cat` across L timesteps or by appending `D_cat` after LSTM). Two common strategies:

     * **Strategy A**: For each timestep t in the lookback, concatenate `embedding_vector` with `numeric_features_t`, giving a per‐timestep input of size `(N_cont + D_cat)`, then feed this sequence of length L into the LSTM.
     * **Strategy B**: Run LSTM on `numeric_features` alone (`L × N_cont`), then concatenate the final LSTM hidden state (size H) with `embedding_vector` (size `D_cat`), and pass through one or more dense layers before output.
  3. **LSTM Layers**:

     * Stack of `n_layers` LSTM layers, hidden size `H` (tuned by Optuna). Potentially add dropout between layers.
  4. **Fully Connected & Output**:

     * If using **Point‐by‐Point Loss**: Last hidden state → Dense → Output shape `(M,)` (predict next M weeks).
     * If using **Flattened Loss**: After the LSTM and dense layers, apply a “sum” operator over the output dimension (e.g., output shape `(M,)`, then sum them, so loss is `|sum(predictions) − sum(actuals)|`). Internally this means the backward pass encourages correct total volume rather than shape. The network still outputs an M-length vector so you can evaluate both point and aggregate errors at inference.

#### 2.3.2. Custom Loss Functions

* **Point‐by‐Point (Standard)**:

  ```python
  def point_loss(pred, actual):
      # pred, actual: shape (batch_size, M)
      return torch.mean((pred - actual)**2)  # or MAE, Huber, etc.
  ```

* **“Flattened” Aggregate Loss**:

  ```python
  def sum_loss(pred, actual):
      # Reduce across the M horizon first:
      pred_sum = pred.sum(dim=1)       # shape (batch_size,)
      actual_sum = actual.sum(dim=1)   # shape (batch_size,)
      return torch.mean((pred_sum - actual_sum)**2)
  ```

* **Switching Mode**:

  * The training script accepts a config flag `--loss_mode = ["point", "sum"]`.
  * Internally, if `loss_mode == "sum"`, it computes `sum_loss` for backpropagation; if `"point"`, it uses `point_loss`.
  * At evaluation time, always compute both (so you can compare “point accuracy” vs. “aggregate accuracy”).

#### 2.3.3. Hyperparameter Tuning with Optuna

* **Search Space Examples**:

  ```yaml
  n_layers: IntUniform(1, 3)
  hidden_size: IntUniform(32, 256, step=32)
  dropout: FloatUniform(0.0, 0.5)
  learning_rate: LogUniform(1e-4, 1e-2)
  batch_size: Categorical[32, 64, 128]
  embedding_dim_rule: lambda V: min(50, ceil(V/2))  # fixed, or tune a scale factor
  ```

* **Procedure**:

  1. Define an Optuna objective that:

     * Builds a model with sampled hyperparameters.
     * Trains for a small number of epochs (e.g., 5) on the training set.
     * Evaluates on the validation fold using the chosen loss mode (point or sum).
     * Returns the validation loss.
  2. Run `study = optuna.create_study(direction="minimize")` for a configured number of trials (e.g., 50–100).
  3. Save the best trial’s hyperparameters into a JSON file in S3 (e.g., `s3://demand-planner-models/best_params.json`).
  4. Re‐train the final model on the entire train+val set using those best hyperparameters (for a larger number of epochs, e.g., 50–100), with early stopping if needed.

---

### 2.4. Post‐Training: Error Tracking & Confidence Intervals

#### 2.4.1. Storing Per‐SKU Prediction Errors

* After final training, run the **model inference** on the historical windows (within the training period) to collect residuals `e_{sku, t} = y_{t} − ŷ_{t}` for each SKU and each forecast horizon step.
* Aggregate residuals to estimate an error distribution per SKU, for each forecast lead time (1-week ahead, 2-weeks ahead, …, M-weeks ahead).
* Store these residuals (or summary stats: mean, std, quantiles) in a dedicated table (e.g., `sku_error_distributions`) in an RDS or DynamoDB, or as Parquet/CSV in S3 (e.g., `s3://demand-planner-metrics/sku_error_stats/`).

#### 2.4.2. Confidence Interval Calculation at Inference

* **Assumption**: Errors are roughly Gaussian (check via QQ plots). If satisfied, a 95% CI for SKU i at lead time ℓ is:

  $$
    \hat{y}_{i, t+ℓ} \pm 1.96 \times σ_{i,ℓ}
  $$

  where $σ_{i,ℓ}$ is the standard deviation of historical residuals for SKU i at horizon ℓ.
* If errors are non‐Gaussian, use empirical quantiles (e.g., 2.5% and 97.5%) stored in the residuals table.
* The inference code will:

  1. Given SKU i and forecast horizon ℓ, load `μ_{i,ℓ}` (mean error) and `σ_{i,ℓ}` from the error table (if mean ≈ 0, you can drop μ).
  2. Output point forecast $\hat{y}_{i,t+ℓ}$ ± CI.
  3. If the customer wants probabilistic predictions at multiple quantiles (e.g., 50%, 90%), return those.

---

## 3. Critique of the Existing Pipeline & Gaps

1. **Seasonality Beyond Weekly**

   * You decompose time into sin/cos of “week\_of\_year.” This captures annual seasonality at weekly granularity.
   * **Missing**: Quarterly or monthly seasonality (e.g., fashion cycles) might require additional sin/cos pairs or learned periodic embeddings. If SKUs have strong intra‐week patterns (e.g., weekend vs. weekday), a daily decomposition could help.
   * **Recommendation**: Allow the user to configure multiple seasonalities (yearly, quarterly, monthly, even day-of-week if daily data is used).

2. **External Regressors & Promotions**

   * The pipeline does not currently ingest promotional calendars, markdown events, competitor promotions, or macroeconomic indicators.
   * **Standard Practice**: Demand‐planning frequently uses a “promotional\_flag” feature, price elasticity data, and store‐level foot‐traffic stats. Missing these may limit accuracy for promotional spikes.
   * **Recommendation**: Design a generic `regressors` table (time, SKU, regressor\_name, value) that can be merged in at a weekly level.

3. **Hierarchical Forecasting & Aggregation**

   * The current method builds SKU‐level models in isolation. There is no mechanism to ensure that, e.g., “Category X total forecast = sum of SKU forecasts in Category X.”
   * **Standard Demand‐Planning**: Once SKU forecasts are computed, re‐conciling them at category/region levels is crucial. Common approaches are:

     * **Top‐Down / Bottom‐Up**: Forecast at aggregate levels then disaggregate (or vice versa).
     * **Proportional Reconciliation**: Use Min‐T (Minimum Trace) methods or forecast reconciliation frameworks.
   * **Recommendation**: After SKU‐level forecasts, implement a reconciliation step (e.g., the Hyndman “forecast reconciliation” library) so that hierarchical consistency holds.

4. **Cold‐Start New SKUs**

   * New SKUs (with no history) will be dropped or cannot be forecasted.
   * **Standard**: Use “analogous SKUs” or category‐level average growth rates to generate a baseline forecast for new items.
   * **Recommendation**: Create a rule‐based fallback: if SKU history < L weeks, forecast using median growth of its product family or “parent category” SKU cluster.

5. **Retraining & Drift Detection**

   * No mention of automated retraining triggers or feature‐drift detection.
   * **Standard**: Continuously monitor forecast accuracy (e.g., MAPE over last 4 weeks) and trigger a retraining if it exceeds a threshold (e.g., 15%).
   * **Recommendation**: Add a monitoring service (e.g., AWS CloudWatch or a custom Airflow DAG) that compares actual weekly sales vs. previous forecasts, computes error metrics, and when drift is detected, automatically enqueue a new training job.

6. **Model Explainability**

   * Pure LSTM lacks interpretability. At a minimum, you should:

     * Compute permutation‐feature‐importance on the numeric features.
     * For embeddings (categories), compute how much each embedding dimension contributes to output variance via SHAP.
   * **Recommendation**: Integrate an explainability step post‐forecast that populates “feature\_contributions” in the output to help end users understand why a forecast changed.

7. **Cold‐Start & Warm‐Start of Models**

   * If a new category or new brand is introduced, the embedding layer must be retrained. The documentation should clarify that any new unseen categorical values require retraining (or an “unknown” bucket).
   * **Recommendation**: Provide a “catch‐all” `<UNK>` embedding for unseen categories at inference, plus a scheduled re‐training to rebuild embeddings whenever a new category appears.

8. **Evaluation Metrics**

   * Only the “loss” (MSE or aggregated) is mentioned.
   * **Standard Metrics**: MAPE, RMSE, MAE, sMAPE, and P50/P90 coverage for the confidence intervals.
   * **Recommendation**: At each stage, compute a suite of metrics on validation/test sets. Persist those to S3 or a metrics DB for dashboarding.

---

## 4. Additional “Standard” Features to Add

1. **Promotional & Price Elasticity Integration**

   * Let the customer upload a “Promotion Calendar” file (time window, SKU, discount %, promo type).
   * Add a “price” time series per SKU so the model can learn price elasticity.
   * Potentially include a “promotion\_lag” feature: number of weeks since last promo.

2. **Holiday & Event Calendar**

   * Provide a default holiday calendar for the customer’s region (e.g., Italy, Europe) and let them upload region‐specific events (e.g., Black Friday, Christmas sale).
   * Generate “is\_holiday” or “days\_until\_holiday” features (numeric or one‐hot) in the time decomposition stage.

3. **Aggregation & Reconciliation Module**

   * After generating SKU‐level forecasts, automatically roll up to:

     * **Category**, **Family**, **Department** levels using a user‐defined hierarchy table.
     * Use hierarchical reconciliation (e.g., bottom‐up, top‐down, or Min‐T) to ensure all levels sum correctly.

4. **Manual Forecast Adjustment UI**

   * A lightweight web interface (React + Tailwind) that:

     * Shows SKU‐level forecast vs. actual chart.
     * Allows planners to slide bars or type adjustments (e.g., “override week 2 from 120 to 150 units”).
     * Locks adjusted forecasts so they’re not overwritten by the next batch run.

5. **Dashboard & Reporting**

   * Use a BI tool (e.g., AWS QuickSight, Tableau, or a built-in Plotly/Dash app) to create:

     * **Accuracy Dashboard**: MAPE by category, by region, by SKU, updated weekly.
     * **Forecast vs. Actuals**: Time series plots per SKU or aggregated view.
     * **Outlier Reports**: Which SKUs had large forecast errors or huge outlier corrections.

6. **Data Versioning & Lineage**

   * Keep track of:

     * Raw data file names/versions ingested.
     * Preprocessing code version (Git SHA).
     * Encoder versions (pickled objects with a timestamp).
     * Model version (Git tag + Optuna trial ID).
   * This enables “re‐run exactly” for audit, and comparing performance across model versions.

7. **Alerting & Notifications**

   * Configure thresholds (e.g., MAPE > 20% in any category) → send Slack/email notifications.
   * Monitor data ingestion health: if no new files arrive by X:00 every day, send an alert.

8. **Multi‐Horizon & Scenario Planning**

   * Support “what‐if” scenario runs: e.g., simulate ±10% changes in price next quarter, then forecast the net sales.
   * Let the user specify “target scenario” variables (e.g., budget for marketing spend) and rerun features/regressors accordingly.

9. **API Versioning & Documentation**

   * Use OpenAPI (Swagger) to define the inference API.
   * Version the API (e.g., `/v1/forecast/skus`, `/v2/forecast/skus`) so that backward compatibility can be maintained if the model’s input schema changes.

10. **Security & Access Control**

    * **IAM & S3**: Restrict S3 buckets so that only the ETL role can write to “cleaned” data, only the model‐serving role can read “model artifacts,” etc.
    * **API Gateway + Cognito**: Use JWT tokens to authenticate API calls. Each planner or analyst has a role with scoped access.
    * **Encryption**: All S3 buckets and RDS/DynamoDB tables must have server‐side encryption enabled.

---

## 5. Cloud Deployment & Architecture

Below is a recommended AWS‐centric design. (Replace with analogous GCP/Azure services if desired.)

````
[ SOURCE SYSTEMS ] 
      ↓
+----------------------------+
| 1. Raw Ingestion / Landing |
|   - S3: s3://demand-planner-raw/                  |
|   - IAM Role: “demand-planner-ingest-role”         |
+----------------------------+
      ↓ (Triggered by S3 Event or Scheduled Polling)
+----------------------------+
| 2. Preprocessing Pipeline  |
|   - AWS Lambda (small files) or AWS Batch/EKS Job |
|   - PySpark/Pandas scripts read raw files, drop   |
|     unused columns, standardize schemas, clean,   |
|     cap outliers, encode categoricals, etc.       |
|   - Output → s3://demand-planner-processed/       |
+----------------------------+
      ↓ (Catalog & Partition in Glue or Athena)
+----------------------------+
| 3. Feature Store / Data Lake |
|   - AWS Glue Data Catalog / AWS Athena tables     |
|   - “transactions_cleaned” partitioned by date    |
+----------------------------+
      ↓ (Triggered daily/weekly via Step Functions)
+------------------------------------------+
| 4. Batch Training & Full‐Catalog Forecast|
|   - AWS Step Functions orchestrates:     |
|     1. **Prepare Training Data**:        |
|        - Query Athena or read from S3    |
|        - Construct sliding windows,      |
|          train/val/test splits, encoders |
|     2. **Hyperparameter Tuning**:        |
|        - SageMaker Hyperparameter Tuning |
|          Job (or custom Optuna in an EKS |
|          Pod).                           |
|     3. **Model Training**:               |
|        - SageMaker Training Job (GPU) or |
|          EKS Job (PyTorch, PyTorch Lightning)|
|     4. **Save Artifacts**:               |
|        - Best model checkpoint → S3      |
|        - Encoder pickles, Scaler pickles → S3  |
|        - Residual/Error stats → RDS/DynamoDB or S3 |
|     5. **Full‐Catalog Forecast**:        |
|        - Load final model, run inference on all SKUs (sliding windows) |
|        - Store full forecasts → S3 (e.g., `s3://demand-planner-forecasts/YYYY-MM-DD/`) |
+------------------------------------------+
      ↓
+------------------------------+
| 5. Post‐Processing & Loading |
|   - Optionally load forecasts |
|     into an RDS (e.g., Postgres) or DynamoDB|
|   - Update BI Dashboard (QuickSight) with |
|     new forecasts and accuracy metrics     |
+------------------------------+
      ↓
+----------------------------------+                  +------------------------+
|6a. Scheduled Batch Inference     | ←—— (trigger: daily/weekly )  | 6b. On‐Demand API Inference |
|   (can overlap with “full‐catalog”)   |                |                        |
|   - AWS Batch/EKS runs inference on |                |  - AWS API Gateway     |
|     designated SKUs (e.g., top 100) |                |  - FastAPI container   |
|   - Uses CPU instances (e.g., C5)   |                |  - ECS Fargate or Lambda|
|   - Reads latest encoders/scalers   |                |  - Reads model from S3 |
|   - Writes per‐SKU forecasts to S3/RDS|                |  - Returns JSON response|
+----------------------------------+                  +------------------------+

**Legend of Key Services**  
- **S3**: Data lake for raw files, processed data, model artifacts, forecasts.  
- **Lambda / AWS Batch / EKS**: For data preprocessing. Choice depends on data volume:
  - Small (<100 MB/day): Lambda with pandas.
  - Medium (100 MB–10 GB/day): AWS Batch with a PySpark/Python script.
  - Large (>10 GB/day): EKS with a Spark cluster or EMR.  
- **AWS Glue & Athena**: Metadata catalog & serverless SQL on processed data.  
- **SageMaker (Optional)**: 
  - **Hyperparameter Tuning Jobs**: Native integration with Optuna equivalent or SageMaker’s built‐in tuner.
  - **Training Jobs**: If you prefer fully managed GPU training.  
- **EKS / ECS**: 
  - EKS for custom container jobs (e.g., training if you want full control).  
  - ECS (Fargate) for inference containers (FastAPI).  
- **API Gateway**: Fronts the inference endpoint; handle authentication, throttling.  
- **RDS / DynamoDB**:  
  - **RDS (Postgres)**: Store error distributions, final forecasts, user adjustments, parameter configurations.  
  - **DynamoDB**: If you need a serverless key‐value store for per‐SKU metadata (very low latency).  
- **CloudWatch**: Centralized logs, error alerts, metric dashboards (e.g., training loss over time).  
- **IAM**: Fine‐grained roles for each service (e.g., “IngestionRole,” “TrainingRole,” “InferenceRole”).  
- **Cognito or IAM Auth**: For JTW tokens on the API, restricting forecast endpoints to authenticated users.  
- **QuickSight / Tableau**: User‐facing dashboards for forecast vs. actual, accuracy metrics, and drift warnings.  
- **Step Functions**: Orchestrates end‐to‐end pipelines (ETL → training/tuning → forecasting → loading → notifications).

---

## 6. Complete Feature List & User Benefits

| Feature | Description | User Benefit |
|---|---|---|
| **Multi‐format Ingestion** | Load CSV/Excel/JSON/Parquet from any producer; auto‐map to unified schema. | Minimizes manual ETL; one pipeline for all vendors. |
| **Configurable Filtering** | Plugin filters (e.g., exclude returns, negative sales). | Customer controls business logic without code changes. |
| **SKU‐level “Missing Attribute Inheritance”** | If color/size missing from one vendor, inherit from another. | Ensures no SKU is dropped for lacking static attributes. |
| **Resampling to N-week Buckets** | Configurable weekly grouping. | Customer picks granularity (1-week, 2-week, 4-week) to match planning cycles. |
| **Categorical Encoding** | MultiLabelBinarizer & LabelBinarizer pipelines, with dynamic embedding layers. | Model automatically adapts if new categories appear; captures rich SKU attributes. |
| **Outlier Detection & Capping** | Z-score‐based capping of extreme weekly sales. | Removes “bad data” spikes (e.g., duplicate shipments, erroneous entries) without discarding entire SKU. |
| **Low‐Volume SKU Handling** | Drop SKUs below threshold or keep mandatory keys. | Focus model capacity on reliable signals; optional preservation of strategic items. |
| **Time Decomposition** | Sin/Cos features for weekly seasonality, optionally monthly/quarterly. | Captures cyclic patterns systematically. |
| **Meta‐Features** | Rolling means, volatility, trend indicators, intermittency metrics. | Provides the LSTM rich inputs to learn level, trend, seasonality, and noise components. |
| **Dynamic LSTM + Embeddings** | Handles any number of categorical variables; can adjust embedding dims via rule. | Future‐proof architecture easily accommodates new data fields. |
| **Two Loss Modes** | “Point‐by‐point” vs. “Flattened (Sum) loss.” | Users can choose to minimize per‐period accuracy or total volume error depending on business needs. |
| **Hyperparameter Tuning with Optuna** | Automated search over layers, hidden size, learning rate, dropout, etc. | Ensures near‐optimal model configuration without manual grid search. |
| **Per-SKU Error Tracking** | Store residuals/histogram per horizon; compute CI. | Provides credible intervals for each forecast, builds trust, and informs safety stocks. |
| **Batch Training & Forecasting** | Scheduled training + full‐catalog forecasting. | Ensures that entire catalog is always up‐to‐date; can run overnight at low‐cost hours. |
| **On‐Demand API Inference** | FastAPI endpoint, CPU only, returns (forecast, CI) per SKU. | Planners or downstream apps can request specific SKU forecasts at any time without spinning up GPU. |
| **Holiday & Promotion Integration** | Default holiday calendar + uploadable promo file. | Captures demand spikes around key events; improves accuracy. |
| **Hierarchical Forecast Reconciliation** | Automatically roll up/disaggregate forecasts along product/store hierarchies. | Guarantees consistency across levels (e.g., sum of SKUs = category total). |
| **Explainability** | Feature importance & SHAP values per forecast. | Helps users understand “why” demand is forecasted a certain way; builds confidence. |
| **Manual Adjustment UI** | Web interface to override model forecasts, lock values. | Empowers planners to incorporate domain knowledge or one-off events. |
| **Dashboard & Reporting** | Visuals for forecast vs actual, accuracy metrics, outlier alerts. | One place for planners to monitor performance and drill into SKU‐level details. |
| **Automated Retraining & Drift Detection** | Retrain when MAPE > threshold or data drift detected. | Maintains model freshness; prevents stale forecasts degrading accuracy. |
| **Data Lineage & Versioning** | Track file versions, code versions, model versions in metadata. | Full audit trail; easy rollback to previous model/data if needed. |
| **Security & Access Control** | IAM roles, API authentication, encryption at rest/in transit. | Keeps sensitive sales data and forecasts secure and compliant. |
| **Scalable Cloud Architecture** | S3, Glue, Athena, SageMaker/EKS/ECS, RDS; auto‐scaling, serverless where possible. | Minimizes operational overhead; pay only for what you use; easy to extend. |

---

## 7. Detailed Cloud Architecture Diagram (Textual)

```text
┌────────────────────────────┐
│ 1. Raw Data Ingestion     │
│                            │
│  - S3 “landing” bucket     │
│  - Producers upload files  │
│  - S3 Events trigger Lambda│
└────────────────────────────┘
               ↓
┌────────────────────────────┐
│ 2. Preprocessing Service   │
│                            │
│  - AWS Lambda / AWS Batch  │
│    or EKS PySpark Job      │
│  - Renames columns, cleans,│
│    standardizes, encodes   │
│  - Writes to “processed”   │
│    S3 bucket               │
└────────────────────────────┘
               ↓
┌────────────────────────────┐
│ 3. Data Lake / Feature     │
│    Store                   │
│                            │
│  - AWS Glue Data Catalog   │
│  - Athena tables:          │
│    “transactions_cleaned”  │
│  - Handles partitioning by │
│    date or SKU category    │
└────────────────────────────┘
               ↓
┌───────────────────────────────────────────────────────────────────┐
│ 4. Orchestrator (AWS Step Functions or Airflow)                  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 4a. Training / Tuning Pipeline                              │   │
│  │                                                             │   │
│  │  - Read cleaned data from S3 / Athena                       │   │
│  │  - Build sliding‐window datasets                            │   │
│  │  - Fit categorical encoders & scalers, save to S3           │   │
│  │  - Launch Optuna tuner (SageMaker Tuning Job or EKS Pod)    │   │
│  │  - Retrieve best hyperparameters, re‐train full model (SageMaker Training or EKS) │   │
│  │  - Run historical inference to compute residuals per SKU    │   │
│  │  - Save residuals to RDS/DynamoDB or S3                      │   │
│  │  - Full‐catalog forecasting (all SKUs)                       │   │
│  │  - Save forecasts to S3 & optionally load them into RDS     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 4b. Batch Inference Pipeline                                   │   │
│  │                                                               │   │
│  │  - Scheduled weekly/daily job                                  │   │
│  │  - Reads latest day’s features from S3 / feature store         │   │
│  │  - Runs on CPU instance (EKS or AWS Batch) for selected SKUs   │   │
│  │  - Saves results to S3 and RDS                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
               ↓
┌───────────────────────────────────────┐    ┌───────────────────────────────────────┐
│ 5a. On‐Demand Inference REST API      │    │ 5b. Dashboard & Analytics             │
│                                       │    │                                       │
│  - API Gateway → ECS Fargate (FastAPI)│    │  - QuickSight / Tableau                │
│  - Reads encoder & model from S3      │    │  - Queries RDS / DynamoDB / S3 data    │
│  - Returns point forecast + CI        │    │  - Displays forecast vs. actual charts │
│  - Auth via Cognito / IAM             │    │  - Shows MAPE, RMSE, Bias by SKU/etc.  │
└───────────────────────────────────────┘    └───────────────────────────────────────┘
````

---

## 8. Data & Compute Flow Summary

1. **Data Arrival**

   * Producers drop files in S3 → triggers preprocessing.
2. **Preprocessing**

   * Clean/filter → unify schema → inherit missing attributes → time resample → encode categoricals → cap outliers → generate meta-features → save “feature store” to S3/Glue.
3. **Model Training & Tuning**

   * Build sliding‐window dataset → train LSTM with embeddings → tune with Optuna → save best model & artifacts → compute residuals → produce full‐catalog forecasts → store predictions and metrics.
4. **Inference**

   * **Batch**: Scheduled job processes all SKUs, writes to S3 or RDS.
   * **Online**: API endpoint loads static features from S3/feature store, runs CPU inference, returns forecast & CI.
5. **User Interaction**

   * Planners view dashboards, drill into SKU-level performance, adjust forecasts.
   * Adjustments are stored in RDS and override model outputs when exporting to ERP/BI.

---

## 9. Security & Compliance

* **IAM Roles & Policies**

  * **IngestRole**: `s3:GetObject` on `…/raw/*` and `s3:PutObject` on `…/processed/*`.
  * **TrainingRole**: `s3:GetObject` on `…/processed/*`, `s3:PutObject` on `…/models/*`, `rds-db:connect` if writing metrics to RDS.
  * **InferenceRole**: `s3:GetObject` on latest model artifact & encoders.
  * **DashboardRole**: `quicksight:PassRole`, `athena:StartQueryExecution`, `athena:GetQueryResults`, `s3:GetObject` on forecast tables.

* **Network Security**

  * Place inference containers behind a private Application Load Balancer.
  * Enforce HTTPS (TLS 1.2+) on API Gateway.
  * Use VPC endpoints for S3 to ensure traffic never leaves AWS backbone.

* **Data Encryption**

  * S3 at rest: SSE‐S3 or SSE‐KMS for all buckets (`raw/`, `processed/`, `models/`, `forecasts/`).
  * RDS: Enable “Encryption at rest” via KMS.
  * In transit: Enforce SSL/TLS for all data movement (e.g., Athena queries, training containers pulling artifacts).

* **Authentication & Authorization**

  * API protected by Cognito User Pool/JWT.
  * Each user has a role (e.g., Planner, Data Engineer, Admin). RBAC enforced in the UI & API.

* **Audit Logging**

  * Enable CloudTrail on S3 buckets and RDS.
  * Inference API writes logs (request, SKU, timestamp, user ID) to CloudWatch Logs.
  * Step Functions log each state transition, capturing any errors in the pipeline.

---

## 10. Suggested Improvements & Roadmap Items

| Priority | Feature                                             | Rationale                                                                            | Timeline    |
| -------- | --------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------- |
| High     | **Holiday & Promotion Integration**                 | Without these, accuracy suffers during sales events.                                 | 1–2 sprints |
| High     | **Hierarchical Reconciliation**                     | Ensures consistency for category‐level planning.                                     | 2–3 sprints |
| Medium   | **Automated Retraining & Drift Detection**          | Keeps the model up to date without manual intervention.                              | 3–4 sprints |
| Medium   | **New SKU Cold‐Start Module**                       | Ensures that brand‐new SKUs have a baseline forecast.                                | 3 sprints   |
| Medium   | **Explainability Engine**                           | Builds trust with domain experts; required for regulatory audits in some industries. | 2–3 sprints |
| Low      | **Scenario Planning UI**                            | Useful but can be built once core features are stable.                               | 4–6 sprints |
| Low      | **Quantile Regression / Probabilistic Forecasting** | More advanced; useful for safety stock but can wait until point + CI are stable.     | 4–6 sprints |
| Low      | **Outlier Root‐Cause Analyzer**                     | Automatically flag why an outlier occurred (e.g., store closure, data feed glitch).  | 5 sprints   |

---

## 11. Operational Considerations & SLAs

1. **Throughput & Latency**

   * **Online Inference**: Aim for < 200 ms per SKU on a c5.large (2 vCPU, 4 GB RAM) for M=13 horizon.
   * **Batch Forecast**: Scale horizontally—if you have 10,000 SKUs and each SKU inference takes 5 ms, total CPU time is 50 seconds; scale to 5 parallel pods to finish in < 15 seconds (plus overhead).
   * **Training**: If you have 5 years of weekly data (≈260 timesteps) and 50,000 SKUs, sliding window creation yields \~50,000 × (260−L−M) ≈ millions of samples. A single GPU can train in 2–3 hours depending on architecture. Consider distributed training if > 100K SKUs.

2. **Availability & SLAs**

   * **Inference API**: 99.9% uptime (two redundant ECS tasks in separate AZs, behind ALB with health checks).
   * **Batch Pipelines**: Designed to complete within 4 hours overnight. If it fails, send an alert immediately; a retry mechanism should automatically re‐run the failed state up to 2 times.
   * **Data Retention**: Keep raw data for 1 year, processed data for 2 years, full forecasts for 5 years. Automatically archive older data to Glacier to reduce costs.

3. **Cost Optimization**

   * **Spot Instances** for non‐urgent training (if tolerable risk of interruption).
   * **Compute Savings Plans** on EC2/EKS for consistent usage.
   * **Reserved Instances** for RDS.
   * **Serverless Athena & S3** to minimize idle cluster costs.

4. **Monitoring & Logging**

   * **CloudWatch Alarms**:

     * ETL failures (if Lambda or Batch job fails).
     * Training job errors or timeouts.
     * Inference latencies spiking above 500 ms.
   * **Dashboards**:

     * Real‐time metrics (requests/min on API, average latency, error rate).
     * Weekly accuracy metrics (MAPE, RMSE by category).
   * **Logs & Traces**:

     * Use X-Ray for distributed tracing of API calls (if needed).

---

## 12. Conclusion

This documentation outlines a robust, scalable, and extensible demand‐planning software suite that:

* **Meets all the stated requirements** (multi-format ingestion, cleaning, feature engineering, LSTM with dynamic categorical handling, dual‐mode loss, hyperparameter tuning, error storage, CI calculations).
* **Addresses gaps** by integrating holiday/promotional calendars, hierarchical reconciliation, drift detection, and fallback strategies for new SKUs.
* **Incorporates industry-standard features** (manual overrides, dashboards, explainability, scenario planning, robust security, and IAM).
* **Provides a fully detailed cloud‐native architecture** leveraging AWS services (S3, Lambda/Batch/EKS, Athena, Step Functions, SageMaker/EKS, ECS Fargate, API Gateway, RDS/DynamoDB, QuickSight, CloudWatch).

By following this design, your demand‐planning application will not only be technically sound but also aligned with best practices for model lifecycle management, security, user experience, and operational excellence.
