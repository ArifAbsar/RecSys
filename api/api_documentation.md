## Architecture Overview

All APIs are modularly split into blueprints registered under the main Flask application factory:

```
f:\RecSys\api\
├── __init__.py          # App initialization & model warmup triggers
├── state.py             # Global models and dynamic configurations
├── jobs_db.py           # In-memory and disk persistence handlers
├── pipeline.py          # Core headless execution mapping pipeline
└── routes\
    ├── health.py        # System health & schema options
    ├── jobs.py          # Job lifecycle CRUD & deployments
    └── orchestrator.py  # Single-call automated runner
```

---

## 1. System & Schema Health Endpoints

### 1.1 GET `/api/health`
Checks service readiness. Since HuggingFace model warmup is CPU/memory intensive, models are loaded asynchronously in a background thread upon startup so the port remains active.

* **Method**: `GET`
* **Headers**: `None`
* **Responses**:
  * **200 OK** (Models loaded successfully):
    ```json
    {
      "status": "ok",
      "registry_size": 251
    }
    ```
  * **503 Service Unavailable** (Models are still loading):
    ```json
    {
      "status": "loading",
      "step": "Step 3/4: Encoding 251 target fields (CPU-heavy, 1-3 min)..."
    }
    ```

---

### 1.2 GET `/api/schema-details`
Exposes the complete list of target entities and fields parsed from `ecommerce/schema.yaml` for client-side dropdown mapping options.

* **Method**: `GET`
* **Headers**: `None`
* **Responses**:
  * **200 OK**:
    ```json
    {
      "CUSTOMER": ["customer_id", "email", "phone", "loyalty_id"],
      "PRODUCT": ["sku", "title", "price", "category"],
      "IGNORED": ["Unmapped"]
    }
    ```
  * **503 Service Unavailable** (If models are still warming up).

---

## 2. Job Lifecycle Endpoints

### 2.1 POST `/api/jobs`
Accepts a dataset to create a new mapping job. It parses column headers, infers high-level datatypes (`string`, `number`, `datetime`), captures samples, and initializes the state.

* **Method**: `POST`
* **Headers**:
  * For JSON: `Content-Type: application/json`
  * For File Upload: `Content-Type: multipart/form-data`
* **Request Body Options**:
  * **Option A (JSON Payload - local path)**:
    ```json
    {
      "file_path": "F:\\RecSys\\data\\steam_with_headers.csv"
    }
    ```
  * **Option B (Multipart form-data - file upload)**:
    * Key: `file` (File type) -> Select `steam_with_headers.csv`
* **Under the Hood**:
  * If a file is uploaded, it is physically written to `output/uploads/{job_id}_{filename}`.
  * Injects job metadata into the active memory cache and persists a JSON tracking state at `output/jobs/{job_id}.json`.
* **Responses**:
  * **200 OK**:
    ```json
    {
      "job_id": "c1f7da39-d8e2-45e0-b918-a68fb11082bb",
      "status": "created",
      "columns": [
        {
          "name": "user_id",
          "data_type": "number",
          "samples": ["151603712"]
        }
      ]
    }
    ```
  * **400 Bad Request**: If CSV parsing fails or parameters are missing.

---

### 2.2 POST `/api/jobs/<job_id>/start`
Launches the headless mapping pipeline in a non-blocking background thread for the specified job.

* **Method**: `POST`
* **Request URL Params**: `job_id` (string, UUID)
* **Body**: `None`
* **Responses**:
  * **200 OK**:
    ```json
    {
      "status": "running",
      "message": "Job launched in background thread.",
      "job_id": "c1f7da39-d8e2-45e0-b918-a68fb11082bb"
    }
    ```
  * **409 Conflict**: If the job is already running.
  * **404 Not Found**: If the job ID does not exist.

---

### 2.3 GET `/api/jobs/<job_id>`
Retrieves the status, real-time progress percentages, processing log streams, and current mapping results.

* **Method**: `GET`
* **Request URL Params**: `job_id` (string, UUID)
* **Responses**:
  * **200 OK**:
    ```json
    {
      "job_id": "c1f7da39-d8e2-45e0-b918-a68fb11082bb",
      "filename": "steam_with_headers.csv",
      "file_path": "F:\\RecSys\\output\\uploads\\c1f7da39-d8e2-45e0-b918-a68fb11082bb_steam_with_headers.csv",
      "status": "completed",
      "progress": {
        "percentage": 100,
        "message": "Headless schema mapping completed successfully."
      },
      "summary": {
        "total_columns": 5,
        "mapped_matches": 1,
        "ignored_unmapped": 0,
        "verification_warnings": 4
      },
      "mappings": [
        {
          "source_column": "user_id",
          "sample_record": "151603712",
          "detected_entity_node": "CUSTOMER",
          "mapped_canonical_attribute": "customer_id",
          "role_designation": "Identity",
          "confidence_score": 84.4,
          "alignment_status": "Mapped",
          "review_metadata": {
            "flagged": false,
            "reasons": [],
            "business_context_prompt": "What is the best stable customer key..."
          }
        }
      ],
      "logs": [
        "[13:20:00] Job created...",
        "[13:20:05] Starting headless schema integration mapping..."
      ]
    }
    ```

---

### 2.4 POST `/api/jobs/<job_id>/update`
Allows administrators or client frontends to submit manual overrides for mapping predictions (human-in-the-loop).

* **Method**: `POST`
* **Headers**: `Content-Type: application/json`
* **Request Body**:
  ```json
  {
    "mappings": [
      {
        "source_column": "other",
        "detected_entity_node": "PRODUCT",
        "mapped_canonical_attribute": "sku",
        "role_designation": "Content Item",
        "alignment_status": "Mapped"
      }
    ]
  }
  ```
* **Under the Hood**:
  * Syncs the target overrides with the active job state.
  * Resets confidence to `"Manual"`.
  * Recalculates metrics summary and rewrites `output/jobs/{job_id}.json`.
* **Responses**:
  * **200 OK**:
    ```json
    {
      "status": "updated",
      "job_id": "c1f7da39-d8e2-45e0-b918-a68fb11082bb"
    }
    ```

---

### 2.5 POST `/api/jobs/<job_id>/deploy`
Deploys the finalized configuration structure to the production environment file `output/rec_config.json`.

* **Method**: `POST`
* **Responses**:
  * **200 OK**:
    ```json
    {
      "status": "deployed",
      "config_path": "F:\\RecSys\\output\\rec_config.json",
      "job_id": "c1f7da39-d8e2-45e0-b918-a68fb11082bb"
    }
    ```
  * **400 Bad Request**: Mappings are not calculated yet.
  * **423 Locked**: If a feature engineering lock file prevents writing configurations.

---

## 3. Synchronous Orchestrator Endpoint

### 3.1 POST `/api/jobs/run-all`
A single-call endpoint that executes all steps synchronously (create, map, and deploy). Useful for CI/CD pipelines, automated testing, or cron triggers where polling is not preferred.

* **Method**: `POST`
* **Request Body Options**: Accepts either JSON `file_path` or standard multipart `file` upload.
* **Under the Hood**:
  1. Spawns the job.
  2. Runs `run_headless_mapping_pipeline()` synchronously (blocking the connection).
  3. Formats BML properties and deploys straight to `output/rec_config.json`.
* **Responses**:
  * **200 OK**:
    ```json
    {
      "job_id": "c1f7da39-d8e2-45e0-b918-a68fb11082bb",
      "status": "deployed",
      "config_path": "F:\\RecSys\\output\\rec_config.json",
      "filename": "steam_with_headers.csv",
      "summary": {
        "total_columns": 5,
        "mapped_matches": 1,
        "ignored_unmapped": 0,
        "verification_warnings": 4
      },
      "mappings": [ ... ],
      "config": {
        "schema_version": "1.0",
        "identity": ["game_name", "user_id"],
        "content_features": ["other", "value"],
        ...
      }
    }
    ```

---

## Processing Engine Details (Under the Hood)

During pipeline execution (`pipeline.py`), the following sequence of scripts and libraries is executed:

1. **Embedding Matcher (`Mapping_Engine/embedding_matcher/embedding.py`)**: 
   * Encodes source columns and target registry variables using the warmed-up **SentenceTransformer (`all-MiniLM-L6-v2`)**.
   * Computes cosine similarity values between datasets.
2. **Cross-Encoder Reranker (`Mapping_Engine/Reranking/Reranker.py`)**:
   * Takes the top-N candidates and reranks them using **`cross-encoder/stsb-distilroberta-base`**.
   * Integrates user search-intent parameters parsed from `keywords.yaml` using a keyword search index.
3. **1:1 Constraint Solver (`Mapping_Engine/Reranking/Resolver.py`)**:
   * Resolves mapping choices globally using stable match algorithms to avoid duplicates.
4. **Business Meaning Layer (`Business_Meaning/business_layer.py`)**:
   * Evaluates each column's business role using rule classifications in `classifier_rules.yaml`.
   * Evaluates metric weights utilizing `Business_Meaning/weight_assigner.py`.
   * Runs the ambiguity checks (`Business_Meaning/ambiguity_detector.py`) and evaluates the `questions.yaml` confidence thresholds.
   * **Auto-Accept Override**: Overrides BML review requests if mapping confidence is $\ge 80\%$, forcing the status to `"Mapped"`.
5. **Config Writer (`Business_Meaning/config_writer.py`)**:
   * serializes final structures to `rec_config.json`.
