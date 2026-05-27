# API Production Readiness & Optimization TODO List

This document lists architectural concerns, performance bottlenecks, security vulnerabilities, and code safety issues identified in the local web service, categorized by severity, with recommended actions for a production-grade deployment.

---

## 1. Critical Security Vulnerabilities

### 1.1 Path Traversal via Unsanitized Upload Filenames
* **File affected**: [api/routes/jobs.py](file:///f:/RecSys/api/routes/jobs.py) (`_ingest_csv` / `upload_csv_and_create_job`)
* **The Issue**: 
  The raw filename provided by the client (`file.filename`) is interpolated directly into path structures:
  `saved_path = os.path.join(uploads_dir, f"{job_id}_{filename}")`
  A malicious client could pass a filename like `../../api_service.py` to overwrite core app files or write files to arbitrary directories on the host machine.
* **Todo**: Sanitise the filename using `werkzeug.utils.secure_filename` before combining paths.
  ```python
  from werkzeug.utils import secure_filename
  filename = secure_filename(file.filename)
  ```

### 1.2 Arbitrary File Uploads (Denial of Service & Remote Execution)
* **File affected**: [api/routes/jobs.py](file:///f:/RecSys/api/routes/jobs.py) and [api/routes/orchestrator.py](file:///f:/RecSys/api/routes/orchestrator.py)
* **The Issue**:
  There are no constraints on file sizes (`MAX_CONTENT_LENGTH`) or file extensions. A user could upload a 20GB file to exhaust disk space (DoS) or upload a `.py` script to attempt execution.
* **Todo**:
  * Set `app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024` (e.g., limit to 16MB).
  * Enforce extension checking (allow only `.csv` and `.txt`).

---

## 2. Concurrency & Threading Issues

### 2.1 Thread-Unsafe Global Redirect of `sys.stdout` (Logs Pollution)
* **File affected**: [api/pipeline.py](file:///f:/RecSys/api/pipeline.py#L17-L34) (`capture_stdout`)
* **The Issue**:
  The context manager hijacks `sys.stdout` globally: `sys.stdout = stream`. 
  Because `sys.stdout` is a process-wide global variable, if two background mapping threads run concurrently, they will overwrite each other's stdout stream. All print logs from Job A will be routed to Job B's logs list, resulting in complete log pollution and leaks between job sessions.
* **Todo**:
  Do not hijack `sys.stdout`. Use a thread-local logger, Python's standard `logging` framework, or pass a dedicated `log_callback(msg)` handler to the processing engines.

### 2.2 Global State Thread Safety
* **File affected**: [api/jobs_db.py](file:///f:/RecSys/api/jobs_db.py)
* **The Issue**:
  `JOBS` is a global in-memory dictionary. Read and write operations (`save_job_state`, `load_job_state`) are done without threading locks. Under concurrent client requests, Flask's multi-threaded WSGI server can throw race condition errors or corrupt JSON states.
* **Todo**:
  Integrate a thread lock wrapper for all mutations on `JOBS`:
  ```python
  import threading
  db_lock = threading.Lock()
  
  with db_lock:
      JOBS[job_id] = data
  ```

---

## 3. Memory & Resource Management

### 3.1 Memory Leak on In-Memory Job Cache
* **File affected**: [api/jobs_db.py](file:///f:/RecSys/api/jobs_db.py)
* **The Issue**:
  The global `JOBS` dictionary grows indefinitely as more jobs are created or queried. It acts as an unbounded cache, causing the server to gradually consume all available RAM over time.
* **Todo**:
  * Offload state management entirely to an external database (e.g., SQLite, PostgreSQL, Redis).
  * If keeping an in-memory cache, implement cache eviction (e.g., using `collections.OrderedDict` as an LRU cache or clearing completed records after a TTL).

### 3.2 Lack of Temp File Cleanup
* **File affected**: [api/routes/jobs.py](file:///f:/RecSys/api/routes/jobs.py) (Ingested CSV uploads folder)
* **The Issue**:
  Uploaded files are written permanently to `output/uploads/` and job states to `output/jobs/`. There is no background worker or expiration policy to purge old jobs or delete uploaded source CSV files, which will eventually exhaust host disk space.
* **Todo**:
  Implement a cleanup routine (e.g., cron job or background thread) that checks file creation dates and deletes uploaded files older than 24 hours.

---

## 4. Performance Bottlenecks

### 4.1 Frequent Disk I/O inside Column Loop
* **File affected**: [api/pipeline.py](file:///f:/RecSys/api/pipeline.py#L56-L69)
* **The Issue**:
  `save_job_state` is executed on every iteration of the column-mapping loop. If a dataset has 80 columns, it performs 80 full serialization and file-write operations to disk.
* **Todo**:
  Save to disk less frequently (e.g., every 5 columns, or only at major stage changes), or use a memory store (like Redis) for fast real-time progress updates.

### 4.2 Dynamic Schema Calculations on Every Request
* **File affected**: [api/routes/health.py](file:///f:/RecSys/api/routes/health.py) (`get_schema_details`)
* **The Issue**:
  The `/api/schema-details` endpoint iterates over all 251 target fields in `TARGET_REGISTRY` and groups them by entity on every single GET request.
* **Todo**:
  Perform this grouping once during system startup (after models finish warming up), cache the resulting dictionary, and return the cached object instantly.

---

## 5. API Design Consistency

### 5.1 Inconsistent Lock Error Codes
* **Files affected**: [api/routes/jobs.py](file:///f:/RecSys/api/routes/jobs.py) and [api/routes/orchestrator.py](file:///f:/RecSys/api/routes/orchestrator.py)
* **The Issue**:
  If a deployment lock is active, the orchestrator handles `LockFileExistsError` explicitly and returns `423 Locked`. The standard `/deploy` route does not catch it specifically, letting it drop to the generic `Exception` block, returning a `500 Internal Server Error`.
* **Todo**:
  Unify lock exception handling in `/api/jobs/<job_id>/deploy` to return `423 Locked`.

### 5.2 Hardcoded Models and Constants
* **File affected**: [api/state.py](file:///f:/RecSys/api/state.py)
* **The Issue**:
  HuggingFace models (`all-MiniLM-L6-v2`, `cross-encoder/stsb-distilroberta-base`) are hardcoded. Changing model assets requires modifying core files.
* **Todo**:
  Move these models, ports, and folder paths to environment variables (e.g., `EMBEDDING_MODEL_NAME`) or a central configuration file.
