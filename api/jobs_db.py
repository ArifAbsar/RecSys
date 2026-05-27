import os
import json
from api.state import OUTPUT_DIR

##stored in json file for now
JOBS = {}

def get_job_state_path(job_id: str) -> str:
    """Returns absolute path of the persisted JSON state file."""
    jobs_dir = os.path.join(OUTPUT_DIR, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    return os.path.join(jobs_dir, f"{job_id}.json")

def save_job_state(job_id: str):
    """Saves the current job state to disk as JSON."""
    if job_id not in JOBS:
        return
    try:
        path = get_job_state_path(job_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(JOBS[job_id], f, indent=2)
    except Exception as e:
        print(f"[Jobs DB] Error saving state for {job_id}: {e}", flush=True)

def load_job_state(job_id: str) -> bool:
    """Loads job state from disk into the active memory cache."""
    global JOBS
    path = get_job_state_path(job_id)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            JOBS[job_id] = json.load(f)
        return True
    except Exception as e:
        print(f"[Jobs DB] Error loading state for {job_id}: {e}", flush=True)
        return False
