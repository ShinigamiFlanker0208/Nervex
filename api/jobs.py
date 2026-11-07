"""
In-memory job registry for long-running ingestion.

Why: OCR + per-chunk LLM extraction takes minutes; an HTTP request must not hang that long.
`POST /ingest/pdf` returns a job_id immediately and the client polls `GET /ingest/status/{id}`.
Deliberately in-memory (single-process, $0 infra): jobs are lost on restart, which is fine because
ingestion is idempotent (content-hash paper_id + MERGE/upsert) so the user can simply re-upload.
"""
import threading
import time
import uuid
from typing import Any, Dict, Optional


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "filename": filename,
                "state": "queued",          # queued | running | done | failed
                "progress": [],
                "result": None,
                "error": None,
                "created_at": time.time(),
            }
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def log(self, job_id: str, message: str) -> None:
        with self._lock:
            self._jobs[job_id]["progress"].append(message)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else {**job, "progress": list(job["progress"])}


jobs = JobStore()
