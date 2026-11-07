"""
Isolated OCR execution (Marker 2 / Surya 0.22).

Why this exists: Surya auto-spawns a vLLM *docker container* as its OCR engine and only tears it down
when the spawning PROCESS exits. Run inside the long-lived API process, that container (and its VRAM)
stays alive forever, so the next /generate/code sandbox has no GPU memory left. Running OCR in a short-lived
child process ties the server's lifetime to one ingestion job: child exits -> atexit stops the container
-> VRAM is free for the sandbox. Trade-off: ~3 min cold start per paper. It also contains Marker's
multi-GB model memory and any native crash to the child instead of the API.

Catastrophes prevented:
  * API process holds 6+ GB VRAM forever          -> child process owns it.
  * OCR hangs forever (stuck server / lock)        -> hard timeout, child is killed.
  * API crash leaves an orphan GPU container       -> `remove_stale_ocr_servers()` on startup/shutdown.
"""
import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from config import settings


def _child(conn: Any, pdf_path: str, use_cuda: bool, device_id: int, gpu_fraction: float) -> None:
    """Runs in the spawned process. Env must be set BEFORE marker/surya are imported."""
    try:
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(gpu_fraction)
        from ingestion.pdf_parser import PDFIngestionEngine

        engine = PDFIngestionEngine(use_cuda=use_cuda, device_id=device_id)
        result = engine.parse_pdf(Path(pdf_path))
        # Keep only picklable, small-ish essentials.
        conn.send(("ok", {
            "raw_markdown": result["raw_markdown"],
            "metadata": result.get("metadata"),
            "tables": result.get("tables", []),
            "images": {},   # figure bitmaps are not needed downstream; avoid pickling large PIL images
        }))
    except BaseException as exc:  # incl. SystemExit from native libs
        conn.send(("err", f"{type(exc).__name__}: {exc}", traceback.format_exc()))
    finally:
        conn.close()


def run_ocr_isolated(
    pdf_path: Path,
    *,
    use_cuda: bool = True,
    device_id: int = 0,
    gpu_fraction: Optional[float] = None,
    timeout: Optional[float] = None,
    target: Callable[..., None] = _child,
    mp_context: str = "spawn",          # spawn, not fork: forking after CUDA init is undefined behaviour
) -> Dict[str, Any]:
    gpu_fraction = settings.OCR_GPU_MEMORY_UTILIZATION if gpu_fraction is None else gpu_fraction
    timeout = settings.OCR_TIMEOUT_SECONDS if timeout is None else timeout

    ctx = mp.get_context(mp_context)
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(child, str(pdf_path), use_cuda, device_id, gpu_fraction), daemon=False)
    proc.start()
    child.close()
    try:
        # Receive BEFORE join: a large payload fills the pipe buffer and the child would block in send().
        if not parent.poll(timeout):
            raise TimeoutError(f"OCR exceeded {timeout}s and was killed")
        try:
            msg = parent.recv()
        except EOFError:
            proc.join(5)
            raise RuntimeError(f"OCR process died without a result (exit code {proc.exitcode})")
        if msg[0] == "err":
            raise RuntimeError(f"OCR failed: {msg[1]}\n{msg[2]}")
        return msg[1]
    finally:
        if proc.is_alive():
            proc.terminate()                # SIGTERM -> atexit runs -> Surya stops its container
            proc.join(30)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
        else:
            proc.join(5)
        parent.close()
        remove_stale_ocr_servers()           # belt and braces if terminate() skipped atexit


class IsolatedPDFParser:
    """Drop-in for PDFIngestionEngine.parse_pdf, safe to cache in the API process (holds no GPU state)."""

    def __init__(self, use_cuda: bool = True, device_id: int = 0):
        self.use_cuda, self.device_id = use_cuda, device_id

    def parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        # Content-addressed cache: same bytes -> same markdown, so skip the multi-minute OCR.
        from ingestion.pipeline import compute_paper_id
        cache = settings.OCR_CACHE_DIR / f"{compute_paper_id(pdf_path)}.json"
        if cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache.unlink(missing_ok=True)       # corrupt cache entry: recompute, never crash
        result = run_ocr_isolated(pdf_path, use_cuda=self.use_cuda, device_id=self.device_id)
        if result.get("raw_markdown", "").strip():  # never cache an empty/failed parse
            settings.OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, default=str), encoding="utf-8")
            tmp.replace(cache)                       # atomic: a crash can't leave a half-written cache
        return result


def remove_stale_ocr_servers() -> int:
    """Force-remove Surya's vLLM containers (named `surya-vllm-<port>`). Returns how many were removed."""
    try:
        import docker
        client = docker.from_env()
        removed = 0
        for c in client.containers.list(all=True, filters={"name": "surya-vllm-"}):
            c.remove(force=True)
            removed += 1
        return removed
    except Exception:
        return 0   # best effort: never let cleanup mask the real result/error
