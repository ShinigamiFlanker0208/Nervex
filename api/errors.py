"""Map upstream LLM / infrastructure failures to typed JSON errors instead of bare 500s."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agents.llm_retry import classify

_MESSAGES = {
    "timeout": "The language-model provider timed out. Retry in a moment.",
    "rate_limited": "The language-model provider is rate limiting this key. Wait and retry, or lower concurrency.",
    "upstream_error": "The language-model provider returned a server error.",
    "upstream_auth": "The language-model provider rejected the API key (check the API key for your LLM_PROVIDER).",
    "connection": "Could not reach the language-model provider.",
}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def upstream_or_500(request: Request, exc: Exception):
        status, kind = classify(exc)
        if status is None:
            # Not an upstream failure: let it be a real 500, but still JSON (UIs parse JSON).
            return JSONResponse(status_code=500, content={"error": "internal_error", "detail": type(exc).__name__})
        headers = {"Retry-After": "30"} if status in (429, 504) else None
        return JSONResponse(status_code=status, headers=headers, content={
            "error": kind, "detail": _MESSAGES[kind], "upstream": f"{type(exc).__name__}"})
