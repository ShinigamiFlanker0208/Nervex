"""
Retry + classification for flaky upstream LLM calls (NVIDIA NIM).

Why: the provider SDK raises plain `Exception("[429] Too Many Requests")` and aiohttp timeouts. Without a
retry, one slow response from a 550B thinking model turns into an HTTP 500 for the user (seen live on
/features/transpile). Without classification, the API cannot tell the client "rate limited, retry later"
from "your request is wrong".
"""
import asyncio
import random
import re
from typing import Any, Optional, Tuple

# Providers format the status differently: NVIDIA "[429] ...", OpenAI "Error code: 429 - ...",
# Google "429 RESOURCE_EXHAUSTED ...".
_STATUS = re.compile(r"\[(\d{3})\]|Error code: (\d{3})|^\s*(\d{3}) [A-Z_]{4,}")


def classify(exc: BaseException) -> Tuple[Optional[int], str]:
    """Return (http_status_to_use, kind) for upstream failures, or (None, '') if it is not one of them."""
    name = type(exc).__name__
    msg = str(exc)
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "Timeout" in name or "timed out" in msg.lower():
        return 504, "timeout"
    low = msg.lower()
    if "resource_exhausted" in low or "quota" in low and "exceed" in low:
        return 429, "rate_limited"
    if "api key not valid" in low or "permission_denied" in low or "unauthenticated" in low:
        return 502, "upstream_auth"
    m = _STATUS.search(msg)
    if m:
        code = int(next(g for g in m.groups() if g))
        if code == 429:
            return 429, "rate_limited"
        if code >= 500:
            return 502, "upstream_error"
        if code in (401, 403):
            return 502, "upstream_auth"          # our key is bad: not the caller's fault
        return None, ""
    if isinstance(exc, ConnectionError) or "ClientConnector" in name or "ClientError" in name or "ClientOSError" in name:
        return 502, "connection"
    if "too many requests" in msg.lower():
        return 429, "rate_limited"
    return None, ""


def is_transient(exc: BaseException) -> bool:
    status, kind = classify(exc)
    return kind in {"timeout", "rate_limited", "upstream_error", "connection"}


async def ainvoke_with_retry(llm: Any, messages: Any, attempts: int = 3, base_delay: float = 4.0) -> Any:
    """`await llm.ainvoke(messages)` retrying transient failures with jittered exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            await asyncio.sleep(base_delay * 2 ** (attempt - 1) * (0.75 + random.random() / 2))
