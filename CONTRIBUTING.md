# Contributing to Nervex

- Python 3.11+, managed with [`uv`](https://docs.astral.sh/uv/). `uv sync --group dev` then `uv run pytest tests -q`.
- A backend function without a pytest test doesn't exist yet. Unit tests must not need a GPU, Docker, Neo4j or an API key (inject fakes).
- Sandbox changes keep the least-privilege posture: no network, `cap_drop=ALL`, non-root, read-only workspace, memory cap, hard timeout. Weakening any of these needs an explicit justification in the PR.
- Prefer deterministic Python over an LLM call wherever the logic can be pure code.
- Keep PRs small and say *why*, not just what.
