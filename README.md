# Nervex / SoulGraph

**Turn a deep-learning paper into verified, runnable model code, and prove it runs.**

Nervex reads a research paper, builds a knowledge graph of its architecture and equations, then runs a multi-agent loop that writes PyTorch/JAX code for a component, executes it inside a locked-down GPU sandbox, and feeds real tracebacks back until the forward pass works. On success you also get the exact tensor shapes at every layer.

> **Status: alpha.** 130+ unit tests (no GPU/Docker/API key needed) plus a live end-to-end run on *Attention Is All You Need*: ingest → knowledge graph → Multi-Head Attention generated, executed in the GPU sandbox on the first attempt, with per-layer tensor shapes. Expect rough edges. Issues and PRs welcome.

## How it works

```
PDF ─► OCR (Marker/Surya) ─► LaTeX masking ─► semantic chunks ─► LLM entity extraction
                                                   │                    │
                                              ChromaDB (vectors)    Neo4j (graph)
                                                   └──────────┬─────────┘
Researcher ─► Architect ─► Coder ─► Sandbox ──fail──► Critic ──► Coder   (max 5 attempts)
                                      │
                                    pass ─► code + per-layer tensor shapes
```

| Stage | What it does |
|---|---|
| Researcher | Pulls the component's neighbourhood from Neo4j and relevant passages from ChromaDB |
| Architect | Writes a class skeleton with explicit `# [B, S, D] -> ...` shape contracts |
| Coder | Implements the math; keeps `# Implements: <Node_ID: ...>` links back to the paper |
| Sandbox | Runs a dummy forward pass in an ephemeral container, captures layer shapes via forward hooks |
| Critic | Static analysis (AST import whitelist) + LLM diagnosis of real tracebacks |

Extras: **Explainer chat** (highlight generated code, get paper-grounded answers), **ablation** (mutate the architecture, re-run, compare), **PyTorch ⇄ JAX/Equinox transpilation**.

## Security model (the sandbox is the product)

AI-generated code is untrusted. Each run gets a fresh container with:
no network (`network_mode=none`), all Linux capabilities dropped, non-root user (1000), read-only code mount, 64 MB `/tmp` tmpfs, memory cap, hard wall-clock timeout with forced kill, and a static import whitelist before anything executes. The API binds to `127.0.0.1` by default; it can run code on your GPU, so do not expose it publicly without auth.

## Requirements

- Linux, Python 3.11+, [`uv`](https://docs.astral.sh/uv/)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) and an NVIDIA GPU (the harness runs models on CUDA)
- Neo4j 5 (one command below)
- An LLM: bring your own. Free NVIDIA NIM key ([build.nvidia.com](https://build.nvidia.com)), a Google AI Studio key (free tier), an OpenAI key, or a local model via Ollama / any OpenAI-compatible server (vLLM, LM Studio, llama.cpp). Set `LLM_PROVIDER` in `.env` (see below)
- ~8 GB VRAM. PDF OCR (Marker/Surya) starts its own short-lived vLLM container, so the **first ingestion of a paper takes several minutes** (cold start + OCR); re-ingesting the same file is served from a content-hash cache. The OCR container is torn down afterwards so the sandbox gets the GPU.

## Quickstart

```bash
git clone <this repo> && cd Nervex
uv sync --group dev
cp .env.example .env            # pick LLM_PROVIDER and add that provider's key

# 1. graph database
docker run -d --name nervex-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

# 2. sandbox image (one-time, several GB)
docker build -f sandbox/Dockerfile.sandbox -t soulgraph-runtime:latest .

# 3. run
uv run python main.py           # API on http://127.0.0.1:8000  (interactive docs at /docs)
```

Check `GET /` — it reports live status of Neo4j, Docker, the sandbox image and CUDA.

## Choosing your LLM

All agents get their model from one place (`agents/llm_factory.py`), driven by `.env`:

| `LLM_PROVIDER` | Needs | Default model |
|---|---|---|
| `nvidia` (default) | `NVIDIA_API_KEY` | nemotron-3-ultra-550b-a55b (thinking on) |
| `google` | `GOOGLE_API_KEY` | gemini-2.5-flash |
| `openai` | `OPENAI_API_KEY` | gpt-4o-mini |
| `openai_compatible` | `LLM_BASE_URL` (+ key if the server wants one) | `LLM_MODEL` |
| `ollama` | a running Ollama | qwen2.5-coder:7b |

Override the model with `LLM_MODEL`. Free tiers rate-limit hard: lower `EXTRACTION_CONCURRENCY` (2 is safe on Google's free tier); 429s trigger a shared cooldown and retries. Small local models will struggle with the structured entity extraction and with the Coder, so expect more Critic loops and failed chunks. `GET /` reports the active provider and model (never the key).

## API (`/api/v1`)

| Endpoint | Purpose |
|---|---|
| `POST /ingest/pdf` | Upload a paper → `202` + `job_id` (ingestion takes minutes) |
| `GET  /ingest/status/{job_id}` | Poll progress / result |
| `POST /graph/query` | Neighbourhood of a graph node (pass `paper_id` to scope it) |
| `POST /generate/code` | Run the full agent loop for a component |
| `POST /explainer/chat` | Ask about highlighted generated lines |
| `POST /features/ablation` | Mutate + re-run + compare (baseline is verified in the sandbox first) |
| `POST /features/transpile` | PyTorch ⇄ JAX/Equinox; pass `expected_inputs` to also execute the result in the sandbox |

## Tests

```bash
uv run pytest tests -q          # no GPU, Docker daemon, Neo4j or API key required
```

## Known limitations

- Needs an NVIDIA GPU; the harness calls `.cuda()`.
- Generated models must be constructible with **no arguments** (every `__init__` parameter needs a default); the Coder is told this, and violations produce an explicit error for the Critic.
- The harness feeds the Architect's dummy inputs: float by default, `bool` for names containing `mask`, `int64` for `*ids`/`tokens`. Anything more exotic needs an explicit dtype word in the shape (`"[4, 128] int64"`).
- Telemetry hooks `Linear`, `Conv2d`, `LayerNorm`, `MultiheadAttention`; every run also records the model's own input/output shapes under `__model__`.
- **Transpilation is execution-checked, not math-checked.** With `expected_inputs` the target code is run in the sandbox on batched dummy inputs (JAX on CPU, PyTorch on GPU) and failures are fed back to the LLM for up to 4 attempts. A pass proves it runs and returns finite numbers of the reported shape. It does **not** prove the math matches the source (e.g. a transposed weight on a square layer would pass). Without `expected_inputs` only static checks run and the response says so (`runtime_verified: null`).
- **Ablation regenerates the whole module**, so the LLM may change more than the one thing you asked.
- Graph retrieval quality is bounded by LLM entity extraction: formula links for a component can be noisy (e.g. complexity-table entries instead of the core equation), and graph nodes are global rather than scoped per paper (vectors are paper-scoped).
- Ingestion makes ~70 LLM calls per paper (default 4 in parallel; 429s trigger a shared cooldown and retries). On a rate-limited key expect 10-25 min; free tiers of any provider behave similarly. If more than 25% of chunks still fail, the job fails **without writing anything**, so a bad run never overwrites a good graph.
- Upstream LLM trouble surfaces as typed JSON errors: `429 rate_limited`, `504 timeout`, `502 upstream_error|upstream_auth|connection` (with `Retry-After` where relevant).
- The sandbox proves code *runs*, not that the math matches the paper.

## License

MIT, see [LICENSE](LICENSE).
