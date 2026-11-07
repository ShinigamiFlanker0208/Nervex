# Nervex / SoulGraph

**Turn a deep-learning paper into verified, runnable model code, and prove it runs.**

Nervex reads a research paper, builds a knowledge graph of its architecture and equations, then runs a multi-agent loop that writes PyTorch/JAX code for a component, executes it inside a locked-down GPU sandbox, and feeds real tracebacks back until the forward pass works. On success you also get the exact tensor shapes at every layer.

> **Status: alpha.** Components are unit-tested (74 tests, no GPU/Docker/API key needed) and the sandbox is verified on a real GPU container. Full paper-to-code runs are being validated; expect rough edges. Issues and PRs welcome.

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
- A free NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com)

## Quickstart

```bash
git clone <this repo> && cd Nervex
uv sync --group dev
cp .env.example .env            # add NVIDIA_API_KEY

# 1. graph database
docker run -d --name nervex-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

# 2. sandbox image (one-time, several GB)
docker build -f sandbox/Dockerfile.sandbox -t soulgraph-runtime:latest .

# 3. run
uv run python main.py           # API on http://127.0.0.1:8000  (interactive docs at /docs)
```

Check `GET /` — it reports live status of Neo4j, Docker, the sandbox image and CUDA.

## API (`/api/v1`)

| Endpoint | Purpose |
|---|---|
| `POST /ingest/pdf` | Upload a paper → `202` + `job_id` (ingestion takes minutes) |
| `GET  /ingest/status/{job_id}` | Poll progress / result |
| `POST /graph/query` | Neighbourhood of a graph node |
| `POST /generate/code` | Run the full agent loop for a component |
| `POST /explainer/chat` | Ask about highlighted generated lines |
| `POST /features/ablation` | Mutate + re-run + compare (baseline is verified in the sandbox first) |
| `POST /features/transpile` | PyTorch ⇄ JAX/Equinox |

## Tests

```bash
uv run pytest tests -q          # no GPU, Docker daemon, Neo4j or API key required
```

## Known limitations

- Needs an NVIDIA GPU; the harness calls `.cuda()`.
- The harness feeds one input named `x` and picks the first `nn.Module` it finds.
- Graph nodes are global (not yet scoped per paper); vectors are paper-scoped.
- LLM quality bounds output quality. The sandbox proves it *runs*, not that the math matches the paper.

## License

MIT, see [LICENSE](LICENSE).
