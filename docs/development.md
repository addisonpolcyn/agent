# Development

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync                                           # installs Python 3.14 + deps from uv.lock
uv run pre-commit install                         # optional: run the checks on every commit
```

## API key (only needed for Claude)

Everything with `--offline`, and the whole test suite, works without a key. To use Claude:

1. **Get a key.** In the [Claude Console](https://platform.claude.com), add credits under Billing, then create a key under Settings → API Keys. A claude.ai Pro or Max subscription does not include API access.
2. **Create `.env`** from the template and paste your key into it:

   ```bash
   cp .env.example .env
   chmod 600 .env                 # readable only by you
   # edit .env:  ANTHROPIC_API_KEY=sk-ant-...
   ```

   `.env` is git-ignored. Never commit it, and never paste a key into chats, issues or logs.
3. **Load it into your shell.** agentlab reads environment variables, not the `.env` file itself. Do this once per terminal:

   ```bash
   set -a; source .env; set +a
   ```

4. **Run without `--offline`:**

   ```bash
   uv run agentlab chat
   uv run agentlab flywheel
   AGENTLAB_LIVE_TESTS=1 uv run pytest -m live
   ```

If a key leaks, revoke it in the Console right away and put the new one in `.env`. The other settings in `.env.example` (`AGENTLAB_MODEL` and the directory overrides) are optional. Each live eval run costs a few cents with the default `claude-opus-5-5`.

## Commands

| Task | Command |
|---|---|
| Ask the agent (offline) | `uv run agentlab ask --offline "What is 123 * 456?"` |
| Ask the agent (Claude) | `uv run agentlab ask "What is 123 * 456?"` |
| Interactive session | `uv run agentlab chat [--offline]` (each question independent; `exit` or Ctrl-D to quit) |
| List discovered skills | `uv run agentlab skills` |
| Run evals (gate) | `uv run agentlab eval --offline` |
| Run a flywheel iteration | `uv run agentlab flywheel --offline` |
| Tests | `uv run pytest` (or `scripts/test.sh`) |
| Live Claude tests | `AGENTLAB_LIVE_TESTS=1 uv run pytest -m live` |
| Format, lint, types | `scripts/lint.sh` (ruff format --check, ruff check, pyright) |
| All pre-commit hooks | `uv run pre-commit run --all-files` |
| Eval and record iteration | `scripts/eval.sh` (offline) / `scripts/eval.sh --live` |

Paths default to `./skills`, `./evals/cases` and `./runs`, relative to the current directory, so run commands from the repo root. Override them with `AGENTLAB_SKILLS_DIR`, `AGENTLAB_CASES_DIR` and `AGENTLAB_RUNS_DIR`.

## The development loop

```text
change → uv run agentlab flywheel --offline → read failures.md → targeted fix → repeat
```

For anything that touches model behavior (prompts, tool descriptions, skill selection), also run the flywheel without `--offline`. See [evaluation.md](evaluation.md).

## Tests

| Directory | What it covers | Runs in CI |
|---|---|---|
| `tests/unit/` | Calculator, catalog, agent state transitions (`ScriptedLLM`), fakes, evals, flywheel, config, Claude mappings | Yes |
| `tests/integration/` | Full offline loop with `OfflineLLM` and the real catalog and cases; CLI via `main(argv)` | Yes |
| `tests/external/` | Real Claude API (`@pytest.mark.live`) | No: opt-in |

`pytest` deselects `live` tests by default (`-m "not live"` in `pyproject.toml`). The live tests also skip themselves unless `AGENTLAB_LIVE_TESTS=1` and a key are set.

## CI

`.github/workflows/ci.yml` runs, in order: `uv sync --locked`, the format check, lint, pyright, pytest, `eval --offline` and `flywheel --offline`. It uses no secrets and no network beyond installing dependencies.

## Tooling notes

- **Python 3.14** is pinned in `.python-version`. `uv` downloads it if needed.
- **Pyright** is installed as `pyright[nodejs]`, which bundles Node as a wheel, so it does not depend on a system Node. It runs in **strict** mode over `src/` and `tests/`.
- **Pre-commit hooks are all local** (`uv run ...`), so their versions come from `uv.lock` and they need no network.
- **Dependencies:** add runtime ones with `uv add <pkg>` and dev ones with `uv add --dev <pkg>`. Always commit `uv.lock`. Read the dependency policy in [style-guide.md](style-guide.md#dependencies) first.
