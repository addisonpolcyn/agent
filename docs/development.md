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
3. **Run without `--offline`.** agentlab and the live tests read `.env` from the current directory automatically, so run from the repo root:

   ```bash
   uv run agentlab chat
   uv run agentlab flywheel
   uv run pytest -m live
   ```

   A variable set in your shell overrides the same key in `.env`, e.g. `AGENTLAB_MODEL=claude-haiku-5-5 uv run agentlab chat`.

If a key leaks, revoke it in the Console right away and put the new one in `.env`. The other settings in `.env.example` (`AGENTLAB_MODEL` and the directory overrides) are optional. Each live eval run costs a few cents with the default `claude-opus-5-5`.

## Commands

| Task | Command |
|---|---|
| Ask the agent (offline) | `uv run agentlab ask --offline "What is 123 * 456?"` |
| Ask the agent (Claude) | `uv run agentlab ask "What is 123 * 456?"` |
| Interactive session | `uv run agentlab chat [--offline]` (the conversation is remembered; `reset` clears it, `exit` or Ctrl-D quits) |
| List discovered and learned skills | `uv run agentlab skills` (learned ones are tagged `[learned]`) |
| Turn off on-demand learning | `uv run agentlab ask --no-learn "..."` (also ignores learned skills) |
| Forget a learned skill | `rm -r .agentlab/learned/<name>` |
| Run evals (gate) | `uv run agentlab eval --offline` |
| Run a flywheel iteration | `uv run agentlab flywheel --offline` |
| Tests | `uv run pytest` (or `scripts/test.sh`) |
| Live Claude tests | `uv run pytest -m live` (needs a key in `.env` or the environment) |
| Format, lint, types | `scripts/lint.sh` (ruff format --check, ruff check, pyright) |
| All pre-commit hooks | `uv run pre-commit run --all-files` |
| Eval and record iteration | `scripts/eval.sh` (offline) / `scripts/eval.sh --live` |

Paths default to `./skills`, `./evals/cases`, `./runs` and `./.agentlab/learned`, relative to the current directory, so run commands from the repo root. Override them with `AGENTLAB_SKILLS_DIR`, `AGENTLAB_CASES_DIR`, `AGENTLAB_RUNS_DIR` and `AGENTLAB_LEARNED_DIR`. `.agentlab/` is local runtime state and git-ignored: learned skills are never committed.

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
| `tests/external/` | Real Claude API (`@pytest.mark.live`) | Yes, in the separate `live-claude` job |

A plain `pytest` deselects `live` tests (`-m "not live"` in `pyproject.toml`), so the everyday run stays offline and free. `pytest -m live` opts in. Live tests skip themselves when no key is configured, and CI checks for the key separately so a missing secret fails instead of skipping. Integration tests run from an empty temporary directory, so your real `.env` never leaks into them.

## Git workflow

Every change goes through a pull request that is **squash-merged once CI passes**:

```text
main ──► branch ──► commit ──► push ──► PR (draft) ──► more commits + pushes
                                                         │  CI runs on each push
                                                         ▼
                                     ready + auto-merge ──► CI green ──► squash into main
```

| Step | Command |
|---|---|
| Start | `git switch main && git pull --ff-only && git switch -c feat/short-name` |
| Commit and push (every time) | `git commit -m "..." && git push -u origin HEAD` |
| Open a PR at the first push | `gh pr create --draft --fill` |
| Finish | `gh pr ready && gh pr merge --auto --squash` |
| Check CI | `gh pr checks --watch` |

Why it works this way:

- **Small branches, one logical change each.** Squash merging turns each PR into one commit on `main`, so `main`'s history reads as a list of changes. The PR title becomes that commit's message.
- **Push and open a PR early.** CI runs on every push, so problems show up while they're cheap to fix. The draft state shows the work isn't finished.
- **CI is the merge gate, not memory.** Both CI jobs must pass: `check` (offline, deterministic) and `live-claude` (real Claude). Model output varies, so if `live-claude` fails, read the failure first. If it's a phrasing flake rather than a real regression, re-run it (`gh run rerun --failed`) and consider making the eval check more robust.

One-time setup: `gh auth login`. Branch protection on `main` requires a pull request with passing `check` and `live-claude` jobs, allows squash merges only, enables auto-merge, and deletes branches after merge.

## CI

`.github/workflows/ci.yml` has two jobs, and both are required to merge:

| Job | Runs | Needs |
|---|---|---|
| `check` | `uv sync --locked`, format check, lint, pyright, pytest, `eval --offline`, `flywheel --offline` | Nothing: offline and deterministic |
| `live-claude` | `pytest -m live` and `agentlab eval` against real Claude, only after `check` passes | The `ANTHROPIC_API_KEY` repository secret |

**Setting up the secret.** Use a separate key just for CI, with a monthly spend limit set in the Console. Add it under GitHub **Settings → Secrets and variables → Actions → New repository secret**, named `ANTHROPIC_API_KEY`, or run `gh secret set ANTHROPIC_API_KEY`. Each `live-claude` run costs a few cents.

**Forks.** GitHub never gives secrets to pull requests from forks, so `live-claude` is skipped for them. Never switch the trigger to `pull_request_target` to work around this: that would expose the key to untrusted code.

## Tooling notes

- **Python 3.14** is pinned in `.python-version`. `uv` downloads it if needed.
- **Pyright** is installed as `pyright[nodejs]`, which bundles Node as a wheel, so it does not depend on a system Node. It runs in **strict** mode over `src/` and `tests/`.
- **Pre-commit hooks are all local** (`uv run ...`), so their versions come from `uv.lock` and they need no network.
- **Dependencies:** add runtime ones with `uv add <pkg>` and dev ones with `uv add --dev <pkg>`. Always commit `uv.lock`. Read the dependency policy in [style-guide.md](style-guide.md#dependencies) first.
