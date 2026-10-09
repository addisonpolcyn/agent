# agentlab

## Glossary

Use these names in code. Details, data model and lifecycle rules: [docs/workflows.md](docs/workflows.md).

| Name | What it is |
|---|---|
| Tool | A validated, versioned Python function with typed input and output schemas |
| Workflow | A DAG of steps (tool calls and LLM reasoning steps) that answers a type of request; parameterized (planned, not built yet) |

Avoid in new identifiers: **task** (collides with the user's request), **chain** (LangChain baggage), **skill** (collides with Claude's Skills, which are instructions, not code), **agent** (reserved for the whole system).

## Overview

A small, evaluation-driven agent that discovers and uses **tools** to solve tasks, and improves through an eval → failure → fix flywheel. The north star is a flight-research agent. The project is at **Stage 0 (bootstrap)**: an offline agent loop, one deterministic tool (`calculator`), capability-gap reporting, on-demand tool learning (user-approved, kept in the git-ignored `.agentlab/learned/`; learned tools run **unrestricted** for now, since this is a single-user toy), evals and the flywheel. This file is a **map**: the substance lives in the linked docs.

## Commands

```bash
uv sync                                              # set up
uv run agentlab ask --offline "What is 123 * 456?"   # run the agent (drop --offline to use Claude)
uv run agentlab chat --offline                       # interactive: ask questions until 'exit'
uv run agentlab eval --offline                       # eval gate (non-zero exit on failure)
uv run agentlab flywheel --offline                   # record an iteration + failure summary in runs/
uv run pytest && scripts/lint.sh                     # tests, format, lint, pyright strict
uv run pytest -m live                                # real Claude tests (key from .env)
```

## Docs

| Doc | Read it when… |
|---|---|
| [docs/style-guide.md](docs/style-guide.md) | **Before writing any code.** Methodologies and the reasons behind them. |
| [docs/architecture.md](docs/architecture.md) | You need the system flow, the reason for each boundary, or the security principles. |
| [docs/workflows.md](docs/workflows.md) | You need the Tool vs. Workflow split, the planned data model, or the create/split/evict rules. |
| [docs/tools.md](docs/tools.md) | You are adding or changing a tool, or touching discovery. |
| [docs/evaluation.md](docs/evaluation.md) | You are writing eval cases, adding checks, or reading flywheel output. |
| [docs/development.md](docs/development.md) | You need setup, the API key and `.env`, commands, the test tiers, CI or tooling notes. |
| [docs/roadmap.md](docs/roadmap.md) | You are deciding what to build next, or checking that a change serves the north star. |

## Code

| Path | What's there |
|---|---|
| [src/agentlab/agent/loop.py](src/agentlab/agent/loop.py) | The agent loop, `request_capability`, `AgentRun` trace. Start here. |
| [src/agentlab/models.py](src/agentlab/models.py) | Provider-neutral boundary types (`ToolSpec`, `ToolCall`, messages). |
| [src/agentlab/llm/](src/agentlab/llm/) | `LLMClient` protocol, `ClaudeClient` (only SDK importer), `ScriptedLLM` / `OfflineLLM` fakes. |
| [src/agentlab/tools/catalog.py](src/agentlab/tools/catalog.py) | Tool discovery and execution. |
| [src/agentlab/learning/](src/agentlab/learning/) | On-demand learning: plan rules, the approval gate, author, runtime eval harness, tool process (`sandbox.py`), local store, starter tools shipped into it (`starter/`, e.g. `run_command`). |
| [tools/](tools/) | Tool manifests (`*/tool.toml`). |
| [src/agentlab/evals/](src/agentlab/evals/) | Case loading, checks, runner. |
| [evals/cases/](evals/cases/) | Eval cases (TOML), including the `flight_sfo_tokyo` north-star fixture. |
| [src/agentlab/flywheel/loop.py](src/agentlab/flywheel/loop.py) | Records runs and diffs them against the previous one. |
| [src/agentlab/cli.py](src/agentlab/cli.py) | CLI wiring; settings are read here only. |
| [tests/](tests/) | `unit/`, `integration/`, `external/` (opt-in live). |

## Git workflow (always)

`main` is protected: every change lands as a squash-merged PR after CI passes. Details are in [docs/development.md](docs/development.md#git-workflow).

1. **Branch off an up-to-date `main`.** Never commit to `main` directly.
   `git switch main && git pull --ff-only && git switch -c <type>/<short-name>`
   (types: `feat`, `fix`, `docs`, `eval`, `tool`, `chore`)
2. **After every commit, push.** `git push -u origin HEAD`
3. **At the first push, open a PR** so CI runs on every commit. Use a draft while the work is in progress:
   `gh pr create --draft --fill` (or without `--draft` if the work is already complete)
4. **When the work is done**, check that tests, lint and evals pass locally, then:
   `gh pr ready && gh pr merge --auto --squash`
   This squash-merges the PR automatically once CI passes. The PR title becomes the commit on `main`, so make it a clear imperative summary.
5. Don't push to `main`, force-push shared branches, or merge with failing CI.

## Non-negotiables (details in the style guide)

- Every behavior change ships with an eval case. Never weaken a check to make it pass.
- Never route on task text. The model selects tools from their manifests.
- Only `llm/claude.py` imports `anthropic`. The loop never knows how tools work.
- Generated tool code runs only after the user approves the plan (one approval, no code review gate), in its own process, and is never committed. It is unrestricted (any import, files, network, commands) while agentlab is a single-user toy; restore limits before anyone else uses it ([roadmap](docs/roadmap.md)).
- The default test run and the `check` CI job stay offline. Real Claude runs only in `pytest -m live` and the `live-claude` job. Fakes are labeled as fakes, and fabricated data is never presented as real.
- When documented behavior changes, update the doc and this map in the same change.
