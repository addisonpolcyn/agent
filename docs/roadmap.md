# Roadmap

The north star is a **flight research and recommendation agent**:

> "Find the best flights from SFO to Tokyo in November. I prefer nonstop flights and reasonable departure times, and I'd rather pay a little more than take a terrible itinerary."

The agent should understand the request, notice what's missing, recognize that it needs current web information, discover a suitable skill, search, extract, normalize, apply preferences, recommend options with links and sources, and then evaluate its own answer.

The deeper question: **can a small, evaluation-driven agent progressively acquire useful capabilities without turning into a pile of agent-specific code?** Every stage below adds one capability *and* the evals that prove it.

```text
CURRENT   Python foundation (uv, ruff, pyright strict, pytest, CI, pre-commit)
    ↓
CURRENT   offline agent loop (ask → select → execute → observe → answer)
    ↓
CURRENT   skill discovery (filesystem catalog, capability tags, request_capability)
    ↓
NEXT      real Claude (ClaudeClient exists; verify live, tune prompt with evals)
    ↓
NEXT      web research skill
    ↓
NEXT      flight research skill
    ↓
NEXT      preference-aware flight recommendations
    ↓
NEXT      agent evaluations for flight research
    ↓
NORTH STAR  evaluation-driven agent that can discover and use capabilities to solve real tasks
```

## Stage 0: Bootstrap ✅

Project tooling, CI, `LLMClient` with fakes, the agent loop, a skill catalog with one deterministic skill (`calculator`), TOML eval cases, and the flywheel. Everything runs offline.

## Stage 1: Real Claude ✅ (verified live; runs in CI)

`ClaudeClient` (`src/agentlab/llm/claude.py`) implements `LLMClient` on the Messages API, using native tool use, refusal fallbacks, explicit effort, and verbatim replay of thinking blocks.

- [x] Live tests pass, and the eval suite passes 5/5 against Claude (after fixing a too-strict number check and steering capability names toward general ones).
- [x] Live tests and evals run in CI (`live-claude` job) and are required to merge.
- [ ] Keep the live pass rate at 100% as cases get harder.

**Done when** the full eval suite passes against Claude across several runs.

## Stage 2: One deterministic skill ✅ (`calculator`)

This stage proves that the agent discovers a capability, selects it, executes it, receives a structured result, and continues reasoning.

## Stage 3: Skill discovery (foundation in place)

Today the whole catalog is offered on every turn, and `request_capability` records gaps. Next:

- [x] On-demand learning: when no skill fits, the agent proposes generic skills (rule-checked, at most 5 steps and 2 new skills), the user approves the plan, a runtime harness builds and tests them, and the skill is used and kept locally (`.agentlab/learned/`). Evals cover ready, declined, rejected, reuse and not-worth-learning, offline and live.
- [ ] Let `request_capability` search a wider catalog by capability tag and load the match into the current run.
- [ ] Promote a learned skill into the repo (review, then a manifest and in-package code with evals). This is manual for now.
- [ ] Before anyone but the author uses agentlab: restore limits on learned skills. Learned skills run unrestricted today (any import, files, network, commands). Bring back per-skill permissions the user approves (read files, fetch the web, write files, …), enforced at the OS level (read-only mounts, network namespaces), not just by an import allowlist.

There is no marketplace and no plugin ecosystem. A local catalog is enough.

## Stage 4: Web research

- [ ] A `web_research` skill with capability `current_information`. Candidate backends: Anthropic server-side web search/fetch, or a search API, kept behind the skill boundary.
- [ ] Update `flight_sfo_tokyo` to expect `web_research` to be selected instead of a gap.
- [ ] Evals: "needs fresh info → selects web", "doesn't need it → doesn't".
- [ ] Apply the security principles in [architecture.md](architecture.md#security-principles-for-future-web-skills): web content is untrusted, and no side effects.

The agent must decide it needs current information from the skill descriptions. Never write `if "flight" in prompt`.

## Stage 5: Flight research

- [ ] A `flight_search` skill (or a flight-specific mode of web research): search → collect → normalize → compare → recommend.
- [ ] Results carry source URLs. Read-only: no booking automation.

## Stage 6: Evaluation-driven improvement

Flight evals that measure:

- finding current information
- correct extraction
- preference adherence
- recommendation quality
- source and link quality
- handling missing information (asking, or stating assumptions)
- avoiding hallucinated flight details

Live data changes, so these evals check properties (each recommendation links to a source, prices are attributed, nonstop preference is respected), not fixed answers. Recorded fixtures of real search results can make extraction evals deterministic. They must be labeled as fixtures, never presented as live data.

From here the development loop is: change → run evals → inspect failures → improve → run evals again. This is the first meaningful flywheel.
