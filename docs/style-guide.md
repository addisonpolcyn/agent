# Style guide

How we work in this repo, and why. Each rule comes with its reason; when a rule and its reason disagree in some situation, follow the reason and update this page.

## Engineering philosophy

**Small, real and tested beats large, abstract and hypothetical.**
*Why:* The point of this project is to find out what a small agent architecture can actually do. Code that doesn't run against real inputs teaches us nothing and still has to be maintained.

**Build the spine before the features.**
*Why:* CLI → loop → skills → LLM → evals → flywheel exist end to end before any of them get clever. Each new capability then lands in a place that already works, with an eval that already runs.

**Don't build a framework for yourself.**
*Why:* "Might be useful someday" abstractions guess wrong about the future and make the present harder to read. Add an abstraction when the second real use appears, not before. No factories or registries for a single implementation.

**Keep the agent loop readable in one sitting.**
*Why:* `agent/loop.py` is the heart of the system. If understanding it means jumping across files, the design has drifted. Keep planning, memory and recursion out until an eval demands them.

## Evaluation-driven development

**Every behavior change ships with an eval case. Every bug fix ships with the case that would have caught it.**
*Why:* Agent behavior regresses silently. Without evals, "it worked when I tried it" is the only evidence, and that doesn't scale past one person and one afternoon.

**The loop is: change → run evals → read failures → one targeted fix → run again.**
*Why:* One change per iteration means a change in the pass rate can be attributed. Batched changes make the flywheel's diff meaningless.

**Never weaken a check to make it pass.**
*Why:* The eval is the specification. If the expectation was wrong, change it deliberately and say why in the commit. Don't quietly relax a regex.

**Evaluate the trace, not just the answer.**
*Why:* "Said 56088" can be luck. "Selected the calculator, got 56088 back and reported it" is behavior. The `AgentRun` trace exists so evals can check the *how*.

## Boundaries

**Protocols only at genuine boundaries.** Today those are `LLMClient` and the skill catalog.
*Why:* Those are where implementations really vary (Claude vs. fake; calculator vs. web vs. MCP). Elsewhere, concrete classes are simpler and just as testable.

**Vendor SDK types stay inside their adapter.** Only `llm/claude.py` imports `anthropic`.
*Why:* Swapping or adding a provider must not ripple through the codebase. Provider-specific state round-trips through the opaque `provider_state`.

**Skills know nothing about the loop, and the loop knows nothing about skills.**
*Why:* Future skills will be backed by APIs, web fetches, browsers or MCP. The loop has to stay stable while they change, and `SkillCatalog.execute` is the only contract between them.

## Capability discovery over hard-coding

**Never route on task text** (`if "flight" in prompt: ...`). The model selects from skill manifests. When nothing fits, it says so through `request_capability`.
*Why:* Exploring capability discovery is the point of the project. Keyword routing would make every new capability a code change in the core, which is exactly the "pile of agent-specific code" we are trying to avoid.

**Invest in `when_to_use` and `limitations`.**
*Why:* They are the model's only basis for selection. A vague description shows up later as a selection failure in the evals.

## Honesty rules

**Test doubles are labeled as test doubles.** `OfflineLLM` says what it is in its docstring and in its offline answers.
*Why:* Nobody should mistake a fake for production capability, and that includes future readers of an eval report.

**Never present fabricated data as real.** Recorded fixtures are labeled as fixtures.
*Why:* A flight agent that invents flights is worse than no agent. The same discipline applies to our own test data.

**Prefer "I can't" to a guess.** The agent reports capability gaps instead of improvising.
*Why:* An honest gap is actionable: it tells us which skill to build next. A plausible hallucination hides the problem.

## Code style

- **Strict typing everywhere** (pyright strict). *Why:* Types are the cheapest documentation, and they keep boundaries honest.
- **Frozen dataclasses for data.** *Why:* Immutable values make traces and results safe to pass around and compare.
- **Small functions with explicit names.** *Why:* Readable diffs and readable stack traces.
- **Composition over inheritance.** *Why:* Behavior is assembled at the CLI edge (`Agent(llm, catalog)`), not inherited.
- **No global mutable state.** Settings are read once in `cli.py` and passed down. *Why:* Tests can build any configuration without monkeypatching modules.
- **Explicit error types** (`LLMError`, `SkillError`, `SkillManifestError`, `EvalCaseError`), caught only where they can be handled. *Why:* An expected failure is information, and a swallowed exception is a hidden bug. Unexpected exceptions propagate.
- **No dead code, no commented-out code.** *Why:* Git remembers; the reader shouldn't have to.
- **Comments explain why, not what.** *Why:* The code already says what it does.
- **Match the surrounding code.** *Why:* Consistency is what lets a small codebase stay small.

## Dependencies

Before adding one, ask:

1. Is it necessary?
2. Does the standard library solve it?
3. Does it materially improve correctness?
4. Does it add operational complexity?

*Why:* Every dependency is code we run but don't read. Today the only runtime dependency is `anthropic`. We use `tomllib` instead of PyYAML, `dataclasses` instead of pydantic, and `argparse` instead of click, because the stdlib versions are enough.

Avoid, unless the current implementation genuinely requires it: LangChain or other agent frameworks, vector databases, databases, Redis, queues, orchestration frameworks, web frameworks, MCP and browser automation.

## Testing

- **Deterministic and offline by default.** *Why:* Flaky tests get ignored. CI must pass without keys or network.
- **Three tiers.** `unit/` (one component), `integration/` (real pieces wired together offline) and `external/` (real services, opt-in with `-m live`).
- **`ScriptedLLM` for state transitions, `OfflineLLM` for end-to-end runs.** *Why:* Scripted responses test the loop's exact behavior on each kind of model turn. The offline stand-in exercises the full plumbing with realistic flow.

## Security

- **Web content and tool results are untrusted input.** They go back to the model as tool results, never as instructions.
- **Side-effecting skills need human approval gates.** Read-only comes first.
- **Secrets never reach prompts, logs or reprs** (`config.Secret`).

*Why:* As soon as the agent reads the web, every page is a potential prompt injection. See [architecture.md § Security principles](architecture.md#security-principles-for-future-web-skills).

## Docs hygiene

**When a change alters documented behavior, update the doc in the same change. Keep [CLAUDE.md](../CLAUDE.md) current as the map.**
*Why:* Stale docs are worse than none. CLAUDE.md is the first thing humans and Claude read, so it must point to the right places.
