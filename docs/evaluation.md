# Evaluation

Evaluation is the project's main feedback loop. A behavior we care about exists only once an eval case checks it.

## Cases

Each file in `evals/cases/*.toml` is one case:

```toml
id = "ask_arithmetic"
kind = "agent"                       # "agent" runs the full loop; "skill" calls one skill
description = "..."
tags = ["agent", "skill-selection"]
task = "What is 123 * 456?"          # agent cases; skill cases use skill = "..." and arguments = {...}

[expect]
skills_used = ["calculator"]
answer_contains = ["56088"]
stop_reason = "answered"
```

### Checks

| Check | Kind | Passes when |
|---|---|---|
| `answer_contains` | agent | Each string appears in the answer. Case-insensitive, and thousands separators are ignored (`56,088` matches `56088`). |
| `answer_excludes_patterns` | agent | No regex matches the answer. Use it to guard against hallucinated prices, flight numbers or links. |
| `skills_used` | agent | Each listed skill was invoked. |
| `no_skills_used` | agent | No skill was invoked. |
| `capability_gap` | agent | The agent reported this missing capability through `request_capability`. |
| `stop_reason` | agent | The run ended as expected (`answered` or `max_steps`). |
| `output_equals` | skill | The skill's output equals this table. |
| `error_contains` | skill | The skill's error message contains this string. |

Unknown checks, or checks used with the wrong kind, are rejected when the case loads. A case must declare at least one check.

## Scoring

- Each check yields a `CheckResult(name, passed, detail)`. The detail explains a failure.
- A case **passes** when all its checks pass. Its **score** is the fraction of checks that pass.
- An `LLMError` during a case is recorded as an `error` result with score 0. It does not crash the suite.
- `EvalSummary` reports the pass rate and mean score, and groups failures **by check name**, which serves as the failure mode.

## Component vs agent evaluation

- **Component (`kind = "skill"`)** answers "does the calculator return the right result?" It calls `catalog.execute` directly, with no model involved.
- **Agent (`kind = "agent"`)** answers "given this task, did the agent select the right capability, avoid the wrong ones, report what it couldn't do, and answer without inventing things?" It runs the full loop and checks the `AgentRun` trace.

The north-star fixture `flight_sfo_tokyo` is an agent case. It currently expects the agent to report a `current_information` gap, use no skills, and include no prices, flight numbers or links in its answer. When a web research skill lands, this case changes to expect that skill to be selected. That change is the first thing that shows Stage 4 is real.

## Offline evaluation

`uv run agentlab eval --offline` runs the suite with `OfflineLLM`, a deterministic stand-in. It needs no key and no network, and CI runs it on every push. It exits non-zero if any case fails.

`OfflineLLM` is not a model. Offline runs prove the plumbing: discovery, selection by declared capability, execution, observation, gap reporting and scoring. They say little about the quality of a real model's judgment. For that, run without `--offline`, which uses Claude and requires `ANTHROPIC_API_KEY` (in `.env` or the environment). CI runs the suite against Claude in the `live-claude` job, which must pass to merge.

## How failures feed development

```text
change agent / skill / prompt
        ↓
uv run agentlab flywheel [--offline]
        ↓
read runs/<timestamp>/failures.md
        ↓
pick the most frequent failure mode, make one targeted change
        ↓
run the flywheel again; check "Newly failing" is empty
```

Each flywheel run writes `runs/<UTC timestamp>/results.json` (full results) and `failures.md` (a readable summary). It compares against the previous run to list newly failing and newly passing cases. `runs/` is git-ignored. The flywheel always exits 0 because its job is to report; `eval` is the gate.

Rules of thumb:

- **When you add a behavior, add a case.** When you fix a bug, add the case that would have caught it.
- **When everything passes, add a harder case.** A suite where everything always passes has stopped telling you anything.
- **Never weaken a check to make it pass.** Fix the behavior, or record why the expectation was wrong.
- **Live (Claude) results vary between runs.** Compare pass rates across runs, not single outcomes.
