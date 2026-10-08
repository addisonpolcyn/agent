# Skills

A **skill** is a capability the agent can discover and invoke: something it can *do*, with a name, a description of when to use it, an input schema and a structured result. A skill is not a prompt.

Two kinds exist:

- **Built-in skills** live in the repo: a manifest in `skills/` and trusted code in the `agentlab` package. Today there is one, `calculator`. Planned ones include `web_research` and `flight_search` (see [roadmap.md](roadmap.md)).
- **Learned skills** are generated at runtime when the agent needs a capability it lacks and the user approves. They live only in the local `.agentlab/learned/` directory and always run in a sandbox. See [Learned skills](#learned-skills).

## Anatomy

```text
skills/calculator/skill.toml                  ← metadata the agent reads (discovered)
src/agentlab/skills/builtin/calculator.py     ← implementation (trusted, in-package)
```

### Manifest (`skill.toml`)

```toml
name = "calculator"
description = "Evaluates an arithmetic expression exactly and returns the numeric result."
when_to_use = "The task requires computing a numeric answer from arithmetic ..."
limitations = "Numbers and operators only. No variables, functions, units, or dates. ..."
capabilities = ["arithmetic"]
implementation = "agentlab.skills.builtin.calculator:run"

[input_schema]            # JSON Schema; sent to the model as the tool's input schema
type = "object"
required = ["expression"]
additionalProperties = false

[input_schema.properties.expression]
type = "string"
description = "An arithmetic expression, e.g. \"120 * 3\"."

[output_schema]           # JSON Schema of the result (documentation and future validation)
type = "object"
required = ["result"]
```

| Field | Purpose |
|---|---|
| `name` | Tool name the model calls. Unique. `request_capability` is reserved. |
| `description`, `when_to_use`, `limitations` | Combined into the tool description the model reads when selecting a skill. |
| `capabilities` | Abstract capability tags (`arithmetic`, later `current_information`, `flight_search`). Shown to the model as `Provides: ...`. They are the vocabulary of capability discovery. |
| `input_schema` / `output_schema` | JSON Schema for arguments and results. |
| `implementation` | `agentlab.<module>:<function>`. Must be inside the `agentlab` package. |

### Implementation

A plain function:

```python
def run(arguments: Mapping[str, Any]) -> JSONObject: ...
```

It validates its own input and raises `SkillError` for expected failures (bad input, division by zero, an upstream outage). `SkillCatalog.execute` turns a `SkillError` into `SkillResult(error=...)`, which the agent feeds back to the model as an error observation. Any other exception is a bug and propagates.

## Discovery and execution

1. `discover_catalog(skills_dir)` reads every `skills/*/skill.toml`, validates it, and resolves its implementation. Malformed manifests fail loudly at startup.
2. The agent offers `catalog.tool_specs()`, plus the built-in `request_capability`, to the model on every turn.
3. The model selects a skill by reading the descriptions. There is no `if "flight" in prompt` routing anywhere.
4. `catalog.execute(name, args)` runs it and returns a `SkillResult`.
5. If no skill fits, the model calls `request_capability`, and the gap is recorded in the trace.

`uv run agentlab skills` lists what was discovered.

## Adding a skill

1. Write `src/agentlab/skills/builtin/<name>.py` with a `run(arguments)` function. Raise `SkillError` on expected failures.
2. Add `skills/<name>/skill.toml`. Spend most of your effort on `when_to_use` and `limitations`: they are what the model selects on.
3. Add unit tests for the function, plus a skill eval case (`kind = "skill"`) in `evals/cases/`.
4. Add an agent eval case (`kind = "agent"`) showing a task where the agent should select it, and ideally one where it shouldn't.
5. Run `uv run agentlab flywheel --offline`. With Claude, run it without `--offline`.

The agent loop does not change. If you find you need to change it, write down why in the PR.

## Learned skills

When no skill fits and the missing piece is a pure data transformation, the agent can offer to build one. The code is in `src/agentlab/learning/`.

```text
propose_skill_plan ─► validate_plan ─► gate 1: "I don't have this, but I can build it. OK?"
   ─► contract + tests (written blind to the code) ─► code ─► runtime harness (≤ 3 attempts)
   ─► gate 2: the code and its test results. "Use it?" ─► saved to .agentlab/learned/<name>/
```

**When to learn at all.** Learning costs model calls and the user's attention. The agent answers small, one-off or judgment tasks itself. It reaches for a tool (existing or learned) when the result must be exact and the input is large or error-prone, when the work will recur, or when the user asks for a reusable capability. Existing skills always come first.

**Plan rules** (`learning/plan.py`, deterministic, applied before the user is asked):

| Rule | Outcome |
|---|---|
| More than 5 steps, or more than 2 new skills | `refused_too_large`: split the request |
| A *new* skill needs the network or has side effects | `refused_not_learnable`: report the gap instead. Reusing an existing trusted skill for such a step is fine. |
| A new skill duplicates an existing name or capability, or nothing new is needed | `refused_reuse`: use what exists |
| Unknown reused skill, non-snake_case or duplicate names, bad shape | `malformed` |

New skills must be generic: name them for the operation (`html_to_text`, `json_query`), not the task.

**Runtime harness** (`learning/harness.py`). A first model call writes the schemas and 6–12 tests from the spec alone (normal, edge and error cases). A second call writes the code; it sees only some of the tests, and the rest are held out. A candidate is ready only if it passes every check:

| Check | Catches |
|---|---|
| `static` | Code outside the sandbox allowlist |
| `visible_tests`, `holdout_tests` | Wrong answers, and code fitted to the examples it saw |
| `no_crashes` | Exceptions instead of a clean `SkillError` |
| `output_schema` | Output that doesn't match the declared schema |
| `non_constant` | The same output for every input |
| `deterministic` | Different output on a second run |
| `hardcoded` | Test inputs embedded as literals |

Failed checks go back to the code writer as feedback, but the held-out inputs never do. The tests stay fixed across attempts. The harness gives up after 3 attempts, or earlier if an attempt repeats the previous failures exactly, and the agent tells the user it isn't working and why.

**Sandbox** (`learning/sandbox.py`). Generated code is untrusted. It is checked against an AST allowlist: pure-data stdlib imports only (`json`, `re`, `html.parser`, …); no `open`, `eval`, `getattr` or `type`; no `_private` or dunder access except `__init__`; no `str.format`. It then runs in a separate `python -I` process with an empty environment and working directory, restricted builtins, a timeout and CPU, memory, file and process limits. There is no OS-level network block; that rests on the import allowlist. This is defense in depth, and the human review at gate 2 is part of it.

**Storage.** `.agentlab/learned/<name>/` holds `skill.json` (the manifest, `implementation = "sandbox:skill.py"`), `skill.py`, `tests.json` and `report.json`. A failed or rejected build leaves only `report.json`, which is never loaded. Learned skills are reloaded on later runs. A tampered or unsafe file fails loudly at startup. To forget a skill, delete its directory.

**Within a run.** When a skill becomes ready, the loop continues in a fresh conversation that starts with the task, as if the run had started with the skill. The tool list changes at that point, and some providers (Claude's thinking blocks) bind earlier turns to the tools they were produced with, so the earlier conversation is left behind rather than edited.

**Offline.** `OfflineLLM` proposes learning for one fixture capability (word counting), and `FixtureAuthorLLM` supplies a canned `word_count` contract and code, both labeled as fixtures. Offline runs and CI therefore exercise the real rules, gates, sandbox, harness and store.

## Current limitations

- **Built-in implementations are in-package Python only.** Manifests are discovered from the filesystem, but code is never loaded from the `skills/` directory. This is deliberate: no arbitrary code execution, no dynamic installs, no marketplace. The one exception is learned skills, which run only in the sandbox (above).
- **No schema validation of arguments by the catalog.** Each skill validates its own input. Revisit when there are enough skills for duplication to hurt.
- **One flat catalog, offered in full on every turn.** That's fine for a handful of skills. Larger catalogs will need search-based discovery, which is the job of `request_capability` in Stage 3.
- **Synchronous execution.** No timeouts or cancellation yet. Web skills will need them.

## Security

Skill output, and web content in particular, is untrusted input. It returns to the model only as a tool result. Skills that cause external side effects need approval gates. See [architecture.md § Security principles](architecture.md#security-principles-for-future-web-skills).
