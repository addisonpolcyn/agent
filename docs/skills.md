# Skills

A **skill** is a capability the agent can discover and invoke: something it can *do*, with a name, a description of when to use it, an input schema and a structured result. A skill is not a prompt.

Today there is one: `calculator`. Planned ones include `web_research` and `flight_search` (see [roadmap.md](roadmap.md)).

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

## Current limitations

- **Implementations are in-package Python only.** Manifests are discovered from the filesystem, but code is never loaded from the skills directory. This is deliberate: no arbitrary code execution, no dynamic installs, no marketplace.
- **No schema validation of arguments by the catalog.** Each skill validates its own input. Revisit when there are enough skills for duplication to hurt.
- **One flat catalog, offered in full on every turn.** That's fine for a handful of skills. Larger catalogs will need search-based discovery, which is the job of `request_capability` in Stage 3.
- **Synchronous execution.** No timeouts or cancellation yet. Web skills will need them.

## Security

Skill output, and web content in particular, is untrusted input. It returns to the model only as a tool result. Skills that cause external side effects need approval gates. See [architecture.md § Security principles](architecture.md#security-principles-for-future-web-skills).
