# Skills

A **skill** is a capability the agent can discover and invoke: something it can *do*, with a name, a description of when to use it, an input schema and a structured result. A skill is not a prompt.

Two kinds exist:

- **Built-in skills** live in the repo: a manifest in `skills/` and trusted code in the `agentlab` package. Today there is one, `calculator`. Planned ones include `web_research` and `flight_search` (see [roadmap.md](roadmap.md)).
- **Learned skills** are generated at runtime when the agent needs a capability it lacks and the user approves. They live only in the local `.agentlab/learned/` directory and run unrestricted in their own process. See [Learned skills](#learned-skills).

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

When no skill fits and the missing piece is a data transformation or reading local files, the agent can offer to build one. The code is in `src/agentlab/learning/`.

```text
propose_skill_plan ─► validate_plan ─► the user: "I don't have this, but I can build it. OK?"
   ─► contract + tests (written blind to the code) ─► code ─► runtime harness (≤ 3 attempts)
   ─► passes ─► saved to .agentlab/learned/<name>/ and used (no second question)
```

**When to learn at all.** Learning costs model calls and the user's attention. The agent answers small, one-off or judgment tasks itself. It reaches for a tool (existing or learned) when the result must be exact and the input is large or error-prone, when the work will recur, or when the user asks for a reusable capability. Existing skills always come first.

**Plan rules** (`learning/plan.py`, deterministic, applied before the user is asked):

| Rule | Outcome |
|---|---|
| More than 5 steps, or more than 2 new skills | `refused_too_large`: split the request |
| A new skill duplicates an existing name or capability, or nothing new is needed | `refused_reuse`: use what exists |
| Unknown reused skill, non-snake_case or duplicate names, bad shape | `malformed` |

New skills must be generic: name them for the operation (`html_to_text`, `json_query`), not the task.

**Runtime harness** (`learning/harness.py`). A first model call writes the schemas and 6–12 tests from the spec alone (normal, edge and error cases). A second call writes the code; it sees only some of the tests, and the rest are held out. A candidate is ready only if it passes every check:

| Check | Catches |
|---|---|
| `static` | Code that isn't valid Python with a top-level `run(arguments)` |
| `visible_tests`, `holdout_tests` | Wrong answers, and code fitted to the examples it saw |
| `no_crashes` | Exceptions instead of a clean `SkillError` |
| `output_schema` | Output that doesn't match the declared schema |
| `non_constant` | The same output for every input |
| `deterministic` | Different output on a second run |
| `hardcoded` | Test inputs embedded as literals |

Failed checks go back to the code writer as feedback, but the held-out inputs never do. The tests stay fixed across attempts. The harness gives up after 3 attempts, or earlier if an attempt repeats the previous failures exactly, and the agent tells the user it isn't working and why.

**Files, network and commands.** Learned skills may do anything Python can. Every skill also gets two conveniences with friendly `SkillError`s:

- `list_dir(path)` returns entries with `name`, `type` and `size`.
- `read_text(path)` returns up to 1 MB of text.

A plan step sets `reads_files` when the skill reads files; that makes the harness require fixture-file tests. Skills whose real output depends on the machine, the clock or live data (CPU usage, a web page) are told to take that data as an input, such as raw text or a path with the real source as the default, so their tests stay deterministic.

Relative paths mean the directory agentlab runs in.

The harness tests file skills on fixture files. Each test declares `files` (relative path → text). The harness creates them in a fresh folder and substitutes that folder for `{root}` in the arguments. Nothing stops a candidate from touching other paths during its tests, so a model that ignores its instructions could; that's the cost of running unrestricted.

**Skill process** (`learning/sandbox.py`). Not a security boundary. Generated code runs in a separate `python -P` process with the user's environment and working directory, full builtins and any import, and a 30-second timeout. The process keeps a crash or hang from taking the agent down, and keeps `print` output out of the reply. agentlab is a single-user toy for now, so the safeguard is the user's approval of the plan, whose prompt says the skill runs with full access; the code is saved in `.agentlab/learned/` for anyone who wants to read it. The previous sandbox (import allowlist, restricted builtins, rlimits, per-folder approval, secret hiding) is in git history, to be restored with real OS-level enforcement before anyone else uses agentlab.

**Storage.** `.agentlab/learned/<name>/` holds `skill.json` (the manifest, `implementation = "sandbox:skill.py"`, plus `reads_files` when set), `skill.py`, `tests.json` and `report.json` (the verdict, every attempt's checks and the generated tests). A failed or rejected build leaves only `report.json`, which is never loaded. Learned skills are reloaded on later runs. A tampered file that isn't a usable skill fails loudly at startup. To forget a skill, delete its directory.

**Within a run.** When a skill becomes ready, the loop continues in a fresh conversation that starts with the task, as if the run had started with the skill. The tool list changes at that point, and some providers (Claude's thinking blocks) bind earlier turns to the tools they were produced with, so the earlier conversation is left behind rather than edited.

**Offline.** `OfflineLLM` proposes learning for three fixture capabilities (word counting, listing a folder, and SHA-256 hashing), and `FixtureAuthorLLM` supplies canned `word_count`, `list_files` and `sha256_hex` contracts and code, all labeled as fixtures. `sha256_hex` imports `hashlib`, which the old allowlist refused. Offline runs and CI therefore exercise the real rules, approval, skill process, harness and store.

## Current limitations

- **Built-in implementations are in-package Python only.** Manifests are discovered from the filesystem, but code is never loaded from the `skills/` directory. This is deliberate: no arbitrary code execution, no dynamic installs, no marketplace. The one exception is learned skills, which run in their own process after the user approves the plan (above).
- **No schema validation of arguments by the catalog.** Each skill validates its own input. Revisit when there are enough skills for duplication to hurt.
- **One flat catalog, offered in full on every turn.** That's fine for a handful of skills. Larger catalogs will need search-based discovery, which is the job of `request_capability` in Stage 3.
- **Synchronous execution.** No timeouts or cancellation yet. Web skills will need them.

## Security

Skill output, and web content in particular, is untrusted input. It returns to the model only as a tool result. Skills that cause external side effects need approval gates. See [architecture.md § Security principles](architecture.md#security-principles-for-future-web-skills).
