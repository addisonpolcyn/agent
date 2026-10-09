# Tools

A **tool** is one of the two concepts in agentlab (the other is the workflow; see [workflows.md](workflows.md)). It is a capability the agent can discover and invoke: something it can *do*, with a name, a description of when to use it, an input schema and a structured result. A tool is not a prompt.

Two kinds exist:

- **Built-in tools** live in the repo: a manifest in `tools/` and trusted code in the `agentlab` package. Today there is one, `calculator`. Planned ones include `web_research` and `flight_search` (see [roadmap.md](roadmap.md)).
- **Learned tools** are generated at runtime when the agent needs a capability it lacks and the user approves. They live only in the local `.agentlab/learned/` directory and run unrestricted in their own process. See [Learned tools](#learned-tools).

## Anatomy

```text
tools/calculator/tool.toml                  ← metadata the agent reads (discovered)
src/agentlab/tools/builtin/calculator.py     ← implementation (trusted, in-package)
```

### Manifest (`tool.toml`)

```toml
name = "calculator"
description = "Evaluates an arithmetic expression exactly and returns the numeric result."
when_to_use = "The task requires computing a numeric answer from arithmetic ..."
limitations = "Numbers and operators only. No variables, functions, units, or dates. ..."
capabilities = ["arithmetic"]
implementation = "agentlab.tools.builtin.calculator:run"

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
| `description`, `when_to_use`, `limitations` | Combined into the tool description the model reads when selecting a tool. |
| `capabilities` | Abstract capability tags (`arithmetic`, later `current_information`, `flight_search`). Shown to the model as `Provides: ...`. They are the vocabulary of capability discovery. |
| `input_schema` / `output_schema` | JSON Schema for arguments and results. |
| `implementation` | `agentlab.<module>:<function>`. Must be inside the `agentlab` package. |

### Implementation

A plain function:

```python
def run(arguments: Mapping[str, Any]) -> JSONObject: ...
```

It validates its own input and raises `ToolError` for expected failures (bad input, division by zero, an upstream outage). `ToolCatalog.execute` turns a `ToolError` into `ToolResult(error=...)`, which the agent feeds back to the model as an error observation. Any other exception is a bug and propagates.

## Discovery and execution

1. `discover_catalog(tools_dir)` reads every `tools/*/tool.toml`, validates it, and resolves its implementation. Malformed manifests fail loudly at startup.
2. The agent offers `catalog.tool_specs()`, plus the built-in `request_capability`, to the model on every turn.
3. The model selects a tool by reading the descriptions. There is no `if "flight" in prompt` routing anywhere.
4. `catalog.execute(name, args)` runs it and returns a `ToolResult`.
5. If no tool fits, the model calls `request_capability`, and the gap is recorded in the trace.

`uv run agentlab tools` lists what was discovered.

## Adding a tool

1. Write `src/agentlab/tools/builtin/<name>.py` with a `run(arguments)` function. Raise `ToolError` on expected failures.
2. Add `tools/<name>/tool.toml`. Spend most of your effort on `when_to_use` and `limitations`: they are what the model selects on.
3. Add unit tests for the function, plus a tool eval case (`kind = "tool"`) in `evals/cases/`.
4. Add an agent eval case (`kind = "agent"`) showing a task where the agent should select it, and ideally one where it shouldn't.
5. Run `uv run agentlab flywheel --offline`. With Claude, run it without `--offline`.

The agent loop does not change. If you find you need to change it, write down why in the PR.

## Learned tools

When no tool fits and the missing piece is a data transformation or reading local files, the agent can offer to build one. The code is in `src/agentlab/learning/`.

```text
propose_tool_plan ─► validate_plan ─► the user: "I don't have this, but I can build it. OK?"
   ─► contract + tests (written blind to the code) ─► code ─► runtime harness (≤ 3 attempts)
   ─► passes ─► saved to .agentlab/learned/<name>/ and used (no second question)
```

**When to learn at all.** Learning costs model calls and the user's attention. The agent answers small, one-off or judgment tasks itself. It reaches for a tool (existing or learned) when the result must be exact and the input is large or error-prone, when the work will recur, or when the user asks for a reusable capability. Existing tools always come first.

**Plan rules** (`learning/plan.py`, deterministic, applied before the user is asked):

| Rule | Outcome |
|---|---|
| More than 5 steps, or more than 2 new tools in one request (counted across all its plans) | `refused_too_large`: split the request |
| A new tool duplicates an existing name or capability, or nothing new is needed | `refused_reuse`: use what exists |
| Unknown reused tool, non-snake_case or duplicate names, bad shape | `malformed` |

New tools must be generic: name them for the operation (`html_to_text`, `json_query`), not the task, and take every request-specific value (host, account, URL, path, date, search term) as an argument. See [workflows.md § Distilling](workflows.md#distilling-workflows-into-generic-tools).

**Runtime harness** (`learning/harness.py`). A first model call writes the schemas and 6–12 tests from the spec alone (normal, edge and error cases). A second call writes the code; it sees only some of the tests, and the rest are held out. Error tests are never held out, since the code writer can't guess an unseen error message. A candidate is ready only if it passes every check:

| Check | Catches |
|---|---|
| `static` | Code that isn't valid Python with a top-level `run(arguments)` |
| `visible_tests`, `holdout_tests` | Wrong answers, and code fitted to the examples it saw |
| `no_crashes` | Exceptions instead of a clean `ToolError` |
| `output_schema` | Output that doesn't match the declared schema |
| `non_constant` | The same output for every input |
| `deterministic` | Different output on a second run |
| `hardcoded` | Test inputs embedded as literals |

Failed checks go back to the code writer as feedback, but the held-out inputs never do. The tests stay fixed across attempts. The harness gives up after 3 attempts, or earlier if an attempt repeats the previous failures exactly, and the agent tells the user it isn't working and why.

**Files, network and commands.** Learned tools may do anything Python can. Every tool also gets two conveniences with friendly `ToolError`s:

- `list_dir(path)` returns entries with `name`, `type` and `size`.
- `read_text(path)` returns up to 1 MB of text.

A plan step sets `reads_files` when the tool reads files; that makes the harness require fixture-file tests. Tools whose real output depends on the machine, the clock or live data (CPU usage, a web page) are told to take that data as an input, such as raw text or a path with the real source as the default, so their tests stay deterministic.

Relative paths mean the directory agentlab runs in.

The harness tests file tools on fixture files. Each test declares `files` (relative path → text). The harness creates them in a fresh folder and substitutes that folder for `{root}` in the arguments. Nothing stops a candidate from touching other paths during its tests, so a model that ignores its instructions could; that's the cost of running unrestricted.

**Tool process** (`learning/sandbox.py`). Not a security boundary. Generated code runs in a separate `python -P` process with the user's environment and working directory, full builtins and any import, and a 30-second timeout. The process keeps a crash or hang from taking the agent down, and keeps `print` output out of the reply. agentlab is a single-user toy for now, so the safeguard is the user's approval of the plan, whose prompt says the tool runs with full access; the code is saved in `.agentlab/learned/` for anyone who wants to read it. The previous sandbox (import allowlist, restricted builtins, rlimits, per-folder approval, secret hiding) is in git history, to be restored with real OS-level enforcement before anyone else uses agentlab.

**Storage.** `.agentlab/learned/<name>/` holds `tool.json` (the manifest, `implementation = "sandbox:tool.py"`, plus `reads_files` when set), `tool.py`, `tests.json` and `report.json` (the verdict, every attempt's checks and the generated tests). A failed or rejected build leaves only `report.json`, which is never loaded. Learned tools are reloaded on later runs. A tampered file that isn't a usable tool fails loudly at startup. To forget a tool, delete its directory.

**Starter tools.** Some learned tools ship with agentlab, in `src/agentlab/learning/starter/<name>/` (`tool.json` + `tool.py`, reviewed and tested in `tests/`). `install_starter_tools` copies each one into the learned directory when `ask`, `chat` or `tools` launches, unless a directory of that name is already there, so local edits survive. After that they are ordinary learned tools: same tool process, same loading. Deleting one brings it back at the next launch. `--no-learn` ignores them, and evals don't install them (each case learns into a fresh, empty directory), so the learning evals still test learning. Today there is one: `run_command`, which runs a shell command and returns its exit code and output. It's a starter rather than a built-in because a shell that's always offered stands in for most tools the agent would otherwise learn (live, Claude ran `ls` and `sha256sum` instead of learning tools, and counted words with the shell after the user declined to build a word counter).

**Within a run.** When a tool becomes ready, the loop continues in a fresh conversation that starts with the task, as if the run had started with the tool. The tool list changes at that point, and some providers (Claude's thinking blocks) bind earlier turns to the tools they were produced with, so the earlier conversation is left behind rather than edited.

**Offline.** `OfflineLLM` proposes learning for three fixture capabilities (word counting, listing a folder, and SHA-256 hashing), and `FixtureAuthorLLM` supplies canned `word_count`, `list_files` and `sha256_hex` contracts and code, all labeled as fixtures. `sha256_hex` imports `hashlib`, which the old allowlist refused. Offline runs and CI therefore exercise the real rules, approval, tool process, harness and store.

## Current limitations

- **Built-in implementations are in-package Python only.** Manifests are discovered from the filesystem, but code is never loaded from the `tools/` directory. This is deliberate: no arbitrary code execution, no dynamic installs, no marketplace. The one exception is learned tools, which run in their own process after the user approves the plan (above).
- **No schema validation of arguments by the catalog.** Each tool validates its own input. Revisit when there are enough tools for duplication to hurt.
- **One flat catalog, offered in full on every turn.** That's fine for a handful of tools. Larger catalogs will need search-based discovery, which is the job of `request_capability` in Stage 3.
- **Synchronous execution.** No timeouts or cancellation yet. Web tools will need them.

## Security

Tool output, and web content in particular, is untrusted input. It returns to the model only as a tool result. Tools that cause external side effects need approval gates. See [architecture.md § Security principles](architecture.md#security-principles-for-future-web-tools).
