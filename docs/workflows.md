# Tools and workflows

The agent builds a library of validated Python **tools** and composes them into **workflows**. That is the whole system: two concepts.

- **Tool**: a validated, versioned Python function with typed inputs and outputs. The building block. See [tools.md](tools.md) for how tools work today.
- **Workflow**: a saved DAG of tool calls and LLM steps that answers a type of request.

Every request runs as a workflow, reused if one fits or built if not, and every workflow is made of tools from the library.

The value of the library is accumulated, verified operational knowledge (auth flows, API quirks, fixes that took several attempts), not the code itself. Measure every design choice below against that.

**Status.** Tools exist: built-in ones in `tools/`, and learned ones in `.agentlab/learned/`. Workflows do not exist yet; today the agent loop composes tools ad hoc on each run. The [build order](roadmap.md#tools-and-workflows) tracks the path from here.

## Glossary

Use these names in code. The same table heads [CLAUDE.md](../CLAUDE.md).

| Name | What it is |
|---|---|
| Tool | A validated, versioned Python function with typed input and output schemas |
| Workflow | A DAG of steps (tool calls and LLM reasoning steps) that answers a type of request; parameterized |

Words to avoid in identifiers:

- **Task**: collides with the user's request and with task-list concepts. Older code still says `task` (`Agent.run(task)`, eval cases' `task` field); don't add new uses.
- **Chain**: drags in LangChain assumptions.
- **Skill**: collides with Claude's Skills, which are instructions, not code. agentlab called tools "skills" until this rename.
- **Agent**: reserved for the system as a whole.

## Data model

Two record types. Fields marked *planned* aren't in the manifest yet.

### Tool

| Field | Status | Purpose |
|---|---|---|
| `name`, `description` | today | Description written for retrieval, not for humans. Today it's `description` + `when_to_use` + `limitations` + `capabilities`. |
| `input_schema`, `output_schema` | today | Typed; used to wire steps and validate chains. |
| `id` | planned | Stable identity across renames. |
| `version_hash` | planned | Hash of code + schemas. |
| `purity` | planned | `pure` or `side_effecting` (unsubscribe, send, delete are side-effecting). |
| `external_systems` | planned | APIs or services it touches. |
| `notes` | planned | Learned quirks, auth details, fixes from failed attempts. |
| Stats | planned | `call_count`, `failure_rate`, `last_used`, `version_changes_30d`. |

### Workflow

| Field | Purpose |
|---|---|
| `id`, `name`, `description` | What kind of request it answers. |
| `params` | Named parameter slots (location, date, mailbox). |
| `steps` | Tool calls and LLM steps, wired by schema. |
| `tool_versions` | The `version_hash` of each tool it uses, so a tool change flags the workflow for re-validation. |
| Stats | `run_count`, `failure_rate`, `last_used`. |

## Tool lifecycle: create, split, evict

Splitting is about reuse and maintenance; eviction is about usage. Cold usage means evict, never split.

### Creation rules (enforced before a tool is saved)

- **Parameterize:** `get_weather(location, date)`, never `get_sf_weather()`. Today the learning prompt asks for generic tools, but no rule checks it ([tools.md § Learned tools](tools.md#learned-tools)).
- **One external system per tool.**
- **Never mix side effects with pure logic in one tool.**
- **Run the dedupe check first:** search the library for overlapping tools by embedding and code similarity; reuse or extract before writing new code. Today only exact name and capability duplicates are refused (`refused_reuse`).
- **Validate with a real smoke call plus schema assertions,** not only model-written unit tests.

### Split a tool when

1. **Partial duplication:** new tools keep reimplementing a chunk of it. Extract the shared chunk into its own tool. This is the main signal.
2. **Mixed purity:** it contains a side effect and pure logic. Always split.
3. **Mixed volatility:** trace failure attribution shows one part breaks often (usually the external call) while the rest is stable.
4. **Hidden judgment:** hardcoded heuristics doing an LLM's job (regex spam detection), showing up as eval failures. Replace with an LLM step.

### Evict a tool when

- No calls for a long horizon (months, not days) and no workflow depends on it. Flag it as unused first; delete later.
- Regeneration fails repeatedly after its external API changed.

## Metrics

The test of the whole system: the 50th run of a request category should beat the 1st on success rate, cost and latency. A flat curve means the library isn't earning its keep.

Track per request category:

- Success rate, cost per run and latency per run, over time.
- Workflow reuse rate (how often a request runs an existing workflow).
- Tool creation rate vs. dedupe hits. A rising creation rate at a steady workload means lookup is failing.
