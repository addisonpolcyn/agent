# Architecture

agentlab is a small agent that solves tasks by discovering and invoking **tools**, and whose behavior is checked by an **evaluation suite** on every change. This page describes the system as it exists today (Stage 0) and why its boundaries are where they are.

## The flow

```text
User task
   │
   ▼
Agent loop  (src/agentlab/agent/loop.py)
   │   offers: catalog.tool_specs() + request_capability
   ▼
LLM  (LLMClient protocol: ClaudeClient | OfflineLLM | ScriptedLLM)
   │   returns: text and/or tool calls
   ▼
Capability selection ──────────────┬──────────────────────────────┐
   │ tool call                    │ request_capability(...)      │ propose_tool_plan(...)
   ▼                               ▼                              ▼
Tool catalog                 CapabilityGap recorded         ToolLearner (src/agentlab/learning/)
(src/agentlab/tools/catalog.py)   ("no tool provides X")   rules → approve plan → build + harness
   │                               │                         → store → catalog
   │ execute(name, args)           │
   ▼                               │
Tool implementation               │
(environment: today, pure Python)  │
   │ ToolResult (output | error)  │
   ▼                               ▼
Observation  (ToolResultMessage) ──► back to the LLM ──► … ──► final answer
   │
   ▼
AgentRun trace (answer, invocations, capability gaps, stop reason, steps)
   │
   ▼
Evaluation  (src/agentlab/evals/) ──► Flywheel  (src/agentlab/flywheel/) ──► runs/<ts>/failures.md
```

## Components

| Component | Module | Responsibility |
|---|---|---|
| Boundary types | `models.py` | `ToolSpec`, `ToolCall`, messages, `LLMResponse`. The only shapes that cross the LLM boundary. |
| LLM protocol | `llm/client.py` | `LLMClient.generate(system, messages, tools) -> LLMResponse`, `LLMError`. |
| Claude adapter | `llm/claude.py` | The only module that imports `anthropic`. Maps to and from the Messages API. |
| Test doubles | `llm/fake.py` | `ScriptedLLM` (canned responses) and `OfflineLLM` (deterministic rule-based stand-in). |
| Agent loop | `agent/loop.py` | Ask the model, run the tools it selects, feed back observations, repeat. Produces an `AgentRun`. |
| Tool catalog | `tools/catalog.py` | Discovers `tools/*/tool.toml`, exposes tool specs, executes tools safely. |
| Evals | `evals/` | Load TOML cases, run them against the agent or a single tool, score the checks. |
| Flywheel | `flywheel/loop.py` | Record each eval run, diff it against the previous one, and group failures by mode. |
| Learning | `learning/` | Plan rules, the approval gate, tool author, runtime eval harness, sandbox, local store. Optional: the loop works without it. |
| CLI | `cli.py` | `ask`, `chat`, `eval`, `flywheel`, `tools`. Reads settings once and wires everything together. |

## Why the boundaries exist

**LLM boundary (`LLMClient`).** The agent must be testable without a network and must not depend on one vendor. Tool calls are first-class in the boundary types (`ToolCall`, `ToolResultMessage`), not parsed out of text, so native tool use works without a rewrite. Vendor state that has to round-trip, such as Claude's thinking blocks, rides in `AssistantMessage.provider_state`, which is opaque to everything except the adapter that created it.

**Tool boundary (`ToolCatalog`).** The loop never knows how a tool works. Future web, flight, browser or MCP-backed tools plug in behind the same `execute(name, args) -> ToolResult`. Adding a tool means adding a manifest and a function; the loop does not change. Expected tool failures (`ToolError`) become error observations the model can react to. Bugs still raise.

**Capability gaps (`request_capability`).** The loop always offers one built-in tool through which the model can say: "this task needs a capability none of my tools provide." The gap is recorded in the trace, so evals can check for it, and the model is told to answer honestly instead of guessing. This is the seam for auto-discovery: in a later stage, handling `request_capability` can mean searching a wider catalog and loading a matching tool, with no change to the loop's shape.

**Learning (`propose_tool_plan`).** With a `ToolLearner`, the loop offers a second built-in tool through which the model can propose building generic tools. The model proposes; deterministic rules, the user (twice) and the runtime harness decide. A learned tool enters the catalog as an ordinary tool whose function runs generated code in the sandbox, so the catalog boundary (`execute(name, args) -> ToolResult`) is unchanged. See [tools.md § Learned tools](tools.md#learned-tools).

**Conversation history.** `Agent.run(task, history=...)` resends earlier turns before the new task (`AgentRun.messages` is the history for the next turn), because the model API is stateless. History is append-only. Provider-private state from earlier turns, such as Claude's thinking blocks, is not replayed: it is bound to the tool list, which changes when a tool is learned.

**The trace is the evaluation surface (`AgentRun`).** Evals check *behavior*: which tools were used, which gaps were reported and how the run stopped. They don't just check the final string. That is what makes "did the agent recognize it needed fresh information?" testable.

**Evaluation is part of the system, not an add-on.** Every behavior worth having gets an eval case. The flywheel turns eval runs into a failure summary that drives the next change. See [evaluation.md](evaluation.md).

## Replaceability check

| Question | Answer |
|---|---|
| Replace Claude? | Yes: implement `LLMClient`. Nothing outside `llm/claude.py` imports the SDK. |
| Replace a tool? | Yes: point its manifest at a different implementation. |
| Add a tool without touching the core? | Yes: add `tools/<name>/tool.toml` plus a function. |
| Test without the internet? | Yes: `OfflineLLM` + deterministic tools, used by CI. |
| Evaluate agent behavior automatically? | Yes: agent cases check the trace. |

## Security principles for future web tools

Web content and tool results are **untrusted input**. This matters as soon as Stage 4 adds web research:

- **Prompt injection.** Pages can contain text written to look like instructions. Tool output is data. It goes back to the model as a tool result, never as a system or user instruction.
- **Malicious pages.** Fetch with allowlists or blocklists where possible. Never execute content from a page.
- **External side effects.** Tools that act on the world (submitting forms, booking, sending) need an explicit human approval gate. The bootstrap has none, and flight research starts read-only: search, compare and recommend, with no booking.
- **Authentication boundaries.** Tools don't share credentials with the model. Secrets stay in config (`Secret`) and are never placed in prompts or logs.
- **Sensitive user data.** Travel preferences and personal details go only to the tools that need them.
- **Generated code.** Learned tools are model-written code. While agentlab is a single-user toy they run **unrestricted**: any import, the user's files, environment (including API keys), network and commands. What protects the user is the runtime harness and their approval of the plan, which says the tool runs with full access. The code is not shown before use; it is saved in `.agentlab/learned/`. They run in their own process with a timeout, so a crash or hang can't take the agent down. Before anyone else uses agentlab, restore per-tool permissions with OS-level enforcement ([roadmap](roadmap.md)).

Anthropic's browser-use guidance warns specifically that web pages can carry prompt injections, and recommends isolating sensitive data and actions and keeping approval controls. Treat it as a requirement for any browser tool.

## Deliberately absent

These are absent on purpose: general planning (the only planning is the small, rule-checked tool plan), long-term memory (chat resends the current conversation and nothing persists between sessions), recursion, and long-running autonomy; MCP; browser automation; databases and dashboards. **MCP** is a plausible future *transport* for tools: an MCP-backed tool would sit behind the same `ToolCatalog` boundary. It gets adopted once the tool boundary has proven stable, not before.
