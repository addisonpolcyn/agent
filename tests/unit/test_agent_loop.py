"""State transitions of the agent loop, driven by scripted model responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentlab.agent.loop import LEARNING_PROMPT, REQUEST_CAPABILITY, SYSTEM_PROMPT, Agent
from agentlab.learning.approval import FixedApprover
from agentlab.learning.author import SkillAuthor
from agentlab.learning.learner import SkillLearner
from agentlab.learning.plan import PROPOSE_SKILL_PLAN
from agentlab.llm.fake import FixtureAuthorLLM, ScriptedLLM
from agentlab.models import AssistantMessage, LLMResponse, ToolCall, ToolResultMessage, UserMessage
from agentlab.skills.catalog import SkillCatalog

if TYPE_CHECKING:
    from pathlib import Path

    from agentlab.models import JSONObject


def call(name: str, arguments: JSONObject, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(text=None, tool_calls=(ToolCall(call_id, name, arguments),))


def answer(text: str) -> LLMResponse:
    return LLMResponse(text=text)


def test_direct_answer_takes_one_step(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([answer("hello")])
    run = Agent(llm, catalog).run("Say hello")

    assert run.answer == "hello"
    assert run.stop_reason == "answered"
    assert run.steps == 1
    assert run.skills_used == ()
    (first,) = llm.calls
    assert first.system == SYSTEM_PROMPT
    assert first.messages == (UserMessage("Say hello"),)


def test_offers_catalog_skills_plus_request_capability(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([answer("ok")])
    Agent(llm, catalog).run("anything")
    assert [t.name for t in llm.calls[0].tools] == ["calculator", REQUEST_CAPABILITY.name]


def test_tool_call_then_answer(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([call("calculator", {"expression": "120 * 3"}), answer("360")])
    run = Agent(llm, catalog).run("What is 120 * 3?")

    assert run.answer == "360"
    assert run.steps == 2
    assert run.skills_used == ("calculator",)
    assert run.invocations[0].result.output == {"result": 360}
    second = llm.calls[1].messages
    assert isinstance(second[1], AssistantMessage)
    assert second[2] == ToolResultMessage("c1", {"result": 360}, is_error=False)


def test_skill_error_is_fed_back_as_error_observation(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([call("calculator", {"expression": "1 / 0"}), answer("can't")])
    run = Agent(llm, catalog).run("1/0?")

    assert run.invocations[0].result.error == "division by zero"
    assert llm.calls[1].messages[-1] == ToolResultMessage(
        "c1", {"error": "division by zero"}, is_error=True
    )


def test_unknown_tool_is_reported_not_raised(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([call("teleport", {}), answer("sorry")])
    run = Agent(llm, catalog).run("beam me up")

    assert run.answer == "sorry"
    observation = llm.calls[1].messages[-1]
    assert isinstance(observation, ToolResultMessage)
    assert observation.is_error


def test_capability_gap_is_recorded(catalog: SkillCatalog) -> None:
    gap = {"capability": "current_information", "reason": "needs live flights"}
    llm = ScriptedLLM([call(REQUEST_CAPABILITY.name, gap), answer("I can't search the web.")])
    run = Agent(llm, catalog).run("Find flights")

    assert run.skills_used == ()
    assert [(g.capability, g.reason) for g in run.capability_gaps] == [
        ("current_information", "needs live flights")
    ]
    observation = llm.calls[1].messages[-1]
    assert isinstance(observation, ToolResultMessage)
    assert observation.content["available"] is False


def test_multiple_tool_calls_in_one_turn(catalog: SkillCatalog) -> None:
    turn = LLMResponse(
        text=None,
        tool_calls=(
            ToolCall("a", "calculator", {"expression": "1 + 1"}),
            ToolCall("b", "calculator", {"expression": "2 + 2"}),
        ),
    )
    llm = ScriptedLLM([turn, answer("2 and 4")])
    run = Agent(llm, catalog).run("two sums")

    assert [i.result.output for i in run.invocations] == [{"result": 2}, {"result": 4}]
    assert [m.call_id for m in llm.calls[1].messages if isinstance(m, ToolResultMessage)] == [
        "a",
        "b",
    ]


def test_stops_after_max_steps(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([call("calculator", {"expression": "1 + 1"}, f"c{i}") for i in range(3)])
    run = Agent(llm, catalog, max_steps=3).run("loop forever")

    assert run.stop_reason == "max_steps"
    assert run.answer is None
    assert run.steps == 3
    assert len(run.invocations) == 3


def test_reserved_skill_name_is_rejected(catalog: SkillCatalog) -> None:
    class Reserved(SkillCatalog):
        def __contains__(self, name: object) -> bool:
            return name == REQUEST_CAPABILITY.name

    with pytest.raises(ValueError, match="reserved"):
        Agent(ScriptedLLM([]), Reserved({}))


PLAN: JSONObject = {
    "goal": "Count words",
    "steps": [
        {
            "summary": "Count the words in a text",
            "capability": "text_statistics",
            "new_skill": {
                "name": "word_count",
                "purpose": "Count words.",
                "inputs": "text",
                "outputs": "count",
            },
        }
    ],
}


def learning_agent(llm: ScriptedLLM, catalog: SkillCatalog, store: Path) -> Agent:
    learner = SkillLearner(SkillAuthor(FixtureAuthorLLM()), store)
    return Agent(llm, catalog, learner=learner)


def test_learning_is_offered_only_with_a_learner(catalog: SkillCatalog, tmp_path: Path) -> None:
    plain = ScriptedLLM([answer("ok")])
    Agent(plain, catalog).run("x")
    learning = ScriptedLLM([answer("ok")])
    learning_agent(learning, catalog, tmp_path).run("x")

    assert PROPOSE_SKILL_PLAN.name not in [t.name for t in plain.calls[0].tools]
    assert [t.name for t in learning.calls[0].tools][-1] == PROPOSE_SKILL_PLAN.name
    assert learning.calls[0].system == SYSTEM_PROMPT + LEARNING_PROMPT


def test_learned_skill_is_used_in_a_fresh_append_only_conversation(
    catalog: SkillCatalog, tmp_path: Path
) -> None:
    llm = ScriptedLLM(
        [
            call(PROPOSE_SKILL_PLAN.name, PLAN),
            call("word_count", {"text": "a b c"}, "c2"),
            answer("3 words, using the newly learned word_count skill."),
        ]
    )
    run = learning_agent(llm, catalog, tmp_path).run(
        "count the words in 'a b c'", approver=FixedApprover(plan=True)
    )

    assert run.skills_learned == ("word_count",)
    assert run.skills_used == ("word_count",)
    assert run.invocations[0].result.output == {"count": 3}
    restart = llm.calls[1]
    assert len(restart.messages) == 1, "earlier turns are never edited, only left behind"
    (first,) = restart.messages
    assert isinstance(first, UserMessage)
    assert first.text.startswith("count the words in 'a b c'")
    assert "word_count" in first.text
    assert "word_count" in [t.name for t in restart.tools]


def test_learning_is_declined_without_an_approver(catalog: SkillCatalog, tmp_path: Path) -> None:
    llm = ScriptedLLM([call(PROPOSE_SKILL_PLAN.name, PLAN), answer("I can't without approval.")])
    run = learning_agent(llm, catalog, tmp_path).run("count the words in 'a b'")

    assert [o.outcome for o in run.learning] == ["declined_plan"]
    observation = llm.calls[1].messages[-1]
    assert isinstance(observation, ToolResultMessage)
    assert observation.is_error
    assert observation.content["outcome"] == "declined_plan"


def test_propose_skill_plan_is_reserved(catalog: SkillCatalog) -> None:
    class Reserved(SkillCatalog):
        def __contains__(self, name: object) -> bool:
            return name == PROPOSE_SKILL_PLAN.name

    with pytest.raises(ValueError, match="reserved"):
        Agent(ScriptedLLM([]), Reserved({}))


def test_history_is_sent_before_the_new_task(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM(
        [call("calculator", {"expression": "1234 * 5"}), answer("6170"), answer("7170")]
    )
    agent = Agent(llm, catalog)
    first = agent.run("What is 1234 * 5?")
    agent.run("Now add 1000 to that result.", history=first.messages)

    sent = llm.calls[2].messages
    assert sent[: len(first.messages)] == first.messages, "history is resent, never rewritten"
    assert sent[-1] == UserMessage("Now add 1000 to that result.")


def test_history_drops_provider_state(catalog: SkillCatalog) -> None:
    earlier = (UserMessage("hi"), AssistantMessage("hello", (), provider_state=object()))
    llm = ScriptedLLM([answer("ok")])
    Agent(llm, catalog).run("again", history=earlier)

    replayed = llm.calls[0].messages[1]
    assert isinstance(replayed, AssistantMessage)
    assert replayed.text == "hello"
    assert replayed.provider_state is None
