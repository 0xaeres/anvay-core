"""The assistant agent loop — the "brain" — see docs/ASSISTANT-LAYER.md §7.

A sequential tool-calling loop: the LLM either calls one curated tool or emits a
final answer; tool results are fed back; repeat until a final answer or the
iteration cap. Implemented as a plain async loop rather than a LangGraph graph —
the assistant turn is strictly sequential (LLM ↔ tool), so LangGraph's
fan-out/fan-in machinery (which the council genuinely needs) would be ceremony
here. State that must survive a crash lives in `AssistantStore`, not in memory.

Tool calls use the JSON-action convention already used across Nexus (the council
parses JSON from text too) — no native tool-calling support is required from
`ChatClient`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from nexus.assistant.capabilities import Tool, ToolContext
from nexus.assistant.models import ActionProposal, ConversationMessage

log = logging.getLogger(__name__)

# Optional progress sink — the streaming API passes one to push live events
# (tool_call / tool_result) to the client. None for the sync / MCP callers.
EventSink = Callable[[dict], Awaitable[None]]

_SYSTEM_TEMPLATE = """You are the Nexus Assistant for product "{product_id}".
You help engineers query and act on their Jira and Confluence, grounded in the
product's indexed codebase and curated skills.

You work by calling tools. On EACH step respond with EXACTLY ONE JSON object,
nothing else:
  {{"action": "call_tool", "tool": "<name>", "args": {{...}}}}
to call a tool, or
  {{"action": "final", "answer": "<markdown answer for the user>"}}
to finish the turn.

Rules:
- To change anything in Jira/Confluence you MUST use a propose_* tool. A propose_*
  tool only DRAFTS a plan — it does not apply anything. A human confirms it
  separately. NEVER tell the user a change was made; say it has been drafted for
  their confirmation.
- Use search_corpus / find_skills for broad questions about the codebase.
- Use get_jira_issue / get_confluence_page for specific, current items.
- Cite issue keys and page titles in your final answer.
- If a tool reports it is unavailable, say so plainly instead of guessing.

Available tools:
{tool_block}
"""


@dataclass
class TurnResult:
    reply: str
    tool_calls: list[dict] = field(default_factory=list)
    action_proposal: ActionProposal | None = None
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AssistantLoop:
    def __init__(
        self,
        *,
        llm: Any,  # ChatClient-like: async chat_json(messages) -> (parsed, usage)
        registry: dict[str, Tool],
        max_iterations: int = 6,
    ):
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations

    def _system_prompt(self, product_id: str) -> str:
        tool_block = "\n".join(
            f"- {t.name}({', '.join(t.parameters.get('required', []))}): {t.description}"
            for t in self.registry.values()
        )
        return _SYSTEM_TEMPLATE.format(product_id=product_id, tool_block=tool_block)

    async def run_turn(
        self,
        *,
        ctx: ToolContext,
        history: list[ConversationMessage],
        user_text: str,
        on_event: EventSink | None = None,
    ) -> TurnResult:
        async def emit(ev: dict) -> None:
            if on_event is not None:
                await on_event(ev)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt(ctx.product_id)}
        ]
        for m in history:
            role = "assistant" if m.role.value == "assistant" else "user"
            messages.append({"role": role, "content": m.content})
        messages.append({"role": "user", "content": user_text})

        result = TurnResult(reply="")
        for i in range(self.max_iterations):
            result.iterations = i + 1
            parsed, usage = await self.llm.chat_json(messages)
            result.prompt_tokens += getattr(usage, "prompt", 0)
            result.completion_tokens += getattr(usage, "completion", 0)

            if not isinstance(parsed, dict):
                messages.append(
                    {"role": "user", "content": "Respond with one valid JSON action object."}
                )
                continue

            action = parsed.get("action")

            if action == "final":
                result.reply = str(parsed.get("answer", "")).strip()
                return result

            if action == "call_tool":
                tool_name = parsed.get("tool", "")
                args = parsed.get("args") or {}
                tool = self.registry.get(tool_name)
                await emit({"type": "tool_call", "tool": tool_name, "args": args})
                if tool is None:
                    obs: dict = {"error": f"unknown tool: {tool_name!r}"}
                else:
                    try:
                        obs = await tool.handler(ctx, **args)
                    except TypeError as e:
                        obs = {"error": f"bad arguments for {tool_name}: {e}"}
                    except Exception as e:
                        log.warning("assistant tool %s failed: %s", tool_name, e)
                        obs = {"error": f"tool {tool_name} failed: {e}"}

                ok = not (isinstance(obs, dict) and "error" in obs)
                await emit({"type": "tool_result", "tool": tool_name, "ok": ok})
                result.tool_calls.append({"tool": tool_name, "args": args, "result": obs})
                if isinstance(obs, dict) and obs.get("proposal_id"):
                    result.action_proposal = ctx.store.get_proposal(obs["proposal_id"])

                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL RESULT ({tool_name}): "
                        + json.dumps(obs)[:4000],
                    }
                )
                continue

            messages.append(
                {
                    "role": "user",
                    "content": 'Respond with {"action":"call_tool",...} or '
                    '{"action":"final",...}.',
                }
            )

        result.reply = (
            "I wasn't able to finish within the tool-call budget. "
            "Please narrow the request and try again."
        )
        return result
