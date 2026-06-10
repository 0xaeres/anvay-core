"""LLM client - thin OpenAI-compatible HTTP wrapper with multi-provider routing.

Every council role goes through `ChatClient.from_role(config, role)`. The provider
field decides the base URL and auth header; the model field decides the request
body. DeepInfra council clients stream token deltas for prose/markdown calls while
still returning a complete response to callers. JSON-mode calls stay non-streamed
by default because they are machine-parsed control messages.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from nexus.config import ModelCfg
from nexus.llm.tracing import record_generation

log = logging.getLogger(__name__)

TokenSink = Callable[[dict[str, str]], Awaitable[None]]

# Provider → base URL. Override with model.base_url / model.url in nexus.yaml.
_PROVIDER_BASES: dict[str, str] = {
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama": "http://localhost:11434/v1",
}


@dataclass(frozen=True)
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion


@dataclass
class ChatResponse:
    content: str
    usage: TokenUsage
    model: str
    finish_reason: str = "stop"  # "stop" | "length" | "content_filter" | other
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True when the response hit max_tokens before the model would have stopped."""
        return self.finish_reason == "length"


class LLMError(RuntimeError):
    pass


class ChatClient:
    """Async chat client. Construct one per role for clean cost attribution."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        role: str,
        timeout_s: float = 300.0,
        stream_chat: bool = False,
        token_sink: TokenSink | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        trace_context: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.role = role
        self._stream_chat = stream_chat
        self._token_sink = token_sink
        self.temperature = temperature
        self.top_p = top_p
        self.trace_context = trace_context or {}
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout_s)

    @classmethod
    def from_cfg(
        cls,
        cfg: ModelCfg,
        *,
        role: str,
        token_sink: TokenSink | None = None,
        trace_context: dict[str, Any] | None = None,
    ) -> ChatClient:
        provider = cfg.provider.lower()
        base = cfg.base_url or cfg.url or _PROVIDER_BASES.get(provider)
        if not base:
            raise LLMError(f"no base URL known for provider={provider}")
        return cls(
            provider=provider,
            model=cfg.model,
            base_url=base,
            api_key=cfg.api_key,
            role=role,
            stream_chat=provider == "deepinfra",
            token_sink=token_sink,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            trace_context=trace_context,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        stream: bool | None = None,
    ) -> ChatResponse:
        """OpenAI-compatible /chat/completions. Returns the assistant content."""
        request_temperature = self.temperature if temperature is None else temperature
        request_top_p = self.top_p if top_p is None else top_p
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request_temperature,
            "max_tokens": max_tokens,
        }
        if request_top_p is not None:
            body["top_p"] = request_top_p
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        should_stream = stream is True or (
            self._stream_chat and (stream if stream is not None else not json_mode)
        )
        start = time.perf_counter()
        try:
            if should_stream:
                try:
                    resp = await self._chat_stream(body)
                except LLMError as e:
                    log.warning(
                        "%s: streaming chat failed; retrying without stream: %s",
                        self.role,
                        e,
                    )
                    resp = await self._chat_non_stream(body)
            else:
                resp = await self._chat_non_stream(body)
        except Exception as e:
            self._trace(messages, None, TokenUsage(), start, error=str(e))
            raise
        self._trace(messages, resp.content, resp.usage, start, finish_reason=resp.finish_reason)
        return resp

    def _trace(
        self,
        messages: list[dict[str, str]],
        output: str | None,
        usage: TokenUsage,
        start: float,
        *,
        finish_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        record_generation(
            name=self.role,
            model=self.model,
            provider=self.provider,
            messages=messages,
            output=output,
            usage={"prompt": usage.prompt, "completion": usage.completion},
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=finish_reason,
            error=error,
            metadata=self.trace_context,
        )

    async def _chat_non_stream(self, body: dict[str, Any]) -> ChatResponse:
        try:
            resp = await self._client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"{self.role}: chat call failed: {e}") from e
        if resp.status_code != 200:
            raise LLMError(
                f"{self.role}: chat returned {resp.status_code}: {resp.text[:200]}"
            )
        payload = resp.json()
        choices = payload.get("choices", [])
        if not choices:
            raise LLMError(f"{self.role}: empty choices in response")
        content = choices[0].get("message", {}).get("content", "")
        finish_reason = str(choices[0].get("finish_reason") or "stop").lower()
        usage_obj = payload.get("usage", {}) or {}
        return ChatResponse(
            content=content or "",
            usage=TokenUsage(
                prompt=int(usage_obj.get("prompt_tokens", 0)),
                completion=int(usage_obj.get("completion_tokens", 0)),
            ),
            model=self.model,
            finish_reason=finish_reason,
            raw=payload,
        )

    async def _chat_stream(self, body: dict[str, Any]) -> ChatResponse:
        """OpenAI-compatible streaming chat; collect text while emitting deltas."""
        stream_body = {
            **body,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        content_parts: list[str] = []
        finish_reason = "stop"
        usage = TokenUsage()
        raw_chunks: list[dict[str, Any]] = []
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=stream_body,
            ) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise LLMError(
                        f"{self.role}: streaming chat returned {resp.status_code}: "
                        f"{text.decode('utf-8', errors='replace')[:200]}"
                    )
                async for line in resp.aiter_lines():
                    payload = _parse_sse_line(line)
                    if payload is None:
                        continue
                    raw_chunks.append(payload)
                    usage_obj = payload.get("usage") or {}
                    if usage_obj:
                        usage = TokenUsage(
                            prompt=int(usage_obj.get("prompt_tokens", usage.prompt)),
                            completion=int(
                                usage_obj.get("completion_tokens", usage.completion)
                            ),
                        )
                    for choice in payload.get("choices", []) or []:
                        choice_finish = choice.get("finish_reason")
                        if choice_finish:
                            finish_reason = str(choice_finish).lower()
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        if not text:
                            continue
                        content_parts.append(text)
                        await self._emit_token(text)
        except httpx.HTTPError as e:
            raise LLMError(f"{self.role}: streaming chat call failed: {e}") from e

        return ChatResponse(
            content="".join(content_parts),
            usage=usage,
            model=self.model,
            finish_reason=finish_reason,
            raw={"stream": raw_chunks},
        )

    async def _emit_token(self, text: str) -> None:
        if self._token_sink is None:
            return
        try:
            await self._token_sink(
                {
                    "role": self.role,
                    "model": self.model,
                    "provider": self.provider,
                    "text": text,
                }
            )
        except Exception:
            log.warning("token sink failed for role=%s", self.role, exc_info=True)

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> tuple[Any, TokenUsage]:
        """Convenience: ask for JSON, parse it. Falls back to extracting the first JSON
        object from the text if `response_format` isn't honoured by the provider."""
        resp = await self.chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            json_mode=True,
            stream=stream,
        )
        try:
            return _parse_json_payload(resp.content), resp.usage
        except LLMError:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": resp.content},
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid complete JSON. "
                        "Return only a valid JSON object that satisfies the requested "
                        "schema. Keep all string fields concise. Do not include Markdown, "
                        "fences, commentary, or repeated prompt text."
                    ),
                },
            ]
            repaired = await self.chat(
                repair_messages,
                temperature=0.0,
                top_p=top_p,
                max_tokens=max_tokens,
                json_mode=True,
                stream=False,
            )
            usage = TokenUsage(
                prompt=resp.usage.prompt + repaired.usage.prompt,
                completion=resp.usage.completion + repaired.usage.completion,
            )
            return _parse_json_payload(repaired.content), usage

    async def chat_markdown(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 2048,
        max_continuations: int = 2,
    ) -> ChatResponse:
        """Long-form markdown with auto-continuation on finish_reason=='length'.

        The aider/cursor pattern: when the API returns length-truncated, send the
        partial content back as an assistant message and ask the model to continue
        from exactly where it stopped. Combines the chunks into a single response.
        Token usage is summed across continuations.
        """
        resp = await self.chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            json_mode=False,
        )
        if not resp.truncated:
            return resp

        combined = resp.content
        total_prompt = resp.usage.prompt
        total_completion = resp.usage.completion
        finish = resp.finish_reason

        for _ in range(max_continuations):
            continuation_messages = [
                *messages,
                {"role": "assistant", "content": combined},
                {
                    "role": "user",
                    "content": (
                        "Continue exactly where you stopped. Do not repeat any prior "
                        "text. Do not add any preamble. Resume mid-sentence if that's "
                        "where you stopped. End cleanly when the document is complete."
                    ),
                },
            ]
            next_resp = await self.chat(
                continuation_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                json_mode=False,
            )
            combined += next_resp.content
            total_prompt += next_resp.usage.prompt
            total_completion += next_resp.usage.completion
            finish = next_resp.finish_reason
            if not next_resp.truncated:
                break

        return ChatResponse(
            content=combined,
            usage=TokenUsage(prompt=total_prompt, completion=total_completion),
            model=self.model,
            finish_reason=finish,
            raw=resp.raw,
        )


def _parse_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to extract a JSON object from a fenced or noisy response
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise LLMError(f"failed to parse JSON from model output: {text[:200]!r}")


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise LLMError(f"failed to parse streaming chat chunk: {data[:200]!r}") from e
