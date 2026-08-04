"""LLM provider abstraction returning schema-validated JSON.

Every provider exposes the same `complete_json` contract. The important difference is
how each one is coerced into valid JSON:

* OpenAI    -- native strict structured outputs, the most reliable option.
* Anthropic -- a forced tool call, whose input schema does the same job.
* Ollama    -- only has a loose JSON mode, so the schema goes in the prompt and we
               validate and retry locally. Weakest of the three on wide schemas.

A silent empty result is the worst possible outcome for this pipeline, so parse and
validation failures raise instead of returning blank fields.

Providers with a hosted web search tool additionally implement `search_json`, which
grounds an answer in live pages and returns the URLs consulted. Only OpenAI and
Anthropic offer this; the others report `supports_search = False` so callers can degrade
instead of failing.
"""

from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator

PROVIDERS = ("openai", "anthropic", "deepseek", "ollama")

SUGGESTED_MODELS = {
    "openai": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
    "anthropic": ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "ollama": ["llama3.1:8b", "qwen2.5:14b", "mistral-nemo"],
}

# Providers whose JSON is guaranteed by the API itself rather than by local validation.
# The rest get the schema in the prompt plus a validate-and-repair loop.
STRICT_SCHEMA_PROVIDERS = ("openai", "anthropic")

# Providers with a hosted web search tool, usable for web-grounded conflict resolution.
SEARCH_PROVIDERS = ("openai", "anthropic")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    data: dict
    raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    attempts: int = 1
    # URLs the hosted search tool consulted, and how many searches it ran. Both stay
    # empty for ordinary completions.
    citations: list[str] = field(default_factory=list)
    searches: int = 0


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    cached: int = 0
    searches: int = 0  # billed separately from tokens by both vendors

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.searches += r.searches


# Rough USD per 1M tokens (input, output), for an order-of-magnitude estimate only.
# These drift constantly -- DeepSeek in particular has repriced several times and applies
# a large cache-hit discount that is not modelled here. Treat as indicative, not billing.
# Longest key wins in `estimate_cost`, so prefixes like "gpt-5.6" cannot shadow a more
# specific tier.
PRICES = {
    # OpenAI (GPT-5.6 generation)
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-luna": (1.00, 6.00),
    # Anthropic (Claude 5 generation)
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # DeepSeek (V4 generation, cache-miss input pricing)
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
}


# USD per hosted web search call, charged on top of tokens. Anthropic publishes $10 per
# 1,000 searches; OpenAI prices per 1,000 calls in the same ballpark.
SEARCH_PRICES = {"openai": 0.010, "anthropic": 0.010}


def estimate_search_cost(provider: str, searches: int) -> float | None:
    price = SEARCH_PRICES.get((provider or "").lower())
    return None if price is None else price * searches


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    # Match the most specific known prefix, so "gpt-5.6-terra" is not priced as a
    # shorter "gpt-5.6" entry that happens to be declared first.
    best: tuple[float, float] | None = None
    best_len = -1
    for name, price in PRICES.items():
        if model.startswith(name) and len(name) > best_len:
            best, best_len = price, len(name)
    if best is None:
        return None
    pin, pout = best
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout


def extract_json_object(text: str) -> dict:
    """Best-effort recovery of a JSON object from a chatty response."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Scan for the first balanced {...} block.
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise LLMError(f"Response contained no parsable JSON object: {t[:250]!r}")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(
        s in msg
        for s in (
            "rate limit",
            "429",
            "overloaded",
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "502",
            "503",
            "504",
            "internal server error",
        )
    ):
        return True
    return type(exc).__name__ in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "APIStatusError",
    }


class BaseProvider(ABC):
    name = "base"
    # True only where the vendor runs the search server-side and returns citations.
    supports_search = False

    def __init__(self, api_key: str, base_url: str = "", model: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @abstractmethod
    def _call(
        self, system: str, user: str, schema: dict, max_tokens: int, temperature: float
    ) -> LLMResponse: ...

    def search_json(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int = 4000,
        max_searches: int = 4,
    ) -> LLMResponse:
        """Answer `user` using live web search, returning JSON matching `schema`.

        Search tools and strict structured output are mutually exclusive on both vendors,
        so the schema travels in the prompt and is validated here instead. `citations`
        carries the URLs consulted, which the caller needs in order to attribute the
        answer.
        """
        raise LLMError(
            f"{self.name} has no hosted web search tool. Use OpenAI or Anthropic for "
            "web-grounded conflict resolution."
        )

    def _validate_searched(self, resp: LLMResponse, schema: dict) -> LLMResponse:
        errors = sorted(
            Draft202012Validator(schema).iter_errors(resp.data), key=lambda e: e.path
        )
        if errors:
            detail = "; ".join(
                f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                for e in errors[:3]
            )
            raise LLMError(f"Search response failed schema validation: {detail}")
        return resp

    @staticmethod
    def _search_system(system: str, schema: dict) -> str:
        return (
            f"{system}\n\n"
            "OUTPUT FORMAT: after searching, reply with a single raw JSON object and "
            "nothing else -- no prose, no markdown fences. It must validate against this "
            "JSON Schema, with every required key present:\n"
            f"{json.dumps(schema, separators=(',', ':'))}"
        )

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int = 8000,
        temperature: float = 0.1,
        retries: int = 3,
    ) -> LLMResponse:
        validator = Draft202012Validator(schema)
        last: Exception | None = None
        attempt_user = user

        for attempt in range(1, retries + 1):
            try:
                resp = self._call(system, attempt_user, schema, max_tokens, temperature)
                errors = sorted(validator.iter_errors(resp.data), key=lambda e: e.path)
                if errors:
                    detail = "; ".join(
                        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                        for e in errors[:3]
                    )
                    raise LLMError(f"Response failed schema validation: {detail}")
                resp.attempts = attempt
                return resp
            except Exception as e:  # noqa: BLE001 - classified below
                last = e
                if attempt >= retries or not (
                    _is_retryable(e) or isinstance(e, LLMError)
                ):
                    break
                # A malformed or invalid response will usually repeat if we resend the
                # identical prompt, so tell the model what was wrong and ask it to fix
                # that specific problem. Providers without strict schema support depend
                # on this to be usable at all on a wide schema.
                if isinstance(e, LLMError):
                    attempt_user = (
                        f"{user}\n\n---\nYour previous reply was rejected: {e}\n"
                        "Return the corrected JSON object only. It must match the "
                        "required schema exactly, including every required key."
                    )
                time.sleep(min(2**attempt + random.random(), 20))
        raise LLMError(
            f"{self.name} failed after {retries} attempt(s): {last}"
        ) from last

    def smoke_test(self) -> str:
        """Cheap connectivity check for the UI."""
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        r = self.complete_json(
            "You reply only with JSON.",
            'Reply with {"ok": true}',
            schema,
            max_tokens=100,
            temperature=0,
            retries=2,
        )
        return f"{self.name}/{self.model} OK (data={r.data})"


class OpenAIProvider(BaseProvider):
    """OpenAI via Chat Completions with strict structured outputs.

    The current generation (gpt-5.x, o-series) are reasoning models: they reject the
    deprecated `max_tokens` in favour of `max_completion_tokens`, ignore or reject
    `temperature`, and spend part of the output budget on hidden reasoning. Extraction is
    a mechanical read-and-fill task, so reasoning effort is pinned low to keep that
    budget for the JSON itself.
    """

    name = "openai"
    supports_search = True
    REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

    def _client(self):
        from openai import OpenAI

        if not self.api_key:
            raise LLMError("OpenAI API key is missing.")
        return OpenAI(
            api_key=self.api_key, base_url=self.base_url or None, timeout=300.0
        )

    @property
    def is_reasoning(self) -> bool:
        return (self.model or "").lower().startswith(self.REASONING_PREFIXES)

    def _budget_kwargs(self, max_tokens: int, temperature: float) -> dict:
        if self.is_reasoning:
            return {
                "max_completion_tokens": max_tokens,
                "reasoning_effort": "low",
            }
        return {"max_tokens": max_tokens, "temperature": temperature}

    def _call(self, system, user, schema, max_tokens, temperature) -> LLMResponse:
        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": schema},
            },
            **self._budget_kwargs(max_tokens, temperature),
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(
                "Output hit the token ceiling and the JSON is truncated. Raise "
                "'Max output tokens' or lower the batch size."
            )
        content = choice.message.content or ""
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"Model refused the request: {choice.message.refusal}")
        u = resp.usage
        return LLMResponse(
            data=extract_json_object(content),
            raw=content,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            model=self.model,
        )

    def search_json(
        self, system, user, schema, max_tokens=4000, max_searches=4
    ) -> LLMResponse:
        # The hosted web_search tool lives on the Responses API only, and cannot be
        # combined with a strict json_schema text format, so the schema goes in the
        # prompt and is validated below.
        kwargs: dict = {
            "model": self.model,
            "instructions": self._search_system(system, schema),
            "input": user,
            "tools": [{"type": "web_search"}],
            "tool_choice": "auto",
            "max_output_tokens": max_tokens,
        }
        if self.is_reasoning:
            kwargs["reasoning"] = {"effort": "low"}

        resp = self._client().responses.create(**kwargs)

        text = (getattr(resp, "output_text", "") or "").strip()
        citations: list[str] = []
        searches = 0
        for item in getattr(resp, "output", []) or []:
            kind = getattr(item, "type", "")
            if kind == "web_search_call":
                searches += 1
            elif kind == "message":
                for block in getattr(item, "content", []) or []:
                    for ann in getattr(block, "annotations", []) or []:
                        url = getattr(ann, "url", "")
                        if url and url not in citations:
                            citations.append(url)

        if not text:
            raise LLMError("Search returned no text output.")
        u = getattr(resp, "usage", None)
        return self._validate_searched(
            LLMResponse(
                data=extract_json_object(text),
                raw=text,
                prompt_tokens=getattr(u, "input_tokens", 0) or 0,
                completion_tokens=getattr(u, "output_tokens", 0) or 0,
                model=self.model,
                citations=citations,
                searches=searches,
            ),
            schema,
        )


class AnthropicProvider(BaseProvider):
    """Anthropic via a forced tool call, whose input schema guarantees the shape.

    Claude 5 and the late 4.x models reject any non-default `temperature`, `top_p` or
    `top_k` outright, and think by default -- so sampling is omitted for them and spend is
    steered with `output_config.effort` instead, kept low because extraction is a
    read-and-fill task rather than a reasoning one.
    """

    name = "anthropic"
    supports_search = True
    # Models that reject non-default sampling parameters and accept `effort`.
    NO_SAMPLING_PREFIXES = (
        "claude-fable-5",
        "claude-mythos",
        "claude-opus-5",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-5",
    )

    def _client(self):
        import anthropic

        if not self.api_key:
            raise LLMError("Anthropic API key is missing.")
        return anthropic.Anthropic(api_key=self.api_key, timeout=300.0)

    @property
    def rejects_sampling(self) -> bool:
        return (self.model or "").lower().startswith(self.NO_SAMPLING_PREFIXES)

    def _sampling_kwargs(self, temperature: float) -> dict:
        if self.rejects_sampling:
            return {"output_config": {"effort": "low"}}
        return {"temperature": temperature}

    def _call(self, system, user, schema, max_tokens, temperature) -> LLMResponse:
        resp = self._client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": "emit_extraction",
                    "description": "Return the extracted fields.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "emit_extraction"},
            **self._sampling_kwargs(temperature),
        )
        if resp.stop_reason == "max_tokens":
            raise LLMError(
                "Output hit the token ceiling and the JSON is truncated. Raise "
                "'Max output tokens' or lower the batch size."
            )
        data = None
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use":
                data = block.input
                break
        if data is None:
            text = " ".join(getattr(b, "text", "") for b in resp.content)
            data = extract_json_object(text)
        return LLMResponse(
            data=data,
            raw=json.dumps(data)[:4000],
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            model=self.model,
        )

    def search_json(
        self, system, user, schema, max_tokens=4000, max_searches=4
    ) -> LLMResponse:
        # A forced tool call cannot be combined with the server-side search tool -- the
        # model needs free turns to search before answering -- so the schema goes in the
        # prompt and the JSON is recovered from the final text blocks.
        client = self._client()
        messages: list[dict] = [{"role": "user", "content": user}]
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self._search_system(system, schema),
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_searches,
                }
            ],
        }

        citations: list[str] = []
        searches = 0
        prompt_tokens = completion_tokens = 0
        text = ""

        # A long search turn can come back as `pause_turn`; the documented way to finish
        # it is to send the partial assistant message straight back.
        for _ in range(4):
            resp = client.messages.create(messages=messages, **kwargs)
            prompt_tokens += resp.usage.input_tokens
            completion_tokens += resp.usage.output_tokens
            server_use = getattr(resp.usage, "server_tool_use", None)
            searches += getattr(server_use, "web_search_requests", 0) or 0

            for block in resp.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text += getattr(block, "text", "")
                    for cit in getattr(block, "citations", []) or []:
                        url = getattr(cit, "url", "")
                        if url and url not in citations:
                            citations.append(url)
                elif btype == "web_search_tool_result":
                    content = getattr(block, "content", None)
                    for r in content if isinstance(content, list) else []:
                        url = getattr(r, "url", "")
                        if url and url not in citations:
                            citations.append(url)

            if resp.stop_reason != "pause_turn":
                break
            messages = messages + [{"role": "assistant", "content": resp.content}]

        if not text.strip():
            raise LLMError("Search returned no text output.")
        return self._validate_searched(
            LLMResponse(
                data=extract_json_object(text),
                raw=text[:4000],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=self.model,
                citations=citations,
                searches=searches,
            ),
            schema,
        )


class JsonModeProvider(BaseProvider):
    """OpenAI-compatible endpoints offering only loose `json_object` mode.

    These APIs guarantee syntactically valid JSON but not conformance to a schema, so the
    schema travels in the system prompt and `complete_json` validates and repairs locally.
    DeepSeek and Ollama both land here.
    """

    name = "json-mode"
    timeout = 300.0
    requires_key = True

    def _client(self):
        from openai import OpenAI

        if self.requires_key and not self.api_key:
            raise LLMError(f"{self.name} API key is missing.")
        return OpenAI(
            api_key=self.api_key or "none",
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _system_with_schema(self, system: str, schema: dict) -> str:
        # The literal word "JSON" must appear in the prompt for json_object mode.
        return (
            f"{system}\n\n"
            "OUTPUT FORMAT: reply with a single raw JSON object and nothing else. No "
            "prose, no explanation, no markdown fences. The object must validate against "
            "this JSON Schema, and every required key must be present:\n"
            f"{json.dumps(schema, separators=(',', ':'))}"
        )

    def _sampling_kwargs(self, temperature: float | None) -> dict:
        return {} if temperature is None else {"temperature": temperature}

    def _call(self, system, user, schema, max_tokens, temperature) -> LLMResponse:
        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_with_schema(system, schema)},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            **self._sampling_kwargs(temperature),
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(
                "Output hit the token ceiling and the JSON is truncated. Raise 'Max "
                "output tokens' or lower the merge batch size."
            )
        content = choice.message.content or ""
        if not content.strip():
            raise LLMError("Model returned an empty response.")
        u = resp.usage
        return LLMResponse(
            data=extract_json_object(content),
            raw=content,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            model=self.model,
        )


class DeepSeekProvider(JsonModeProvider):
    """DeepSeek via its OpenAI-compatible endpoint.

    Only `json_object` mode is available -- there is no strict `json_schema` support -- so
    conformance relies on the validate-and-repair loop in `complete_json`.

    V4 models think by default, which silently disables `temperature` and spends part of
    the output budget on a chain of thought. Extraction is a read-and-fill task, so
    thinking is turned off explicitly: that keeps the whole budget for the JSON and makes
    the low temperature actually take effect. `deepseek-v4-flash` is the sensible default.
    """

    name = "deepseek"
    timeout = 600.0  # long pages are slow

    def _extra_body(self) -> dict:
        return {"thinking": {"type": "disabled"}}

    def _sampling_kwargs(self, temperature: float | None) -> dict:
        kw: dict = {"extra_body": self._extra_body()}
        if temperature is not None:
            kw["temperature"] = temperature
        return kw


class OllamaProvider(JsonModeProvider):
    name = "ollama"
    timeout = 600.0
    requires_key = False


DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://localhost:11434/v1",
}


def build_provider(
    provider: str, api_key: str, base_url: str, model: str
) -> BaseProvider:
    p = (provider or "openai").lower()
    if p == "openai":
        return OpenAIProvider(api_key, base_url or DEFAULT_BASE_URLS["openai"], model)
    if p == "anthropic":
        return AnthropicProvider(api_key, "", model)
    if p == "deepseek":
        return DeepSeekProvider(
            api_key, base_url or DEFAULT_BASE_URLS["deepseek"], model
        )
    if p == "ollama":
        return OllamaProvider("ollama", base_url or DEFAULT_BASE_URLS["ollama"], model)
    raise LLMError(f"Unknown provider {provider!r}. Expected one of {PROVIDERS}.")
