"""LLM provider abstraction returning schema-validated JSON.

Every provider exposes the same `complete_json` contract. The important difference is
how each one is coerced into valid JSON:

* OpenAI    -- native strict structured outputs, the most reliable option.
* Anthropic -- a forced tool call, whose input schema does the same job.
* Ollama    -- only has a loose JSON mode, so the schema goes in the prompt and we
               validate and retry locally. Weakest of the three on wide schemas.

A silent empty result is the worst possible outcome for this pipeline, so parse and
validation failures raise instead of returning blank fields.
"""

from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from jsonschema import Draft202012Validator

PROVIDERS = ("openai", "anthropic", "deepseek", "ollama")

SUGGESTED_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    "anthropic": ["claude-sonnet-4-5", "claude-opus-4-1", "claude-3-5-haiku-latest"],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-reasoner"],
    "ollama": ["llama3.1:8b", "qwen2.5:14b", "mistral-nemo"],
}

# Providers whose JSON is guaranteed by the API itself rather than by local validation.
# The rest get the schema in the prompt plus a validate-and-repair loop.
STRICT_SCHEMA_PROVIDERS = ("openai", "anthropic")


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


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    cached: int = 0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens


# Rough USD per 1M tokens (input, output), for an order-of-magnitude estimate only.
# These drift constantly -- DeepSeek in particular has repriced several times and applies
# a large cache-hit discount that is not modelled here. Treat as indicative, not billing.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "deepseek-v4-flash": (0.28, 0.42),
    "deepseek-v4-pro": (2.50, 10.00),
    "deepseek-reasoner": (0.28, 0.42),
}


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    for name, (pin, pout) in PRICES.items():
        if model.startswith(name):
            return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout
    return None


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

    def __init__(self, api_key: str, base_url: str = "", model: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @abstractmethod
    def _call(
        self, system: str, user: str, schema: dict, max_tokens: int, temperature: float
    ) -> LLMResponse: ...

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
    name = "openai"

    def _client(self):
        from openai import OpenAI

        if not self.api_key:
            raise LLMError("OpenAI API key is missing.")
        return OpenAI(
            api_key=self.api_key, base_url=self.base_url or None, timeout=180.0
        )

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
            temperature=temperature,
            max_tokens=max_tokens,
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


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def _client(self):
        import anthropic

        if not self.api_key:
            raise LLMError("Anthropic API key is missing.")
        return anthropic.Anthropic(api_key=self.api_key, timeout=180.0)

    def _call(self, system, user, schema, max_tokens, temperature) -> LLMResponse:
        resp = self._client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
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

    `deepseek-reasoner` is a reasoning model: it burns output tokens on hidden thinking and
    ignores sampling parameters, so those are omitted for it. `deepseek-v4-flash` is the
    sensible default for extraction.
    """

    name = "deepseek"
    timeout = 600.0  # long pages plus reasoning traces are slow

    @property
    def is_reasoner(self) -> bool:
        return "reasoner" in (self.model or "").lower()

    def _sampling_kwargs(self, temperature: float | None) -> dict:
        if self.is_reasoner:
            return {}
        return {} if temperature is None else {"temperature": temperature}


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
