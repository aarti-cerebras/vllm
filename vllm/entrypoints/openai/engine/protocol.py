# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from http import HTTPStatus
from typing import Any, ClassVar, Literal, TypeAlias

import regex as re
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # Compare against both field names and aliases
        if any(k not in field_names for k in data):
            logger.debug(
                "The following fields were present in the request but ignored: %s",
                data.keys() - field_names,
            )
        return result


class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo


class ModelPermission(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None = None


class TimingMetrics(OpenAIBaseModel):
    """Per-request timing metrics, mirroring ``RequestStateStats`` plus a few
    computed durations. One entry is emitted per prompt in a batched request."""

    num_generation_tokens: int = 0

    # Engine frontend timestamp (wall-clock).
    arrival_time: float = 0.0

    # Engine core timestamps (monotonic). Only meaningful as diffs.
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0

    first_token_latency: float = 0.0
    is_corrupted: bool = False

    # Computed durations (seconds).
    queue_waiting_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0


class SpecDecodeStats(OpenAIBaseModel):
    """Per-request speculative decoding metrics."""

    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int
    acceptance_rate: float
    mean_acceptance_length: float
    acceptance_rate_per_pos: list[float]
    draft_throughput: float
    accepted_throughput: float


def build_spec_decode_stats(metrics_list: list[Any]) -> "SpecDecodeStats | None":
    """Aggregate per-request spec-decode counters from one or more
    ``RequestStateStats`` (one per prompt in a batched request) into a single
    ``SpecDecodeStats``. Returns ``None`` when the request(s) used no drafting.

    Mirrors the derivations in ``SpecDecodingLogging.log``:
    mean acceptance length includes the bonus token (``1 + accepted/drafts``),
    and per-position rate is ``accepted_per_pos[i] / drafts``. Throughput uses
    the decode interval (``last_token_ts - first_token_ts``).
    """
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_per_pos: list[int] = []
    first_token_ts = float("inf")
    last_token_ts = 0.0
    for m in metrics_list:
        if m is None or getattr(m, "num_spec_drafts", 0) == 0:
            continue
        num_drafts += m.num_spec_drafts
        num_draft_tokens += m.num_spec_draft_tokens
        num_accepted_tokens += m.num_spec_accepted_tokens
        for i, count in enumerate(m.spec_accepted_per_pos):
            if i >= len(accepted_per_pos):
                accepted_per_pos.append(0)
            accepted_per_pos[i] += count
        first_token_ts = min(first_token_ts, m.first_token_ts)
        last_token_ts = max(last_token_ts, m.last_token_ts)

    if num_drafts == 0:
        return None

    decode_seconds = max(last_token_ts - first_token_ts, 0.0)
    acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens else 0.0
    )
    mean_acceptance_length = 1.0 + num_accepted_tokens / num_drafts
    acceptance_rate_per_pos = [count / num_drafts for count in accepted_per_pos]
    draft_throughput = num_draft_tokens / decode_seconds if decode_seconds > 0 else 0.0
    accepted_throughput = (
        num_accepted_tokens / decode_seconds if decode_seconds > 0 else 0.0
    )
    return SpecDecodeStats(
        num_drafts=num_drafts,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        acceptance_rate=acceptance_rate,
        mean_acceptance_length=mean_acceptance_length,
        acceptance_rate_per_pos=acceptance_rate_per_pos,
        draft_throughput=draft_throughput,
        accepted_throughput=accepted_throughput,
    )


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None
    timing: list[TimingMetrics] | None = None
    spec_decode_stats: SpecDecodeStats | None = None


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: UsageInfo | None = None


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: str | None = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class LegacyStructuralTag(OpenAIBaseModel):
    begin: str
    # schema is the field, but that causes conflicts with pydantic so
    # instead use structural_tag_schema with an alias
    structural_tag_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    end: str


class LegacyStructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    structures: list[LegacyStructuralTag]
    triggers: list[str]


class StructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    format: Any


AnyStructuralTagResponseFormat: TypeAlias = (
    LegacyStructuralTagResponseFormat | StructuralTagResponseFormat
)


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object", or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None = None


AnyResponseFormat: TypeAlias = (
    ResponseFormat | StructuralTagResponseFormat | LegacyStructuralTagResponseFormat
)


class StreamOptions(OpenAIBaseModel):
    include_usage: bool | None = False
    continuous_usage_stats: bool | None = False


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    defer_loading: bool | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.defer_loading is None:
            data.pop("defer_loading", None)
        return data


# extra="forbid" is a workaround to have kwargs as a field,
# see https://github.com/pydantic/pydantic/issues/3125
class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


LogitsProcessors = list[str | LogitsProcessorConstructor]


def get_logits_processors(
    processors: LogitsProcessors | None, pattern: str | None
) -> list[Any] | None:
    if processors and pattern:
        logits_processors = []
        for processor in processors:
            qualname = processor if isinstance(processor, str) else processor.qualname
            if not re.match(pattern, qualname):
                raise ValueError(
                    f"Logits processor '{qualname}' is not allowed by this "
                    "server. See --logits-processor-pattern engine argument "
                    "for more information."
                )
            try:
                logits_processor = resolve_obj_by_qualname(qualname)
            except Exception as e:
                raise ValueError(
                    f"Logits processor '{qualname}' could not be resolved: {e}"
                ) from e
            if isinstance(processor, LogitsProcessorConstructor):
                logits_processor = logits_processor(
                    *processor.args or [], **processor.kwargs or {}
                )
            logits_processors.append(logits_processor)
        return logits_processors
    elif processors:
        raise ValueError(
            "The `logits_processors` argument is not supported by this "
            "server. See --logits-processor-pattern engine argument "
            "for more information."
        )
    return None


class FunctionCall(OpenAIBaseModel):
    # Internal field to preserve native tool call ID from tool parser.
    # Excluded from serialization to maintain OpenAI API compatibility
    # (function object should only contain 'name' and 'arguments').
    id: str | None = Field(default=None, exclude=True)
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: str | None = None


class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class GenerationError(Exception):
    """raised when finish_reason indicates internal server error (500)"""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message)
        self.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
