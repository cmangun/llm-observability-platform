"""
LLM Observability - Distributed Tracing for AI Systems

Production-grade tracing for LLM applications:
- Request-level tracing with spans
- Token usage tracking
- Cost attribution
- Latency analysis
- Error correlation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Iterator

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class SpanKind(str, Enum):
    """Types of spans in LLM traces."""
    
    LLM_CALL = "llm_call"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    TOOL_CALL = "tool_call"
    CHAIN = "chain"
    AGENT = "agent"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    GUARDRAIL = "guardrail"


class SpanStatus(str, Enum):
    """Span execution status."""
    
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """Event within a span."""
    
    name: str
    timestamp: datetime
    attributes: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class TokenUsage:
    """Token usage for LLM calls."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    
    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class Span:
    """
    A span represents a unit of work in the LLM pipeline.
    
    Spans can be nested to form a trace tree.
    """
    
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    status: SpanStatus
    start_time: datetime
    end_time: datetime | None
    duration_ms: float | None
    
    # LLM-specific attributes
    model: str | None
    provider: str | None
    token_usage: TokenUsage | None
    cost_usd: float | None
    
    # General attributes
    attributes: dict[str, Any]
    events: list[SpanEvent]
    
    # Error info
    error_message: str | None
    error_type: str | None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "provider": self.provider,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "cost_usd": self.cost_usd,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "error_message": self.error_message,
            "error_type": self.error_type,
        }
    
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=datetime.utcnow(),
            attributes=attributes or {},
        ))
    
    def set_error(self, error: Exception) -> None:
        """Set error information on the span."""
        self.status = SpanStatus.ERROR
        self.error_type = type(error).__name__
        self.error_message = str(error)
    
    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK


@dataclass
class Trace:
    """A complete trace containing multiple spans."""
    
    trace_id: str
    name: str
    root_span_id: str
    spans: list[Span]
    start_time: datetime
    end_time: datetime | None
    total_duration_ms: float | None
    total_tokens: int
    total_cost_usd: float
    user_id: str | None
    session_id: str | None
    metadata: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "root_span_id": self.root_span_id,
            "spans": [s.to_dict() for s in self.spans],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "span_count": len(self.spans),
        }
    
    def get_span_tree(self) -> dict[str, Any]:
        """Get spans as a nested tree structure."""
        span_map = {s.span_id: s for s in self.spans}
        children_map: dict[str | None, list[Span]] = defaultdict(list)
        
        for span in self.spans:
            children_map[span.parent_span_id].append(span)
        
        def build_tree(span: Span) -> dict[str, Any]:
            return {
                "span": span.to_dict(),
                "children": [build_tree(c) for c in children_map.get(span.span_id, [])],
            }
        
        root = span_map.get(self.root_span_id)
        if root:
            return build_tree(root)
        return {}


class TracerConfig(BaseModel):
    """Configuration for the tracer."""
    
    # Sampling
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Storage
    max_traces_in_memory: int = Field(default=10000, ge=100)
    trace_ttl_hours: int = Field(default=24, ge=1)
    
    # Export
    export_batch_size: int = Field(default=100, ge=1)
    export_interval_seconds: float = Field(default=5.0, ge=0.1)
    
    # Content
    log_prompts: bool = False  # Security: disable by default
    log_completions: bool = False
    max_content_length: int = Field(default=1000, ge=0)


class TraceExporter:
    """Base class for trace exporters."""
    
    async def export(self, traces: list[Trace]) -> bool:
        """Export traces to backend."""
        raise NotImplementedError


class ConsoleExporter(TraceExporter):
    """Export traces to console for debugging."""
    
    async def export(self, traces: list[Trace]) -> bool:
        for trace in traces:
            logger.info(
                "trace_exported",
                trace_id=trace.trace_id,
                name=trace.name,
                duration_ms=trace.total_duration_ms,
                span_count=len(trace.spans),
                total_tokens=trace.total_tokens,
                total_cost_usd=trace.total_cost_usd,
            )
        return True


class InMemoryStore:
    """In-memory trace storage."""
    
    def __init__(self, max_traces: int = 10000):
        self.max_traces = max_traces
        self._traces: dict[str, Trace] = {}
        self._lock = asyncio.Lock()
    
    async def store(self, trace: Trace) -> None:
        async with self._lock:
            # Evict old traces if at capacity
            if len(self._traces) >= self.max_traces:
                oldest_key = min(
                    self._traces.keys(),
                    key=lambda k: self._traces[k].start_time
                )
                del self._traces[oldest_key]
            
            self._traces[trace.trace_id] = trace
    
    async def get(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)
    
    async def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        min_duration_ms: float | None = None,
        has_error: bool | None = None,
        limit: int = 100,
    ) -> list[Trace]:
        results = list(self._traces.values())
        
        if start_time:
            results = [t for t in results if t.start_time >= start_time]
        if end_time:
            results = [t for t in results if t.start_time <= end_time]
        if user_id:
            results = [t for t in results if t.user_id == user_id]
        if min_duration_ms:
            results = [
                t for t in results
                if t.total_duration_ms and t.total_duration_ms >= min_duration_ms
            ]
        if has_error is not None:
            results = [
                t for t in results
                if any(s.status == SpanStatus.ERROR for s in t.spans) == has_error
            ]
        
        return sorted(results, key=lambda t: t.start_time, reverse=True)[:limit]


class LLMTracer:
    """
    Production LLM tracing system.
    
    Features:
    - Distributed tracing with span trees
    - Token and cost tracking
    - Async-safe context management
    - Configurable sampling
    - Multiple export backends
    """
    
    def __init__(
        self,
        config: TracerConfig | None = None,
        exporter: TraceExporter | None = None,
    ):
        self.config = config or TracerConfig()
        self.exporter = exporter or ConsoleExporter()
        self.store = InMemoryStore(self.config.max_traces_in_memory)
        
        self._active_traces: dict[str, Trace] = {}
        self._active_spans: dict[str, Span] = {}
        self._span_stack: dict[str, list[str]] = defaultdict(list)  # trace_id -> [span_ids]
        self._lock = asyncio.Lock()
    
    def start_trace(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Trace:
        """Start a new trace."""
        import random
        
        # Sampling
        if random.random() > self.config.sample_rate:
            return None  # type: ignore
        
        trace_id = self._generate_id("trace")
        root_span_id = self._generate_id("span")
        now = datetime.utcnow()
        
        trace = Trace(
            trace_id=trace_id,
            name=name,
            root_span_id=root_span_id,
            spans=[],
            start_time=now,
            end_time=None,
            total_duration_ms=None,
            total_tokens=0,
            total_cost_usd=0.0,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        
        self._active_traces[trace_id] = trace
        
        logger.debug("trace_started", trace_id=trace_id, name=name)
        
        return trace
    
    @contextmanager
    def span(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind = SpanKind.CHAIN,
        model: str | None = None,
        provider: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Context manager for creating spans."""
        span = self._start_span(trace_id, name, kind, model, provider, attributes)
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self._end_span(span)
    
    @asynccontextmanager
    async def async_span(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind = SpanKind.CHAIN,
        model: str | None = None,
        provider: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """Async context manager for creating spans."""
        span = self._start_span(trace_id, name, kind, model, provider, attributes)
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self._end_span(span)
    
    def _start_span(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind,
        model: str | None,
        provider: str | None,
        attributes: dict[str, Any] | None,
    ) -> Span:
        """Start a new span."""
        trace = self._active_traces.get(trace_id)
        if not trace:
            raise ValueError(f"Trace not found: {trace_id}")
        
        span_id = self._generate_id("span")
        
        # Get parent span (top of stack)
        parent_span_id = None
        if self._span_stack[trace_id]:
            parent_span_id = self._span_stack[trace_id][-1]
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            status=SpanStatus.UNSET,
            start_time=datetime.utcnow(),
            end_time=None,
            duration_ms=None,
            model=model,
            provider=provider,
            token_usage=None,
            cost_usd=None,
            attributes=attributes or {},
            events=[],
            error_message=None,
            error_type=None,
        )
        
        self._active_spans[span_id] = span
        self._span_stack[trace_id].append(span_id)
        
        return span
    
    def _end_span(self, span: Span) -> None:
        """End a span and update trace."""
        span.end()
        
        trace = self._active_traces.get(span.trace_id)
        if trace:
            trace.spans.append(span)
            
            # Update trace totals
            if span.token_usage:
                trace.total_tokens += span.token_usage.total_tokens
            if span.cost_usd:
                trace.total_cost_usd += span.cost_usd
        
        # Pop from stack
        if span.span_id in self._span_stack.get(span.trace_id, []):
            self._span_stack[span.trace_id].remove(span.span_id)
        
        if span.span_id in self._active_spans:
            del self._active_spans[span.span_id]
    
    def record_llm_call(
        self,
        span: Span,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        model: str | None = None,
    ) -> None:
        """Record LLM call details on a span."""
        span.token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        span.cost_usd = cost_usd
        if model:
            span.model = model
    
    async def end_trace(self, trace_id: str) -> Trace | None:
        """End a trace and export."""
        trace = self._active_traces.pop(trace_id, None)
        if not trace:
            return None
        
        trace.end_time = datetime.utcnow()
        trace.total_duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000
        
        # Store and export
        await self.store.store(trace)
        await self.exporter.export([trace])
        
        # Cleanup
        if trace_id in self._span_stack:
            del self._span_stack[trace_id]
        
        logger.info(
            "trace_ended",
            trace_id=trace_id,
            duration_ms=trace.total_duration_ms,
            span_count=len(trace.spans),
            total_tokens=trace.total_tokens,
            total_cost_usd=trace.total_cost_usd,
        )
        
        return trace
    
    async def get_trace(self, trace_id: str) -> Trace | None:
        """Get a trace by ID."""
        return await self.store.get(trace_id)
    
    async def query_traces(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        min_duration_ms: float | None = None,
        has_error: bool | None = None,
        limit: int = 100,
    ) -> list[Trace]:
        """Query traces with filters."""
        return await self.store.query(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            min_duration_ms=min_duration_ms,
            has_error=has_error,
            limit=limit,
        )
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:16]}"
