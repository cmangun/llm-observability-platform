"""
LLM Observability Tracer

Production tracing for LLM applications:
- Request/response tracing
- Token usage tracking
- Latency monitoring
- Cost attribution
- Quality evaluation
- Error tracking
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TraceStatus(str, Enum):
    """Status of a trace."""
    
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SpanKind(str, Enum):
    """Type of span."""
    
    LLM_CALL = "llm_call"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    TOOL_CALL = "tool_call"
    CHAIN = "chain"
    AGENT = "agent"
    CUSTOM = "custom"


class EvaluationType(str, Enum):
    """Types of quality evaluation."""
    
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    COHERENCE = "coherence"
    TOXICITY = "toxicity"
    HALLUCINATION = "hallucination"
    CUSTOM = "custom"


@dataclass
class TokenUsage:
    """Token usage for an LLM call."""
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    
    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class CostInfo:
    """Cost information for a span."""
    
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "currency": self.currency,
        }


@dataclass
class Evaluation:
    """Quality evaluation result."""
    
    evaluation_type: EvaluationType
    score: float  # 0.0 to 1.0
    explanation: str | None = None
    evaluator: str = "automated"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_type": self.evaluation_type.value,
            "score": round(self.score, 4),
            "explanation": self.explanation,
            "evaluator": self.evaluator,
            "metadata": self.metadata,
        }


@dataclass
class Span:
    """A span representing a unit of work."""
    
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: datetime | None = None
    status: TraceStatus = TraceStatus.RUNNING
    
    # LLM-specific fields
    model: str | None = None
    provider: str | None = None
    input_text: str | None = None
    output_text: str | None = None
    token_usage: TokenUsage | None = None
    cost: CostInfo | None = None
    
    # Quality
    evaluations: list[Evaluation] = field(default_factory=list)
    
    # Metadata
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    
    @property
    def duration_ms(self) -> float | None:
        """Calculate span duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "model": self.model,
            "provider": self.provider,
            "input_text": self.input_text[:500] if self.input_text else None,
            "output_text": self.output_text[:500] if self.output_text else None,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "cost": self.cost.to_dict() if self.cost else None,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


@dataclass
class Trace:
    """A complete trace representing an end-to-end request."""
    
    trace_id: str
    name: str
    start_time: datetime
    end_time: datetime | None = None
    status: TraceStatus = TraceStatus.RUNNING
    spans: list[Span] = field(default_factory=list)
    
    # Request context
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    
    # Aggregated metrics
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float | None:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    @property
    def total_tokens(self) -> int:
        return sum(
            s.token_usage.total_tokens
            for s in self.spans
            if s.token_usage
        )
    
    @property
    def total_cost(self) -> float:
        return sum(
            s.cost.total_cost
            for s in self.spans
            if s.cost
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "span_count": len(self.spans),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }


class TracerConfig(BaseModel):
    """Configuration for the tracer."""
    
    # Sampling
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Storage
    max_traces_in_memory: int = Field(default=10000, ge=100)
    trace_ttl_hours: int = Field(default=24, ge=1)
    
    # Content
    capture_input: bool = True
    capture_output: bool = True
    max_content_length: int = Field(default=10000, ge=100)
    
    # Cost tracking
    enable_cost_tracking: bool = True
    
    # Quality evaluation
    enable_auto_evaluation: bool = False


# Model pricing for cost calculation
MODEL_COSTS = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}


class LLMTracer:
    """
    Production LLM observability tracer.
    
    Features:
    - Distributed tracing with spans
    - Token usage and cost tracking
    - Latency monitoring
    - Quality evaluation integration
    - Error tracking and alerting
    """
    
    def __init__(self, config: TracerConfig | None = None):
        self.config = config or TracerConfig()
        self._traces: dict[str, Trace] = {}
        self._spans: dict[str, Span] = {}
        self._active_trace_id: str | None = None
        self._metrics: dict[str, Any] = defaultdict(lambda: defaultdict(float))
    
    def start_trace(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Trace:
        """
        Start a new trace.
        
        Args:
            name: Trace name (e.g., "chat_completion", "rag_query")
            user_id: User identifier
            session_id: Session identifier
            request_id: Request identifier
            metadata: Additional metadata
        
        Returns:
            New Trace object
        """
        # Sampling
        import random
        if random.random() > self.config.sample_rate:
            return None
        
        trace = Trace(
            trace_id=str(uuid.uuid4()),
            name=name,
            start_time=datetime.utcnow(),
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            metadata=metadata or {},
        )
        
        self._traces[trace.trace_id] = trace
        self._active_trace_id = trace.trace_id
        
        # Enforce memory limits
        self._cleanup_old_traces()
        
        logger.debug(
            "trace_started",
            trace_id=trace.trace_id,
            name=name,
        )
        
        return trace
    
    def start_span(
        self,
        name: str,
        kind: SpanKind,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Start a new span within a trace.
        
        Args:
            name: Span name
            kind: Type of span
            trace_id: Parent trace ID (uses active trace if not specified)
            parent_span_id: Parent span ID
            model: LLM model name
            provider: LLM provider name
            attributes: Additional attributes
        
        Returns:
            New Span object
        """
        trace_id = trace_id or self._active_trace_id
        if not trace_id or trace_id not in self._traces:
            raise ValueError("No active trace. Call start_trace first.")
        
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=datetime.utcnow(),
            model=model,
            provider=provider,
            attributes=attributes or {},
        )
        
        self._spans[span.span_id] = span
        self._traces[trace_id].spans.append(span)
        
        logger.debug(
            "span_started",
            span_id=span.span_id,
            trace_id=trace_id,
            name=name,
            kind=kind.value,
        )
        
        return span
    
    def end_span(
        self,
        span_id: str,
        status: TraceStatus = TraceStatus.SUCCESS,
        output_text: str | None = None,
        token_usage: TokenUsage | None = None,
        error: str | None = None,
    ) -> Span:
        """
        End a span.
        
        Args:
            span_id: Span to end
            status: Final status
            output_text: LLM output
            token_usage: Token usage
            error: Error message if failed
        
        Returns:
            Updated Span
        """
        span = self._spans.get(span_id)
        if not span:
            raise ValueError(f"Span not found: {span_id}")
        
        span.end_time = datetime.utcnow()
        span.status = status
        span.error = error
        
        if output_text and self.config.capture_output:
            span.output_text = output_text[:self.config.max_content_length]
        
        if token_usage:
            span.token_usage = token_usage
            
            # Calculate cost
            if self.config.enable_cost_tracking and span.model:
                span.cost = self._calculate_cost(span.model, token_usage)
        
        # Update metrics
        self._update_metrics(span)
        
        logger.debug(
            "span_ended",
            span_id=span_id,
            status=status.value,
            duration_ms=span.duration_ms,
        )
        
        return span
    
    def end_trace(
        self,
        trace_id: str,
        status: TraceStatus = TraceStatus.SUCCESS,
    ) -> Trace:
        """
        End a trace.
        
        Args:
            trace_id: Trace to end
            status: Final status
        
        Returns:
            Updated Trace
        """
        trace = self._traces.get(trace_id)
        if not trace:
            raise ValueError(f"Trace not found: {trace_id}")
        
        trace.end_time = datetime.utcnow()
        trace.status = status
        
        if self._active_trace_id == trace_id:
            self._active_trace_id = None
        
        logger.info(
            "trace_ended",
            trace_id=trace_id,
            status=status.value,
            duration_ms=trace.duration_ms,
            total_tokens=trace.total_tokens,
            total_cost=trace.total_cost,
        )
        
        return trace
    
    def add_evaluation(
        self,
        span_id: str,
        evaluation_type: EvaluationType,
        score: float,
        explanation: str | None = None,
        evaluator: str = "automated",
    ) -> Evaluation:
        """Add a quality evaluation to a span."""
        span = self._spans.get(span_id)
        if not span:
            raise ValueError(f"Span not found: {span_id}")
        
        evaluation = Evaluation(
            evaluation_type=evaluation_type,
            score=score,
            explanation=explanation,
            evaluator=evaluator,
        )
        
        span.evaluations.append(evaluation)
        return evaluation
    
    def log_event(
        self,
        span_id: str,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Log an event within a span."""
        span = self._spans.get(span_id)
        if not span:
            return
        
        span.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })
    
    def set_input(self, span_id: str, input_text: str) -> None:
        """Set the input text for a span."""
        span = self._spans.get(span_id)
        if span and self.config.capture_input:
            span.input_text = input_text[:self.config.max_content_length]
    
    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a trace by ID."""
        return self._traces.get(trace_id)
    
    def get_span(self, span_id: str) -> Span | None:
        """Get a span by ID."""
        return self._spans.get(span_id)
    
    def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics."""
        return {
            "total_traces": len(self._traces),
            "total_spans": len(self._spans),
            "by_model": dict(self._metrics["by_model"]),
            "by_status": dict(self._metrics["by_status"]),
            "total_tokens": self._metrics["totals"]["tokens"],
            "total_cost": round(self._metrics["totals"]["cost"], 4),
            "avg_latency_ms": (
                self._metrics["totals"]["latency"] / self._metrics["totals"]["span_count"]
                if self._metrics["totals"]["span_count"] > 0 else 0
            ),
        }
    
    def _calculate_cost(self, model: str, usage: TokenUsage) -> CostInfo:
        """Calculate cost for token usage."""
        pricing = MODEL_COSTS.get(model, {"input": 0.01, "output": 0.03})
        
        input_cost = (usage.prompt_tokens / 1000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1000) * pricing["output"]
        
        return CostInfo(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )
    
    def _update_metrics(self, span: Span) -> None:
        """Update aggregated metrics."""
        if span.model:
            self._metrics["by_model"][span.model] += 1
        
        self._metrics["by_status"][span.status.value] += 1
        
        if span.token_usage:
            self._metrics["totals"]["tokens"] += span.token_usage.total_tokens
        
        if span.cost:
            self._metrics["totals"]["cost"] += span.cost.total_cost
        
        if span.duration_ms:
            self._metrics["totals"]["latency"] += span.duration_ms
            self._metrics["totals"]["span_count"] += 1
    
    def _cleanup_old_traces(self) -> None:
        """Remove old traces to enforce memory limits."""
        if len(self._traces) <= self.config.max_traces_in_memory:
            return
        
        cutoff = datetime.utcnow() - timedelta(hours=self.config.trace_ttl_hours)
        
        to_remove = [
            tid for tid, trace in self._traces.items()
            if trace.start_time < cutoff
        ]
        
        for tid in to_remove[:len(to_remove) // 2]:  # Remove half at a time
            trace = self._traces.pop(tid, None)
            if trace:
                for span in trace.spans:
                    self._spans.pop(span.span_id, None)
