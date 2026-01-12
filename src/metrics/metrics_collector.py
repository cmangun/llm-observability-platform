"""
LLM Metrics Collector for Observability Platform.

Production-grade metrics collection for LLM systems:
- Request/response latency tracking
- Token usage and cost monitoring
- Model performance metrics
- Error rate and classification
- Custom business metrics
- Prometheus-compatible export
"""

from __future__ import annotations

import hashlib
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics."""
    
    COUNTER = "counter"           # Monotonically increasing
    GAUGE = "gauge"               # Point-in-time value
    HISTOGRAM = "histogram"       # Distribution of values
    SUMMARY = "summary"           # Quantile summaries


class MetricStatus(Enum):
    """Metric health status."""
    
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: list[str] = field(default_factory=list)
    warning_threshold: float | None = None
    critical_threshold: float | None = None


@dataclass
class MetricValue:
    """Single metric value."""
    
    name: str
    value: float
    labels: dict[str, str]
    timestamp: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HistogramBucket:
    """Histogram bucket for distribution tracking."""
    
    le: float  # Less than or equal
    count: int


@dataclass
class HistogramValue:
    """Histogram metric value."""
    
    name: str
    labels: dict[str, str]
    buckets: list[HistogramBucket]
    sum: float
    count: int
    
    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


@dataclass
class LLMRequestMetrics:
    """Metrics for a single LLM request."""
    
    request_id: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    time_to_first_token_ms: float | None
    cost_usd: float
    status: str  # success, error, timeout
    error_type: str | None
    timestamp: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "error_type": self.error_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a time window."""
    
    window_start: datetime
    window_end: datetime
    request_count: int
    success_count: int
    error_count: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tokens_per_second: float
    error_rate: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "error_rate": self.error_rate,
        }


# Default histogram buckets for latency (in milliseconds)
DEFAULT_LATENCY_BUCKETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000]

# Default histogram buckets for tokens
DEFAULT_TOKEN_BUCKETS = [100, 500, 1000, 2000, 4000, 8000, 16000, 32000]


class MetricsCollector:
    """
    Production metrics collector for LLM systems.
    
    Features:
    - Real-time metrics collection
    - Histogram and summary statistics
    - Label-based aggregation
    - Prometheus export format
    - Alerting thresholds
    """
    
    def __init__(
        self,
        latency_buckets: list[float] | None = None,
        token_buckets: list[float] | None = None,
        retention_hours: int = 24,
    ):
        self.latency_buckets = latency_buckets or DEFAULT_LATENCY_BUCKETS
        self.token_buckets = token_buckets or DEFAULT_TOKEN_BUCKETS
        self.retention_hours = retention_hours
        
        # Thread-safe storage
        self._lock = threading.Lock()
        
        # Counters
        self._request_count: dict[tuple, int] = defaultdict(int)
        self._token_count: dict[tuple, int] = defaultdict(int)
        self._error_count: dict[tuple, int] = defaultdict(int)
        self._cost_total: dict[tuple, float] = defaultdict(float)
        
        # Histograms (latency)
        self._latency_histogram: dict[tuple, list[float]] = defaultdict(list)
        self._ttft_histogram: dict[tuple, list[float]] = defaultdict(list)
        
        # Request log for detailed analysis
        self._request_log: list[LLMRequestMetrics] = []
        
        # Metric definitions
        self._metrics: dict[str, MetricDefinition] = {}
        self._register_default_metrics()
    
    def _register_default_metrics(self) -> None:
        """Register default LLM metrics."""
        self._metrics["llm_requests_total"] = MetricDefinition(
            name="llm_requests_total",
            metric_type=MetricType.COUNTER,
            description="Total number of LLM requests",
            labels=["model", "provider", "status"],
        )
        
        self._metrics["llm_tokens_total"] = MetricDefinition(
            name="llm_tokens_total",
            metric_type=MetricType.COUNTER,
            description="Total tokens processed",
            unit="tokens",
            labels=["model", "provider", "type"],
        )
        
        self._metrics["llm_cost_total"] = MetricDefinition(
            name="llm_cost_total",
            metric_type=MetricType.COUNTER,
            description="Total cost in USD",
            unit="usd",
            labels=["model", "provider"],
        )
        
        self._metrics["llm_request_latency_ms"] = MetricDefinition(
            name="llm_request_latency_ms",
            metric_type=MetricType.HISTOGRAM,
            description="Request latency in milliseconds",
            unit="ms",
            labels=["model", "provider"],
            warning_threshold=2000,
            critical_threshold=10000,
        )
        
        self._metrics["llm_time_to_first_token_ms"] = MetricDefinition(
            name="llm_time_to_first_token_ms",
            metric_type=MetricType.HISTOGRAM,
            description="Time to first token in milliseconds",
            unit="ms",
            labels=["model", "provider"],
            warning_threshold=500,
            critical_threshold=2000,
        )
        
        self._metrics["llm_error_rate"] = MetricDefinition(
            name="llm_error_rate",
            metric_type=MetricType.GAUGE,
            description="Error rate (0-1)",
            labels=["model", "provider"],
            warning_threshold=0.01,
            critical_threshold=0.05,
        )
    
    def record_request(self, metrics: LLMRequestMetrics) -> None:
        """
        Record metrics for an LLM request.
        
        Args:
            metrics: Request metrics to record
        """
        with self._lock:
            # Update counters
            labels = (metrics.model, metrics.provider, metrics.status)
            self._request_count[labels] += 1
            
            # Token counters
            self._token_count[(metrics.model, metrics.provider, "prompt")] += metrics.prompt_tokens
            self._token_count[(metrics.model, metrics.provider, "completion")] += metrics.completion_tokens
            self._token_count[(metrics.model, metrics.provider, "total")] += metrics.total_tokens
            
            # Cost
            self._cost_total[(metrics.model, metrics.provider)] += metrics.cost_usd
            
            # Errors
            if metrics.status != "success":
                error_type = metrics.error_type or "unknown"
                self._error_count[(metrics.model, metrics.provider, error_type)] += 1
            
            # Latency histogram
            latency_labels = (metrics.model, metrics.provider)
            self._latency_histogram[latency_labels].append(metrics.latency_ms)
            
            # TTFT histogram
            if metrics.time_to_first_token_ms:
                self._ttft_histogram[latency_labels].append(metrics.time_to_first_token_ms)
            
            # Request log
            self._request_log.append(metrics)
            
            # Cleanup old entries
            self._cleanup_old_entries()
        
        logger.debug(
            "llm_metrics_recorded",
            request_id=metrics.request_id,
            model=metrics.model,
            latency_ms=metrics.latency_ms,
            tokens=metrics.total_tokens,
        )
    
    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> int:
        """Get counter value."""
        with self._lock:
            if name == "llm_requests_total":
                if labels:
                    key = (labels.get("model", ""), labels.get("provider", ""), labels.get("status", ""))
                    return self._request_count.get(key, 0)
                return sum(self._request_count.values())
            
            elif name == "llm_tokens_total":
                if labels:
                    key = (labels.get("model", ""), labels.get("provider", ""), labels.get("type", "total"))
                    return self._token_count.get(key, 0)
                return sum(v for k, v in self._token_count.items() if k[2] == "total")
            
            elif name == "llm_errors_total":
                if labels:
                    key = (labels.get("model", ""), labels.get("provider", ""), labels.get("error_type", ""))
                    return self._error_count.get(key, 0)
                return sum(self._error_count.values())
        
        return 0
    
    def get_histogram(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> HistogramValue | None:
        """Get histogram value."""
        with self._lock:
            if name == "llm_request_latency_ms":
                data = self._latency_histogram
                buckets = self.latency_buckets
            elif name == "llm_time_to_first_token_ms":
                data = self._ttft_histogram
                buckets = self.latency_buckets
            else:
                return None
            
            if labels:
                key = (labels.get("model", ""), labels.get("provider", ""))
                values = data.get(key, [])
            else:
                values = []
                for v in data.values():
                    values.extend(v)
            
            if not values:
                return None
            
            # Calculate buckets
            histogram_buckets = []
            for bucket_le in buckets:
                count = sum(1 for v in values if v <= bucket_le)
                histogram_buckets.append(HistogramBucket(le=bucket_le, count=count))
            
            histogram_buckets.append(HistogramBucket(le=float("inf"), count=len(values)))
            
            return HistogramValue(
                name=name,
                labels=labels or {},
                buckets=histogram_buckets,
                sum=sum(values),
                count=len(values),
            )
    
    def get_percentile(
        self,
        name: str,
        percentile: float,
        labels: dict[str, str] | None = None,
    ) -> float | None:
        """Get percentile value from histogram."""
        with self._lock:
            if name == "llm_request_latency_ms":
                data = self._latency_histogram
            elif name == "llm_time_to_first_token_ms":
                data = self._ttft_histogram
            else:
                return None
            
            if labels:
                key = (labels.get("model", ""), labels.get("provider", ""))
                values = data.get(key, [])
            else:
                values = []
                for v in data.values():
                    values.extend(v)
            
            if not values:
                return None
            
            sorted_values = sorted(values)
            idx = int(len(sorted_values) * percentile / 100)
            return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def get_aggregated_metrics(
        self,
        window_minutes: int = 5,
        model: str | None = None,
        provider: str | None = None,
    ) -> AggregatedMetrics:
        """
        Get aggregated metrics for a time window.
        
        Args:
            window_minutes: Size of time window
            model: Filter by model
            provider: Filter by provider
            
        Returns:
            Aggregated metrics
        """
        with self._lock:
            window_end = datetime.utcnow()
            window_start = window_end - timedelta(minutes=window_minutes)
            
            # Filter requests
            requests = [
                r for r in self._request_log
                if r.timestamp >= window_start
                and (model is None or r.model == model)
                and (provider is None or r.provider == provider)
            ]
            
            if not requests:
                return AggregatedMetrics(
                    window_start=window_start,
                    window_end=window_end,
                    request_count=0,
                    success_count=0,
                    error_count=0,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    avg_latency_ms=0.0,
                    p50_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    tokens_per_second=0.0,
                    error_rate=0.0,
                )
            
            # Calculate metrics
            request_count = len(requests)
            success_count = sum(1 for r in requests if r.status == "success")
            error_count = request_count - success_count
            total_tokens = sum(r.total_tokens for r in requests)
            total_cost_usd = sum(r.cost_usd for r in requests)
            
            latencies = [r.latency_ms for r in requests]
            sorted_latencies = sorted(latencies)
            
            avg_latency_ms = statistics.mean(latencies)
            p50_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.50)]
            p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            
            window_seconds = window_minutes * 60
            tokens_per_second = total_tokens / window_seconds
            error_rate = error_count / request_count
            
            return AggregatedMetrics(
                window_start=window_start,
                window_end=window_end,
                request_count=request_count,
                success_count=success_count,
                error_count=error_count,
                total_tokens=total_tokens,
                total_cost_usd=total_cost_usd,
                avg_latency_ms=avg_latency_ms,
                p50_latency_ms=p50_latency_ms,
                p95_latency_ms=p95_latency_ms,
                p99_latency_ms=p99_latency_ms,
                tokens_per_second=tokens_per_second,
                error_rate=error_rate,
            )
    
    def get_health_status(self) -> dict[str, Any]:
        """Get health status based on metric thresholds."""
        metrics = self.get_aggregated_metrics(window_minutes=5)
        
        status = MetricStatus.HEALTHY
        issues: list[str] = []
        
        # Check error rate
        error_def = self._metrics.get("llm_error_rate")
        if error_def:
            if error_def.critical_threshold and metrics.error_rate >= error_def.critical_threshold:
                status = MetricStatus.CRITICAL
                issues.append(f"Error rate {metrics.error_rate:.2%} exceeds critical threshold")
            elif error_def.warning_threshold and metrics.error_rate >= error_def.warning_threshold:
                if status != MetricStatus.CRITICAL:
                    status = MetricStatus.WARNING
                issues.append(f"Error rate {metrics.error_rate:.2%} exceeds warning threshold")
        
        # Check latency
        latency_def = self._metrics.get("llm_request_latency_ms")
        if latency_def:
            if latency_def.critical_threshold and metrics.p95_latency_ms >= latency_def.critical_threshold:
                status = MetricStatus.CRITICAL
                issues.append(f"P95 latency {metrics.p95_latency_ms:.0f}ms exceeds critical threshold")
            elif latency_def.warning_threshold and metrics.p95_latency_ms >= latency_def.warning_threshold:
                if status != MetricStatus.CRITICAL:
                    status = MetricStatus.WARNING
                issues.append(f"P95 latency {metrics.p95_latency_ms:.0f}ms exceeds warning threshold")
        
        return {
            "status": status.value,
            "issues": issues,
            "metrics_summary": {
                "request_count": metrics.request_count,
                "error_rate": metrics.error_rate,
                "p95_latency_ms": metrics.p95_latency_ms,
                "tokens_per_second": metrics.tokens_per_second,
            },
            "checked_at": datetime.utcnow().isoformat(),
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines: list[str] = []
        
        # Request counter
        lines.append("# HELP llm_requests_total Total number of LLM requests")
        lines.append("# TYPE llm_requests_total counter")
        for labels, count in self._request_count.items():
            model, provider, status = labels
            lines.append(
                f'llm_requests_total{{model="{model}",provider="{provider}",status="{status}"}} {count}'
            )
        
        # Token counter
        lines.append("# HELP llm_tokens_total Total tokens processed")
        lines.append("# TYPE llm_tokens_total counter")
        for labels, count in self._token_count.items():
            model, provider, token_type = labels
            lines.append(
                f'llm_tokens_total{{model="{model}",provider="{provider}",type="{token_type}"}} {count}'
            )
        
        # Cost counter
        lines.append("# HELP llm_cost_usd_total Total cost in USD")
        lines.append("# TYPE llm_cost_usd_total counter")
        for labels, cost in self._cost_total.items():
            model, provider = labels
            lines.append(
                f'llm_cost_usd_total{{model="{model}",provider="{provider}"}} {cost:.6f}'
            )
        
        # Latency histogram
        lines.append("# HELP llm_request_latency_ms Request latency in milliseconds")
        lines.append("# TYPE llm_request_latency_ms histogram")
        for labels, values in self._latency_histogram.items():
            if not values:
                continue
            
            model, provider = labels
            sorted_values = sorted(values)
            
            for bucket_le in self.latency_buckets:
                count = sum(1 for v in sorted_values if v <= bucket_le)
                lines.append(
                    f'llm_request_latency_ms_bucket{{model="{model}",provider="{provider}",le="{bucket_le}"}} {count}'
                )
            
            lines.append(
                f'llm_request_latency_ms_bucket{{model="{model}",provider="{provider}",le="+Inf"}} {len(values)}'
            )
            lines.append(
                f'llm_request_latency_ms_sum{{model="{model}",provider="{provider}"}} {sum(values):.2f}'
            )
            lines.append(
                f'llm_request_latency_ms_count{{model="{model}",provider="{provider}"}} {len(values)}'
            )
        
        return "\n".join(lines)
    
    def _cleanup_old_entries(self) -> None:
        """Remove entries older than retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        # Clean request log
        self._request_log = [
            r for r in self._request_log
            if r.timestamp >= cutoff
        ]
        
        # Note: Counters are not cleaned as they're cumulative
        # Histograms should be cleaned based on time windows if needed
