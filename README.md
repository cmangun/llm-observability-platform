# LLM Observability Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production observability platform for LLM applications with distributed tracing, cost tracking, and quality evaluation.**

## 🎯 Business Impact

- **Real-time cost tracking** preventing budget overruns
- **Latency analysis** identifying performance bottlenecks
- **Quality evaluation** ensuring response quality
- **Full trace visibility** for debugging and optimization

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   LLM Request   │────▶│   Tracer         │────▶│   Exporter      │
│                 │     │   (Spans)        │     │   (Backend)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   Metrics Store  │◀─────────────┘
                        └────────┬─────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Cost Dashboard │    │  Latency Graphs │    │  Quality Scores │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ✨ Key Features

### 📊 Distributed Tracing
- Hierarchical span trees
- Cross-service correlation
- Error propagation tracking
- Async-safe context management

### 💰 Cost Tracking
- Token usage per request
- Cost attribution by user/session
- Budget alerting
- Model cost comparison

### ⚡ Performance Metrics
- Latency percentiles (P50, P95, P99)
- Throughput monitoring
- Time-to-first-token
- Streaming performance

## 🚀 Quick Start

```python
from src.tracing.tracer import LLMTracer, SpanKind

tracer = LLMTracer()

# Start a trace
trace = tracer.start_trace(
    name="chat_completion",
    user_id="user_123",
)

# Create spans for each operation
with tracer.span(trace.trace_id, "retrieval", SpanKind.RETRIEVAL) as span:
    # Retrieval logic
    span.add_event("documents_retrieved", {"count": 5})

with tracer.span(trace.trace_id, "llm_call", SpanKind.LLM_CALL, model="gpt-4") as span:
    # LLM call
    tracer.record_llm_call(
        span,
        prompt_tokens=150,
        completion_tokens=200,
        cost_usd=0.012,
    )

# End and export trace
await tracer.end_trace(trace.trace_id)
```

## 📁 Project Structure

```
llm-observability-platform/
├── src/
│   ├── tracing/
│   │   └── tracer.py            # Distributed tracing
│   ├── metrics/
│   │   └── collector.py         # Prometheus metrics
│   ├── evaluation/
│   │   └── quality_scorer.py    # Response evaluation
│   └── dashboards/
│       └── grafana/             # Dashboard configs
├── tests/
└── configs/
```

## 👤 Author

**Christopher Mangun** - [github.com/cmangun](https://github.com/cmangun)
