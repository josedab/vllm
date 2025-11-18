# RFC-0006: Enhanced Observability Dashboard

**Status:** Draft
**Author:** Analysis Team
**Created:** November 18, 2025
**Commit Reference:** `67745d189fd981ee824bde35666a3737a962c031`

---

## Summary

Provide pre-built Grafana dashboards, improved metrics organization, and better integration with common observability stacks to reduce mean-time-to-resolution (MTTR) for production issues.

---

## Motivation

### Current State

vLLM exposes 40+ Prometheus metrics at `/metrics`, but:

1. **No Visualization:** Users must build dashboards from scratch
2. **Metrics Overload:** Hard to know which metrics matter
3. **No Alerting:** No recommended thresholds
4. **Poor Organization:** Metrics not grouped logically

### User Pain Points

- "What should I monitor in production?"
- "How do I debug slow requests?"
- "What are good alert thresholds?"

---

## Detailed Design

### 1. Pre-built Dashboards

#### Main Dashboard

```
┌─────────────────────────────────────────────────────────┐
│  vLLM Overview Dashboard                                 │
├─────────────────────────────────────────────────────────┤
│ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ │
│ │ Throughput│ │ Latency   │ │ Queue     │ │ Memory    │ │
│ │ 3.2K/s    │ │ P99: 45ms │ │ Depth: 12 │ │ 87%       │ │
│ └───────────┘ └───────────┘ └───────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────┤
│                    Throughput Over Time                  │
│    █                                                     │
│   ██                    ████                            │
│  ████████████████████████████████████                   │
│ ─────────────────────────────────────────────────        │
├─────────────────────────────────────────────────────────┤
│                    Latency Distribution                  │
│  TTFT: [========|====] P50: 25ms, P99: 45ms             │
│  ITL:  [======|==] P50: 12ms, P99: 28ms                 │
├─────────────────────────────────────────────────────────┤
│                    Resource Utilization                  │
│  KV Cache:  [██████████████████░░] 87%                  │
│  GPU Mem:   [███████████████████░] 94%                  │
│  Requests:  Running: 45, Waiting: 12                     │
└─────────────────────────────────────────────────────────┘
```

#### Request Analytics Dashboard

- Request rate by endpoint
- Latency breakdown (queue, prefill, decode)
- Error rate and types
- Token distribution (input/output)

#### Resource Dashboard

- GPU memory usage breakdown
- KV cache utilization
- Prefix cache hit rate
- Preemption rate

### 2. Metrics Organization

Group metrics by domain:

```yaml
# metrics/organized_metrics.yaml

latency:
  - name: vllm:time_to_first_token_seconds
    description: Time to generate first token
    importance: critical
    alert_threshold:
      warning: p99 > 100ms
      critical: p99 > 500ms
  - name: vllm:inter_token_latency_seconds
    description: Time between tokens
    importance: critical

throughput:
  - name: vllm:prompt_tokens
    description: Input tokens processed
    type: counter
  - name: vllm:generation_tokens
    description: Output tokens generated
    type: counter

resources:
  - name: vllm:kv_cache_usage_perc
    description: KV cache memory utilization
    importance: critical
    alert_threshold:
      warning: > 90%
      critical: > 98%
  - name: vllm:num_requests_running
    description: Requests currently being processed

cache:
  - name: vllm:prefix_cache_hits
    description: Prefix cache hits
  - name: vllm:prefix_cache_queries
    description: Total prefix cache queries
```

### 3. Recommended Alerts

```yaml
# alerts/vllm_alerts.yaml

groups:
  - name: vllm
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, vllm:time_to_first_token_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High TTFT latency (P99 > 500ms)"
          runbook: "https://docs.vllm.ai/runbooks/high-latency"

      - alert: HighKVCacheUsage
        expr: vllm:kv_cache_usage_perc > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "KV cache usage above 95%"
          runbook: "https://docs.vllm.ai/runbooks/memory-pressure"

      - alert: HighQueueDepth
        expr: vllm:num_requests_waiting > 100
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Request queue depth > 100"

      - alert: HighPreemptionRate
        expr: rate(vllm:num_preemptions[5m]) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High preemption rate"
```

### 4. Runbooks

```markdown
# runbooks/high-latency.md

## High Latency Alert

### Symptoms
- P99 TTFT > 500ms
- User-reported slow responses

### Investigation Steps

1. Check batch size
   ```
   vllm:num_requests_running
   ```
   - If high: reduce max_num_seqs

2. Check queue depth
   ```
   vllm:num_requests_waiting
   ```
   - If high: scale horizontally

3. Check prefill time
   - Long prompts may dominate
   - Enable chunked_prefill

4. Check KV cache
   ```
   vllm:kv_cache_usage_perc
   ```
   - If high: preemption causing delays

### Remediation

1. Scale horizontally
2. Reduce max_num_seqs
3. Enable chunked prefill
4. Increase GPU memory or add GPUs
```

### 5. Integration Guides

#### Grafana Setup

```bash
# Download dashboards
curl -O https://raw.githubusercontent.com/vllm-project/vllm/main/dashboards/grafana/vllm-overview.json

# Import via Grafana UI or API
```

#### Docker Compose

```yaml
# docker-compose.observability.yaml

services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    volumes:
      - ./dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
```

---

## Implementation Plan

### Phase 1: Dashboard Creation (Week 1-2)
- Design main dashboard
- Create Grafana JSON
- Test with real metrics

### Phase 2: Alerts and Runbooks (Week 2-3)
- Define alert thresholds
- Write runbooks
- Test alert conditions

### Phase 3: Integration (Week 3-4)
- Docker compose setup
- Kubernetes manifests
- Documentation

### Phase 4: Community Feedback (Week 4+)
- Release beta
- Gather feedback
- Iterate

---

## Backwards Compatibility

- Metrics unchanged
- Dashboards are additive
- No breaking changes

---

## Alternatives Considered

### 1. Built-in Dashboard

**Rejected:** Adds dependencies, better to use existing tools

### 2. DataDog/NewRelic Only

**Rejected:** Should support open-source first

---

## Open Questions

1. Should we include tracing dashboards?
2. How to handle custom metrics?
3. Should dashboards be versioned?

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| MTTR | 2 hours | 30 min |
| Dashboard adoption | 0% | 50% |
| Alert effectiveness | - | 80% actionable |

---

## Effort Estimate

- **Total:** 8 dev-days
- **Risk:** Low
- **Required Approvals:** Core team

---

*RFC based on analysis of commit `67745d189fd981ee824bde35666a3737a962c031`*
