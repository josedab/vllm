# vLLM Observability Stack

This directory contains a complete observability setup for vLLM deployments, including Prometheus metrics, Grafana dashboards, alerting rules, and operational runbooks.

## Quick Start

### 1. Start vLLM with Metrics

```bash
vllm serve your-model --port 8000
```

Metrics are available at `http://localhost:8000/metrics`

### 2. Start the Monitoring Stack

```bash
cd examples/online_serving/observability
docker-compose up -d
```

This starts:
- **Prometheus** at http://localhost:9090
- **Grafana** at http://localhost:3000 (admin/vllm-monitoring)
- **Alertmanager** at http://localhost:9093

### 3. Access Dashboards

1. Open Grafana at http://localhost:3000
2. Login with admin/vllm-monitoring
3. Navigate to the vLLM folder to find pre-built dashboards

## Components

### Dashboards

Located in `../dashboards/grafana/`:

| Dashboard | Description |
|-----------|-------------|
| `vllm_overview.json` | Key metrics overview (throughput, latency, queue, memory) |
| `resource_utilization.json` | GPU memory, KV cache, prefix cache, preemption metrics |
| `performance_statistics.json` | Detailed latency and throughput analysis |
| `query_statistics.json` | Request patterns and token distributions |

### Metrics Organization

See `metrics/organized_metrics.yaml` for a categorized list of all vLLM metrics:

- **Latency**: TTFT, ITL, E2E latency
- **Throughput**: Prompt/generation tokens per second
- **Resources**: KV cache usage, running/waiting requests
- **Cache**: Prefix cache hit rate, multi-modal cache
- **Requests**: Success counts, parameter distributions

### Alerting Rules

Pre-configured alerts in `alerts/vllm_alerts.yaml`:

| Alert | Severity | Condition |
|-------|----------|-----------|
| VLLMHighTTFTLatency | Warning | P99 TTFT > 500ms |
| VLLMCriticalTTFTLatency | Critical | P99 TTFT > 1s |
| VLLMHighKVCacheUsage | Warning | KV cache > 95% |
| VLLMCriticalKVCacheUsage | Critical | KV cache > 98% |
| VLLMHighQueueDepth | Warning | Queue > 100 |
| VLLMHighPreemptionRate | Warning | Preemptions > 1/min |

### Runbooks

Operational procedures in `runbooks/`:

- `high-latency.md` - Debugging and fixing latency issues
- `memory-pressure.md` - Handling KV cache pressure and preemptions
- `high-queue-depth.md` - Managing request backlogs

## Configuration

### Prometheus

Edit `prometheus/prometheus.yaml` to configure:

- **Scrape targets**: Update vLLM endpoint addresses
- **Scrape interval**: Default is 5s for vLLM metrics
- **Remote write**: Optionally send to long-term storage

### Alertmanager

Edit `alertmanager/alertmanager.yaml` to configure:

- **Slack notifications**: Add webhook URL
- **PagerDuty**: Add service key
- **Email**: Configure SMTP settings
- **Custom routing**: Define alert routing rules

### Grafana

Dashboards are auto-provisioned from the dashboards directory. To add custom dashboards:

1. Create JSON dashboard in `../dashboards/grafana/`
2. Restart Grafana or wait for auto-reload

## Kubernetes Deployment

For Kubernetes deployments, wrap the alert rules in a PrometheusRule custom resource:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vllm-alerts
  namespace: monitoring
spec:
  groups:
    # Copy content from alerts/vllm_alerts.yaml
```

For Grafana dashboards, use the Grafana Operator's GrafanaDashboard resource.

## Best Practices

### Monitoring

1. **Set up alerts first** - Don't wait for incidents
2. **Monitor the queue** - High queue depth predicts latency issues
3. **Track KV cache** - Memory pressure causes preemptions
4. **Watch preemption rate** - Early indicator of capacity issues

### Capacity Planning

1. **Monitor P99 latency** - SLO target should be achievable
2. **Track throughput trends** - Plan scaling based on growth
3. **Set resource limits** - Prevent runaway memory usage

### Alerting

1. **Use severity levels** - Critical alerts should be rare
2. **Include runbook links** - Help on-call engineers
3. **Group related alerts** - Reduce noise
4. **Set appropriate thresholds** - Tune based on your SLOs

## Troubleshooting

### Prometheus can't scrape vLLM

1. Verify vLLM is running: `curl http://localhost:8000/metrics`
2. Check Prometheus targets: http://localhost:9090/targets
3. If using Docker on Mac/Windows, use `host.docker.internal` instead of `localhost`

### Grafana shows no data

1. Check Prometheus datasource is configured
2. Verify the metric names match (use Prometheus UI to explore)
3. Check time range selection in Grafana

### Alerts not firing

1. Check Prometheus rules: http://localhost:9090/rules
2. Verify Alertmanager configuration
3. Test with lower thresholds temporarily

## Customization

### Adding Custom Metrics

vLLM supports custom stat logger plugins. See the vLLM documentation for creating custom loggers.

### Creating Custom Dashboards

1. Export dashboard JSON from Grafana
2. Replace datasource UID with `${DS_PROMETHEUS}` variable
3. Add to `../dashboards/grafana/`
4. Update this documentation

### Modifying Alert Thresholds

Edit `alerts/vllm_alerts.yaml` and reload Prometheus:

```bash
curl -X POST http://localhost:9090/-/reload
```

## Contributing

When adding new observability features:

1. Update metrics organization in `metrics/organized_metrics.yaml`
2. Add relevant alerts in `alerts/vllm_alerts.yaml`
3. Create runbooks for new alert types
4. Update dashboards to visualize new metrics
5. Update this README

## Related Documentation

- [vLLM Metrics Documentation](https://docs.vllm.ai/en/latest/serving/metrics.html)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/)
- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
