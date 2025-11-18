# Grafana Dashboards for vLLM Monitoring

This directory contains Grafana dashboard configurations (as JSON) designed to monitor
vLLM performance and metrics.

## Requirements

- Grafana 8.0+
- Prometheus data source configured in Grafana
- vLLM deployment with Prometheus metrics enabled

## Dashboard Descriptions

- **vllm_overview.json**: Main overview dashboard with key metrics at a glance -
  throughput, latency, queue depth, and memory utilization.
- **resource_utilization.json**: Detailed resource metrics including KV cache usage,
  prefix cache hit rates, and preemption tracking.
- **performance_statistics.json**: Tracks performance metrics including latency and
  throughput for your vLLM service.
- **query_statistics.json**: Tracks query performance, request volume, and key
  performance indicators for your vLLM service.

## Quick Setup with Observability Stack

For a complete monitoring setup with pre-configured Prometheus and Grafana, see the
[observability stack](../observability/README.md) which includes:

- Docker Compose setup
- Pre-configured alerting rules
- Operational runbooks

## Deployment Options

### Manual Import (Recommended)

The easiest way to use these dashboards is to manually import the JSON configurations
directly into your Grafana instance:

1. Navigate to your Grafana instance
2. Click the '+' icon in the sidebar
3. Select 'Import'
4. Copy and paste the JSON content from the dashboard files, or upload the JSON files
   directly

### Grafana Operator

If you're using the [Grafana Operator](https://github.com/grafana-operator/grafana-operator)
in Kubernetes, you can wrap these JSON configurations in a `GrafanaDashboard` custom
resource:

```yaml
# Note: Adjust the instanceSelector to match your Grafana instance's labels
# You can check with: kubectl get grafana -o yaml
apiVersion: grafana.integreatly.org/v1beta1
kind: GrafanaDashboard
metadata:
  name: vllm-performance-dashboard
spec:
  instanceSelector:
    matchLabels:
      dashboards: grafana  # Adjust to match your Grafana instance labels
  folder: "vLLM Monitoring"
  json: |
    # Replace this comment with the complete JSON content from
    # performance_statistics.json - The JSON should start with { and end with }
```

Then apply to your cluster:

```bash
kubectl apply -f your-dashboard.yaml -n <namespace>
```
