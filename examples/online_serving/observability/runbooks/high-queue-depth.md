# High Queue Depth Runbook

## Alert Names
- `VLLMHighQueueDepth`
- `VLLMCriticalQueueDepth`

## Symptoms
- Request queue depth > 100
- Increased time-to-first-token
- User-reported timeouts
- Growing request backlog

## Impact
- Long wait times before processing starts
- Request timeouts
- Poor user experience
- Potential cascading failures

## Investigation Steps

### 1. Check Queue Depth Trend

```promql
# Current queue depth
sum(vllm:num_requests_waiting)

# Queue depth over time
sum(vllm:num_requests_waiting)[30m]
```

**If queue depth is growing:**
- Arrival rate exceeds processing rate
- Immediate scaling required

### 2. Check Request Processing Rate

```promql
# Requests completed per minute
sum(rate(vllm:request_success[5m])) * 60

# Running requests
sum(vllm:num_requests_running)
```

**If completion rate is low:**
- Check for latency issues
- May need to optimize batch processing

### 3. Check Throughput

```promql
# Generation throughput
sum(rate(vllm:generation_tokens_total[5m]))

# Prompt throughput
sum(rate(vllm:prompt_tokens_total[5m]))
```

**If throughput is lower than expected:**
- Check for resource constraints
- See [Low Throughput Runbook](./low-throughput.md)

### 4. Check KV Cache Usage

```promql
# KV cache utilization
avg(vllm:kv_cache_usage_perc)
```

**If KV cache is full:**
- New requests cannot be scheduled
- Queue builds up while waiting for memory
- See [Memory Pressure Runbook](./memory-pressure.md)

### 5. Check Request Characteristics

```promql
# Average request size
histogram_quantile(0.50, sum by(le) (rate(vllm:request_prompt_tokens_bucket[5m])))
```

**If requests are unusually large:**
- Large requests take longer to process
- Consider request limits

## Remediation Steps

### Immediate Actions

1. **Scale horizontally**
   Add more vLLM replicas immediately to handle backlog

2. **Enable load shedding**
   Return 503 for new requests when queue is too deep
   ```python
   # At load balancer level
   if queue_depth > 200:
       return 503, "Service temporarily overloaded"
   ```

3. **Prioritize requests**
   Process high-priority requests first if using request prioritization

### Short-term Fixes

1. **Increase batch size** (if latency allows)
   ```bash
   vllm serve model --max-num-seqs 256
   ```

2. **Reduce per-request latency**
   - Enable chunked prefill
   - Optimize batch tokens
   ```bash
   vllm serve model --enable-chunked-prefill --max-num-batched-tokens 8192
   ```

3. **Implement request queuing at gateway**
   - Set reasonable timeouts
   - Implement circuit breakers

### Long-term Solutions

1. **Auto-scaling**
   Configure Kubernetes HPA based on queue depth:
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: vllm-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: vllm
     minReplicas: 2
     maxReplicas: 10
     metrics:
       - type: External
         external:
           metric:
             name: vllm_requests_waiting
           target:
             type: Value
             value: "50"
   ```

2. **Request routing**
   Route requests to least-loaded instances

3. **Capacity planning**
   Monitor trends and provision ahead of demand

4. **Implement admission control**
   Reject requests early when system is overloaded

## Queue Management Best Practices

1. **Set appropriate timeouts**
   - Client timeout should exceed expected max wait time
   - Server should track time-in-queue

2. **Monitor queue wait time**
   ```promql
   histogram_quantile(0.99, sum by(le) (rate(vllm:request_queue_time_seconds_bucket[5m])))
   ```

3. **Implement backpressure**
   - Return 429 (Too Many Requests) when overloaded
   - Include Retry-After header

4. **Use request priorities**
   - Interactive requests: high priority
   - Batch jobs: low priority

## Escalation

If queue depth remains critical:

1. Verify all replicas are healthy
2. Check for network issues
3. Review recent traffic pattern changes
4. Consider emergency scaling

## Related Runbooks

- [High Latency](./high-latency.md)
- [Memory Pressure](./memory-pressure.md)
- [Low Throughput](./low-throughput.md)
