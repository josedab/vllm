# Memory Pressure Runbook

## Alert Names
- `VLLMHighKVCacheUsage`
- `VLLMCriticalKVCacheUsage`
- `VLLMHighPreemptionRate`

## Symptoms
- KV cache usage > 90%
- High preemption rate
- Increasing latency
- Requests being interrupted and restarted

## Impact
- Preempted requests must restart from scratch
- Increased latency and resource waste
- Potential OOM errors in severe cases
- Degraded user experience

## Investigation Steps

### 1. Check KV Cache Utilization

```promql
# Current KV cache usage
avg(vllm:kv_cache_usage_perc)

# KV cache usage over time
avg(vllm:kv_cache_usage_perc)[1h]
```

**If KV cache usage is sustained > 90%:**
- System is memory-bound
- New requests cannot be scheduled
- Preemptions will occur

### 2. Check Preemption Rate

```promql
# Preemptions per minute
sum(rate(vllm:num_preemptions[5m])) * 60

# Total preemptions in last hour
sum(increase(vllm:num_preemptions[1h]))
```

**If preemption rate is > 1/min:**
- Active requests are being interrupted
- This wastes GPU compute and increases latency
- Urgent action required

### 3. Check Concurrent Requests

```promql
# Running requests
sum(vllm:num_requests_running)

# Average generation tokens per request
histogram_quantile(0.50, sum by(le) (rate(vllm:request_generation_tokens_bucket[5m])))
```

**If many long-running requests:**
- Long generations consume more KV cache
- Consider limiting `max_tokens` per request

### 4. Check Request Patterns

```promql
# Distribution of prompt lengths
histogram_quantile(0.99, sum by(le) (rate(vllm:request_prompt_tokens_bucket[5m])))

# Distribution of generation lengths
histogram_quantile(0.99, sum by(le) (rate(vllm:request_generation_tokens_bucket[5m])))
```

**If prompts or generations are very long:**
- Each request consumes significant KV cache
- Consider truncation or request limits

## Remediation Steps

### Immediate Actions

1. **Reduce concurrent requests**
   ```bash
   # Reduce max_num_seqs
   vllm serve model --max-num-seqs 64
   ```

2. **Limit max generation length**
   Set `max_tokens` limit at API gateway level

3. **Scale horizontally**
   Add more replicas to distribute memory pressure

### Short-term Fixes

1. **Increase GPU memory utilization**
   ```bash
   vllm serve model --gpu-memory-utilization 0.95
   ```
   This allocates more GPU memory for KV cache.

2. **Enable prefix caching**
   ```bash
   vllm serve model --enable-prefix-caching
   ```
   Reuses KV cache for common prompt prefixes.

3. **Tune swap space** (if using CPU offload)
   ```bash
   vllm serve model --swap-space 8
   ```

### Long-term Solutions

1. **Upgrade GPU hardware**
   - More VRAM directly increases KV cache capacity
   - Consider A100 80GB or H100 for large context workloads

2. **Use tensor parallelism**
   ```bash
   vllm serve model --tensor-parallel-size 2
   ```
   Distributes model and KV cache across multiple GPUs.

3. **Use quantization**
   ```bash
   vllm serve model --quantization awq
   ```
   Reduces model memory footprint, leaving more for KV cache.

4. **Enable KV cache compression** (when available)
   Compresses KV cache entries to fit more requests.

5. **Implement request routing**
   Route requests based on context length to specialized instances.

## Capacity Planning

To estimate KV cache requirements:

```
KV Cache per token ≈ 2 * num_layers * hidden_size * 2 bytes (fp16)
Total KV Cache = num_requests * (prompt_length + max_generation) * cache_per_token
```

For example, Llama-2-7B:
- 32 layers, 4096 hidden size
- ≈ 0.5 MB per 1K tokens per request
- 100 requests * 4K context = 200 GB KV cache

## Escalation

If memory pressure persists:

1. Review traffic patterns for anomalies
2. Check for memory leaks in vLLM logs
3. Profile with `nvidia-smi` and `torch.cuda.memory_summary()`
4. Consider emergency capacity expansion

## Related Runbooks

- [High Latency](./high-latency.md)
- [High Queue Depth](./high-queue-depth.md)
