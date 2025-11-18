# High Latency Runbook

## Alert Names
- `VLLMHighTTFTLatency`
- `VLLMCriticalTTFTLatency`
- `VLLMHighITLLatency`

## Symptoms
- P99 Time to First Token (TTFT) > 500ms
- P99 Inter-Token Latency (ITL) > 100ms
- User-reported slow responses
- Increased request timeouts

## Impact
- Degraded user experience
- Potential request timeouts
- Increased queue depth as requests pile up

## Investigation Steps

### 1. Check Current Load

```promql
# Running requests
sum(vllm:num_requests_running)

# Waiting requests
sum(vllm:num_requests_waiting)
```

**If running requests are high:**
- The system may be overloaded
- Consider reducing `max_num_seqs` in vLLM configuration
- Scale horizontally by adding more replicas

### 2. Check Queue Depth

```promql
# Queue depth over time
sum(vllm:num_requests_waiting)
```

**If queue depth is high (> 50):**
- Requests are waiting too long before processing
- Scale horizontally to handle more concurrent requests
- Consider load shedding for non-critical requests

### 3. Check Prompt Sizes

```promql
# Average prompt tokens per request
histogram_quantile(0.99, sum by(le) (rate(vllm:request_prompt_tokens_bucket[5m])))
```

**If prompts are very long:**
- Long prompts require more prefill time, increasing TTFT
- Enable `chunked_prefill` to interleave prefill with decode
- Consider truncating system prompts if possible

### 4. Check KV Cache Usage

```promql
# KV cache utilization
avg(vllm:kv_cache_usage_perc)
```

**If KV cache usage is high (> 90%):**
- High cache pressure can cause preemptions
- Preempted requests must restart, increasing latency
- See [Memory Pressure Runbook](./memory-pressure.md)

### 5. Check Preemption Rate

```promql
# Preemptions per minute
sum(rate(vllm:num_preemptions[5m])) * 60
```

**If preemption rate is high:**
- Requests are being interrupted and restarted
- This directly increases latency
- Reduce batch size or increase GPU memory

## Remediation Steps

### Immediate Actions

1. **Scale horizontally** - Add more vLLM replicas to distribute load
2. **Reduce batch size** - Lower `max_num_seqs` to reduce per-request latency
3. **Enable request prioritization** - Prioritize latency-sensitive requests

### Short-term Fixes

1. **Enable chunked prefill**
   ```bash
   vllm serve model --enable-chunked-prefill
   ```
   This allows prefill to be interleaved with decode, improving TTFT for long prompts.

2. **Tune scheduling parameters**
   ```bash
   vllm serve model --max-num-seqs 128 --max-num-batched-tokens 4096
   ```

3. **Review GPU memory allocation**
   ```bash
   vllm serve model --gpu-memory-utilization 0.95
   ```

### Long-term Solutions

1. **Upgrade GPU hardware** - More VRAM allows larger batches with lower latency
2. **Use tensor parallelism** - Distribute model across multiple GPUs
3. **Optimize model** - Use quantization to reduce memory footprint
4. **Implement caching layer** - Cache common responses upstream

## Escalation

If latency remains high after following these steps:

1. Check vLLM logs for errors or warnings
2. Profile GPU utilization with `nvidia-smi`
3. Review recent configuration or model changes
4. Consider opening a GitHub issue with detailed metrics

## Related Runbooks

- [Memory Pressure](./memory-pressure.md)
- [High Queue Depth](./high-queue-depth.md)
- [Low Throughput](./low-throughput.md)
