# Troubleshooting Guides

This directory contains detailed troubleshooting guides for common issues in vLLM.

## Available Guides

- [Memory Issues](memory_issues.md) - OOM errors, memory optimization, and memory estimation
- [Performance Issues](performance_issues.md) - Throughput optimization, latency tuning, and benchmarking

## Quick Reference

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | [Memory Issues](memory_issues.md#out-of-memory-oom-errors) |
| Low throughput | [Performance Issues](performance_issues.md#low-throughput) |
| High latency | [Performance Issues](performance_issues.md#high-latency) |
| Slow model loading | [Performance Issues](performance_issues.md#model-loading-issues) |

### General Troubleshooting

For other issues, see the [main troubleshooting guide](../usage/troubleshooting.md).

### Distributed Issues

For multi-GPU and multi-node issues, see:
- [Distributed Troubleshooting](../serving/distributed_troubleshooting.md)
- [Parallelism and Scaling](../serving/parallelism_scaling.md)
