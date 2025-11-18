# Extending and Integrating vLLM

**Part 4 of the vLLM Technical Blog Series**
**Commit SHA:** `67745d189fd981ee824bde35666a3737a962c031`

---

## What You'll Learn

- Extension points and plugin architecture
- Adding new models to vLLM
- Creating custom attention backends
- API integration patterns
- Structured output and constrained generation

---

## Introduction

vLLM is designed to be extensible. Whether you need to add support for a new model architecture, integrate with your existing infrastructure, or customize generation behavior, vLLM provides clear extension points.

In this post, we'll explore how to extend vLLM for your specific needs.

---

## Adding a New Model

Let's walk through adding support for a new model architecture.

### Step 1: Create the Model File

```python
# vllm/model_executor/models/my_model.py

from vllm.model_executor.models.llama import LlamaForCausalLM

class MyModelForCausalLM(LlamaForCausalLM):
    """My custom model based on Llama architecture."""

    def __init__(self, config, ...):
        super().__init__(config, ...)
        # Add custom components

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        # Custom forward pass
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        logits = self.lm_head(hidden_states)
        return logits
```

### Step 2: Register the Model

```python
# vllm/model_executor/models/registry.py

_MODELS = {
    # ... existing models
    "MyModelForCausalLM": ("my_model", "MyModelForCausalLM"),
}
```

### Step 3: Handle Weight Loading

If your model has different weight names:

```python
# vllm/model_executor/models/my_model.py

class MyModelForCausalLM(LlamaForCausalLM):
    # Map HuggingFace weight names to vLLM names
    stacked_params_mapping = {
        "qkv_proj": {
            "q_proj": 0,
            "k_proj": 1,
            "v_proj": 2,
        },
        "gate_up_proj": {
            "gate_proj": 0,
            "up_proj": 1,
        },
    }

    def load_weights(self, weights):
        # Custom weight loading logic
        for name, weight in weights:
            # Transform weight names
            param = self._get_param(name)
            param.copy_(weight)
```

### Step 4: Add Tests

```python
# tests/models/test_my_model.py

import pytest
from tests.utils import VllmRunner

@pytest.mark.parametrize("model", ["my-org/my-model"])
def test_my_model(vllm_runner, model):
    with vllm_runner(model) as vllm:
        output = vllm.generate(["Hello, world!"])
        assert len(output) > 0
```

---

## Creating an Attention Backend

Attention backends must implement the `AttentionBackend` interface:

```python
# vllm/attention/backends/my_backend.py

from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl

class MyAttentionBackend(AttentionBackend):
    """My custom attention implementation."""

    @staticmethod
    def get_name() -> str:
        return "MY_BACKEND"

    @staticmethod
    def get_impl_cls():
        return MyAttentionImpl

    @staticmethod
    def get_supported_head_sizes():
        return [64, 128, 256]

class MyAttentionImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, scale, ...):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Your attention implementation
        if attn_metadata.is_prompt:
            return self._prefill_attention(query, key, value)
        else:
            return self._decode_attention(query, kv_cache, attn_metadata)

    def _prefill_attention(self, query, key, value):
        # Prefill implementation
        pass

    def _decode_attention(self, query, kv_cache, attn_metadata):
        # Decode implementation with paged KV cache
        pass
```

### Register the Backend

```python
# vllm/attention/selector.py

_BACKENDS["MY_BACKEND"] = MyAttentionBackend
```

### Use the Backend

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    # Use custom backend
)
# Set via environment variable:
# VLLM_ATTENTION_BACKEND=MY_BACKEND
```

---

## API Integration Patterns

### OpenAI-Compatible API

vLLM's API is a drop-in replacement for OpenAI:

```python
from openai import OpenAI

# Point to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Custom Endpoints

Add custom endpoints to the API server:

```python
# vllm/entrypoints/openai/api_server.py

from fastapi import APIRouter

router = APIRouter()

@router.post("/v1/my-endpoint")
async def my_endpoint(request: MyRequest):
    """Custom endpoint for specialized functionality."""
    # Process request
    result = await engine_client.generate(...)
    return MyResponse(result=result)
```

---

## Structured Output

vLLM supports constraining generation to specific formats.

### JSON Schema

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
    },
    "required": ["name", "age", "city"]
}

params = SamplingParams(
    max_tokens=100,
    guided_decoding={
        "json": schema
    }
)

output = llm.generate(
    ["Generate a person's info:"],
    params
)
# Output will be valid JSON matching schema
```

### Regex Patterns

```python
params = SamplingParams(
    max_tokens=50,
    guided_decoding={
        "regex": r"\d{3}-\d{3}-\d{4}"  # Phone number
    }
)
```

### Grammar-Based

```python
params = SamplingParams(
    max_tokens=100,
    guided_decoding={
        "grammar": """
        start: sentence+
        sentence: subject verb object "."
        subject: "The" noun
        verb: "eats" | "sees" | "likes"
        object: "the" noun
        noun: "cat" | "dog" | "bird"
        """
    }
)
```

---

## LoRA Adapters

vLLM supports serving multiple LoRA adapters simultaneously:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=4,
    max_lora_rank=64
)

# Define LoRA adapters
lora_request_1 = LoRARequest(
    lora_name="adapter1",
    lora_int_id=1,
    lora_local_path="/path/to/adapter1"
)

lora_request_2 = LoRARequest(
    lora_name="adapter2",
    lora_int_id=2,
    lora_local_path="/path/to/adapter2"
)

# Generate with different adapters
output1 = llm.generate(
    ["Prompt for adapter 1"],
    SamplingParams(max_tokens=100),
    lora_request=lora_request_1
)

output2 = llm.generate(
    ["Prompt for adapter 2"],
    SamplingParams(max_tokens=100),
    lora_request=lora_request_2
)
```

---

## Custom Sampling

### Custom Logits Processor

```python
from vllm import SamplingParams

def my_logits_processor(token_ids, logits):
    """Custom logits processing."""
    # Example: Boost probability of specific tokens
    special_token_id = 1234
    logits[special_token_id] += 10.0
    return logits

params = SamplingParams(
    max_tokens=100,
    logits_processors=[my_logits_processor]
)
```

### Stop Conditions

```python
params = SamplingParams(
    max_tokens=500,
    stop=["```", "\n\n\n"],  # Stop on these strings
    stop_token_ids=[2],      # Stop on these token IDs
    include_stop_str_in_output=False
)
```

---

## Embedding Integration

vLLM can serve embedding models:

```python
from vllm import LLM

# Load embedding model
llm = LLM(model="BAAI/bge-base-en-v1.5", task="embed")

# Generate embeddings
embeddings = llm.encode(["Hello world", "How are you?"])

for i, emb in enumerate(embeddings):
    print(f"Text {i}: {len(emb.outputs.embedding)} dimensions")
```

### API Integration

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")

response = client.embeddings.create(
    model="BAAI/bge-base-en-v1.5",
    input=["Search query"]
)

embedding = response.data[0].embedding
```

---

## Multi-Modal Integration

vLLM supports vision-language models:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    max_model_len=4096
)

# With image input
prompt = "<image>\nDescribe this image."
image_path = "/path/to/image.jpg"

output = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": image_path}
    }],
    SamplingParams(max_tokens=200)
)
```

---

## Monitoring Integration

### Prometheus Metrics

```python
import requests

# Get metrics
response = requests.get("http://localhost:8000/metrics")
print(response.text)

# Key metrics:
# vllm:num_requests_running
# vllm:num_requests_waiting
# vllm:kv_cache_usage_perc
# vllm:time_to_first_token_seconds
```

### Custom Metrics

```python
from prometheus_client import Counter, Histogram

# Add custom metrics
my_counter = Counter(
    "my_custom_requests",
    "Custom request counter",
    ["model", "status"]
)

my_histogram = Histogram(
    "my_custom_latency",
    "Custom latency histogram"
)

# Use in your code
my_counter.labels(model="llama", status="success").inc()
with my_histogram.time():
    result = process_request()
```

### OpenTelemetry

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --otlp-traces-endpoint http://collector:4317 \
    --collect-detailed-traces model,worker
```

---

## Plugin System

vLLM has a plugin system for extending functionality:

```python
# Register a plugin via entry points
# In your package's setup.py or pyproject.toml:

[project.entry-points."vllm.general_plugins"]
my_plugin = "my_package.plugin:register"

# In my_package/plugin.py:
def register():
    """Register custom functionality."""
    from vllm.model_executor.models import _MODELS
    _MODELS["MyCustomModel"] = ("my_module", "MyCustomModel")
```

---

## Key Takeaways

1. **Model registration** via the registry pattern makes adding models straightforward

2. **Attention backends** implement a clear interface for custom attention

3. **OpenAI-compatible API** enables drop-in replacement

4. **Structured output** (JSON, regex, grammar) constrains generation

5. **LoRA support** enables multi-adapter serving

6. **Multi-modal** integration supports vision-language models

7. **Observability** through Prometheus and OpenTelemetry

---

## Next Steps

In [Part 5: Performance Analysis](05-performance-analysis.md), we'll explore vLLM's performance characteristics, profiling tools, and optimization techniques including quantization and CUDA graphs.

---

## Code References

- **Model Registry:** [vllm/model_executor/models/registry.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/model_executor/models/registry.py)
- **Attention Backends:** [vllm/attention/backends/](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/attention/backends/)
- **API Server:** [vllm/entrypoints/openai/api_server.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/entrypoints/openai/api_server.py)
- **LoRA:** [vllm/lora/](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/lora/)

---

*This is Part 4 of the vLLM Technical Blog Series based on commit `67745d189fd981ee824bde35666a3737a962c031`.*
