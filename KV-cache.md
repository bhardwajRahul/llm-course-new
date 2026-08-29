[cagvault](https://github.com/letslego/cagvault)

```

┌─────────────────────────────────────────────────────────────────────────────┐
│               CACHE-AUGMENTED GENERATION (CAG) WORKFLOW                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─── SETUP PHASE (One-time) ─────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  All Documents                                                     │   │
│  │      │                                                             │   │
│  │      ▼                                                             │   │
│  │  ┌──────────────────┐                                             │   │
│  │  │  LLM Processor   │  Precompute KV-Cache                        │   │
│  │  │  (Batch Process) │  (Encodes all knowledge)                    │   │
│  │  └────────┬─────────┘                                             │   │
│  │           │                                                        │   │
│  │           ▼                                                        │   │
│  │  ┌──────────────────────┐                                         │   │
│  │  │  Cached KV-State     │  💾  Stored on Disk/Memory             │   │
│  │  │  (Ready to use)      │                                         │   │
│  │  └──────────┬───────────┘                                         │   │
│  │             │                                                     │   │
│  └─────────────┼─────────────────────────────────────────────────────┘   │
│                │                                                           │
│  ┌─── INFERENCE PHASE (Fast) ────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  User Query        Cached KV-State                               │   │
│  │      │                  │                                        │   │
│  │      └──────────┬───────┘                                        │   │
│  │                 ▼                                                │   │
│  │        ┌──────────────────────┐                                 │   │
│  │        │  LLM with Preloaded  │  ✨ NO RETRIEVAL!               │   │
│  │        │  Context + KV-Cache  │  ✨ NO LATENCY!                │   │
│  │        │                      │  ✨ GUARANTEED CONTEXT!        │   │
│  │        └──────────┬───────────┘                                 │   │
│  │                   │                                              │   │
│  │                   ▼                                              │   │
│  │              Answer (Instant)                                    │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌─── MULTI-TURN OPTIMIZATION ───────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  For next query: Simply truncate and reuse cached knowledge     │   │
│  │  (No need to reprocess documents)                              │   │
│  │                                                                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

[vLLM + LMCache: A Starter Guide, No GPU Required](https://blog.lmcache.ai/en/2026/06/23/vllm-lmcache-a-starter-guide-no-gpu-required/)

- ① What problem LMCache solves;

- ② How to set up your workspace on a MacBook;

- ③ How to run vLLM + LMCache end to end;

- ④ What to do when you run out of memory, or when the model won’t download;

- ⑤ What specifically you can work on across four directions.


[KV, Prefix, Prompt and Semantic Caching in LLMs, clearly explained](https://x.com/_avichawla/status/2093265776266637739)

- `get_seq_length()` then reports how many token positions the cache holds. When you run this, the output contains the prompt length plus the tokens generated, minus one.
  

<img width="505" height="269" alt="Screenshot 2026-08-29 at 2 31 22 PM" src="https://github.com/user-attachments/assets/450c7c6b-cb1c-44cb-ad35-6d77ac85a770" />

*** For a 70B model at BF16, a single 128K context holds around 40 GB of cache, comparable to the entire model at 4-bit weights.

<img width="500" height="276" alt="Screenshot 2026-08-29 at 2 33 43 PM" src="https://github.com/user-attachments/assets/36e06e2f-429e-4501-9c94-ff7f458125f1" />



```
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
)

SHARED_PREFIX = """You are a careful assistant. 
                   Answer in one short sentence."""

prompt_cache = StaticCache(config=model.config, max_cache_len=1024)

prefix_inputs = tokenizer(SHARED_PREFIX, return_tensors="pt")
prefix_inputs = prefix_inputs.to(model.device)

# Prefill the shared prefix exactly once. No token is sampled here.
with torch.no_grad():
    prompt_cache = model(**prefix_inputs, past_key_values=prompt_cache)
    prompt_cache = prompt_cache.past_key_values

questions = ["What is the capital of France?", "Name one ocean."]

for question in questions:
    inputs = tokenizer(SHARED_PREFIX + question, return_tensors="pt")
    inputs = inputs.to(model.device)

    # each request gets its own copy
    past_key_values = copy.deepcopy(prompt_cache)   

    outputs = model.generate(
        **inputs, past_key_values=past_key_values, do_sample=False
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```





