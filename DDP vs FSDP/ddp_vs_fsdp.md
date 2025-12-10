# Why Distributed Training Exists in the First Place

Before comparing DDP and FSDP, it helps to restate the fundamental constraint behind all distributed training:

**Large models do not fit on a single GPU.**

Not in terms of:
- parameters,
- gradients,
- optimizer states,
- activations produced during forward pass.

If you visualize memory usage for a model of size P, the actual footprint per GPU is closer to:

```
Total Memory ≈ P (params)
             + P (grads)
             + 2P (optimizer states for Adam)
             + activations (depends on depth, sequence length)
```

Even a "2B parameter model" can silently blow up to 15–20 GB once optimizer states and activations are accounted for.

Distributed training exists to partition this footprint across multiple devices.

And the core question every system has to answer is:

**What gets replicated and what gets sharded?**

This question is what separates DDP and FSDP.

---

## Distributed Data Parallel (DDP)
### Replicate the Model, Split the Data

DDP is the mental model most engineers start with because the system behavior mirrors the conceptual one.

**DDP Visual Overview (4 GPUs)**
```
GPU0: [Full Model]
GPU1: [Full Model]
GPU2: [Full Model]
GPU3: [Full Model]
```

Each GPU:
- receives a different microbatch,
- computes forward + backward independently,
- synchronizes gradients across devices,
- applies identical optimizer updates.

**What DDP Replicates**
```
Params:       Replicated
Gradients:    Replicated (before all-reduce)
Opt States:   Replicated
Activations:  Replicated
```

This yields a very simple mental model:

```python
# DDP is compute-parallel
# Every GPU performs the same computation on different data
```

**DDP Backward Pass Communication**
```
dW0   dW1   dW2   dW3
  \     \     \     \
   ------ All-Reduce ------
  /     /     /     /  
dW0=dW1=dW2=dW3 (averaged)
```

All GPUs end with identical gradients.

### Why DDP Breaks for Large Models

DDP requires:
- full parameter memory,
- full gradient memory,
- full optimizer memory,
- on every device.

If the model is larger than the memory of one GPU, even by a few hundred MB, DDP cannot run. You can buy more GPUs, but you cannot buy more VRAM per GPU.

**DDP Works Best When:**
- The model fits on each GPU.
- You want simplicity and high throughput.
- Model size is < 2–3B parameters.
- You are running research experiments, prototyping, or training CNNs/medium-sized Transformers.

---

## Fully Sharded Data Parallel (FSDP)
### Shard Everything, Reconstruct Just in Time

FSDP takes a radically different approach.

Instead of replicating the whole model everywhere, FSDP shards:
- parameters,
- gradients,
- optimizer states,
- and optionally activations.

**FSDP Visual Overview (Parameters Sharded Across 4 GPUs)**
```
GPU0: [P0]
GPU1: [P1]
GPU2: [P2]
GPU3: [P3]
```

Each GPU holds only 1/4 of the model.

But during forward/backward of a particular layer, FSDP temporarily reconstructs the full parameters using all-gather, computes, and discards.

**FSDP Forward Pass (per layer)**
```
1. All-Gather shards → reconstruct W_full
2. Compute y = f(x, W_full)
3. Free W_full → keep only shard
```

**FSDP Backward Pass**
```
1. All-Gather W_full (optional for backward)
2. Compute dW_full
3. Reduce-Scatter → keep only dW_shard
4. Update optimizer states for shard only
```

### Why FSDP Enables Large-Model Training

Instead of requiring each GPU to store the full parameter + gradient + optimizer memory footprint, it divides them:

```
Per-GPU Memory (FSDP) ≈ (P + G + O) / world_size
```

This turns a 60B parameter model from:

**Impossible with DDP → Feasible with FSDP**

**FSDP Works Best When:**
- Your model does not fit in GPU memory.
- You are training LLMs or massive multimodal models.
- You want ZeRO-3–level memory efficiency inside PyTorch.
- Optimizer states dominate memory usage (Adam, Adafactor).
- You need model-parallel scaling without rewriting architectures.

---

## How to Think About the Difference

Here's the intuition that matters:

> **DDP = compute parallelism**  
> **FSDP = memory parallelism**

**DDP says:**  
"Everyone has a copy; let's compute in parallel."

**FSDP says:**  
"No one has a full copy; let's cooperate to simulate one."

### A Physical Analogy

If DDP is a team where:
- Each person lifts their own 50-lb weight independently,

FSDP is a team where:
- Everyone collaborates to lift a single 500-lb weight by each holding only their fraction at a time.

**DDP parallelizes work.**  
**FSDP parallelizes capacity.**

---

## Pseudocode — How They Actually Operate

Below is the low-level intuition reflected in their pseudocode.

### DDP Pseudocode

```python
# Replicated Model on All GPUs
model = DistributedDataParallel(model)

for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(batch)
    loss = loss_fn(outputs)
    
    # triggers gradient all-reduce internally
    loss.backward()
    
    optimizer.step()
```

Under the hood:

```python
for each parameter p:
    p.grad = all_reduce(p.grad) / world_size
```

Memory is never reduced. Only compute is parallelized.

### FSDP Pseudocode (Layer-Wise)

```python
# Parameters are sharded across GPUs
fsdp_model = FullyShardedDataParallel(model)
```

**Forward (per layer)**
```python
def fsdp_forward(layer, x):
    W_full = all_gather(layer.weight_shard)
    y = layer(x, W_full)
    del W_full
    return y
```

**Backward (per layer)**
```python
def fsdp_backward(layer, grad_out):
    W_full = all_gather(layer.weight_shard)
    grad_full = compute_grad(layer, grad_out, W_full)
    grad_shard = reduce_scatter(grad_full)
    update_optimizer(layer.weight_shard, grad_shard)
    del W_full
```

Memory shrinks linearly with the number of GPUs.

---

## Bringing It All Together

Here is the summary mental model you should leave with:

### DDP
- Simple, fast, stable.
- Replicates everything.
- Breaks when model > GPU memory.
- Ideal for small/medium models.

### FSDP
- Complex, memory-efficient.
- Shards everything.
- Trains models far larger than a single GPU can hold.
- Ideal for LLM-scale workloads.

---

If DDP is the starting point, FSDP is what you reach for when the model no longer fits inside that starting point.

And once you see the difference in terms of memory movement and communication topology, the entire landscape of large-model training feels less magical and more like a carefully designed distributed system.
