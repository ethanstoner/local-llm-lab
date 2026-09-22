# Methodology

The decisions behind the numbers, and why each one is made the way it is. Most of these
exist because the obvious alternative produces a figure that looks fine and is wrong.

---

## 1. Timing

### Synchronization

CUDA kernel launches are asynchronous. A timer stopped immediately after
`model.generate` returns measures how long it took to *enqueue* the work, not to do it.
Every timed region in `src/benchmarks/latency.py` ends with `torch.cuda.synchronize()`
before the clock is read.

### Prefill and decode are separate regimes

Processing a prompt and generating a token are different computations:

| | prefill | decode |
|---|---|---|
| parallelism | all prompt tokens at once | one token |
| bottleneck | arithmetic | memory bandwidth |
| scaling | ~linear in prompt length, plus attention's quadratic term | flat in prompt length until the KV cache gets large |

A single "tokens per second" figure averages the two and hides both. This project
measures and reports:

* `prefill_latency_s` - a dedicated forward pass over the prompt with `use_cache=True`,
  timed on its own.
* `ttft_s` - time to first token, stamped inside `generate` when the first token's
  logits become available.
* `decode_tokens_per_s` - `(new_tokens - 1) / (t_last - t_first)`, which excludes prefill
  entirely.
* `end_to_end_tokens_per_s` - `new_tokens / total_time`, which includes it. This is what
  a user experiences and it is always the lower of the two.

`ttft_s` and `prefill_latency_s` are two independent measurements of nearly the same
quantity. They are both reported so they can be checked against each other; a large gap
between them indicates a measurement problem rather than a model property.

### Per-token timestamps

`TokenTimestampProcessor` is a `LogitsProcessor`. Transformers calls the processor list
once per generated token, immediately after the forward pass that produced those logits,
so stamping the clock there yields time-to-first-token and the full inter-token latency
distribution from one generation call - no second pass, no streaming thread.

It synchronizes before stamping. Without that it would record when work was queued.
Synchronizing once per token gives up some CPU/GPU overlap, so the harness *measures*
that cost rather than assuming it away: `measure_sync_overhead` runs the same
configuration with and without per-token synchronization and records the difference in
`validation.sync_overhead` in every benchmark run.

### Fixed token budgets

Generation uses `min_new_tokens == max_new_tokens`. Without it, a run that emits an
end-of-sequence marker early produces fewer tokens in less time, and dividing one by the
other yields a rate that flatters whichever configuration happened to stop soonest. This
is the single most common way a quantization comparison lies: the 4-bit model degrades,
starts producing shorter answers, and appears to be "faster".

### Warm-up and repeats

One generation per configuration is run and discarded before measurement, so kernel
selection, allocator growth and cache warming are not counted. PyTorch's peak-memory
counters are reset after the warm-up for the same reason. Then N repeats, reported as
**median** and standard deviation - a single scheduler hiccup or a background process
waking up should not move the headline number.

---

## 2. Memory

Three numbers, all reported, because they answer different questions:

| Number | Source | What it means |
|---|---|---|
| `allocator.max_reserved_mib` | `torch.cuda.max_memory_reserved` | what PyTorch's caching allocator held |
| `gpu.peak_memory_used_mib` | NVML sampling thread | what the device held, including the CUDA context, cuBLAS workspaces and every other process |
| `baseline_device_used_mib` | NVML, before loading | what was already in use before this experiment started |

On this machine the baseline is 1.5-2.2 GiB of desktop applications, and the gap between
the allocator view and the driver view is consistently 0.6-1.0 GiB. Quoting only the
allocator figure understates what a deployment actually needs; quoting only the driver
figure overstates the model's own footprint.

### The cuBLAS workspace trap

Found by the smoke run, and worth recording because it silently breaks any sweep that
loads more than one model in a process.

After dropping every Python reference to a 2944 MiB model and calling
`gc.collect()` + `torch.cuda.empty_cache()`, the device still held the full 2946 MiB.
Instrumentation showed:

```
allocated_mib  8.1
reserved_mib   2946.0
live cuda tensors (gc-tracked): 0
```

Nothing was referenced from Python, but 8.1 MiB was still allocated. That is cuBLAS's
per-stream workspace, which is allocated *through PyTorch's caching allocator* and held
by C++ rather than by any Python object. The caching allocator can only return a segment
to the driver when the entire segment is free, so those few megabytes pinned gigabytes.

`src/monitoring/memory.py` therefore calls `torch._C._cuda_clearCublasWorkspaces()`
before `empty_cache()`. Reserved memory then drops to 0. Without it, the precision sweep
would OOM on its second precision while appearing to have released the first.

---

## 3. Prompt construction

Context length is the independent variable in Phase 1, so it is controlled exactly
rather than approximately. `src/benchmarks/prompts.py` tokenises a fixed corpus that is
tracked in the repository (`data/benchmark_corpus.txt`), tiles it if necessary, and
slices to a precise token count. `add_special_tokens=False` throughout, because a
special token would make the count wrong.

**Benchmark prompts do not use the chat template.** A template adds a model-dependent
number of tokens, so "2048 context" would silently mean a different length for each
checkpoint. The interpretability experiments *do* apply it, because there the
instruction format is the thing being studied.

Every row in a batch is identical. The measurement targets throughput at a given tensor
shape, not the variance of natural prompts.

---

## 4. Determinism

Full bitwise determinism is not achievable for every CUDA kernel a transformer touches,
so `src/utils/seed.py` seeds everything it can and then *records what took effect* -
`use_deterministic_algorithms`, the cuDNN flags, `CUBLAS_WORKSPACE_CONFIG`, and any
warnings - into every run's metadata.

Benchmark configs set `deterministic: false` deliberately. Requesting deterministic
algorithms disables cuDNN autotuning and can select slower kernels, which would make the
throughput numbers a measurement of the determinism flag rather than of the model.
Greedy decoding already makes the generated text reproducible, which is what the quality
comparisons need. The interpretability config sets `deterministic: true`, where
reproducibility matters more than throughput.

---

## 5. Quality metrics

An LLM judge would introduce a second, unvalidated model into the measurement. A
benchmark suite would take hours per configuration. Instead, three deterministic
comparisons against the bf16 reference:

**Greedy continuation agreement.** With sampling off, the reference and the candidate
should emit identical tokens. Reported as exact-match rate, mean token-level agreement,
and the mean index at which they first diverge - the last being the most interpretable:
"they agree for the first N tokens".

**Next-token distribution divergence.** Exact KL(reference ‖ candidate) and
Jensen-Shannon over a fixed prompt set, computed in float64 from full log-probability
vectors. KL is asymmetric and the reference is deliberately the first argument: it
measures the cost of substituting the candidate for the reference.

**Teacher-forced agreement and perplexity.** A strided sliding window over a fixed
corpus gives ~1088 scored positions instead of the 32 the prompt-set comparisons use.
The bundled corpus is 1089 tokens, so that is the ceiling; each run records its actual
`n_positions` rather than assuming. Because it is a *paired* comparison over identical
positions, relative differences between precisions are well determined even though the
absolute perplexity has correspondingly wide uncertainty. Storing full distributions
for all of them would run to gigabytes, so only two things are kept per position: the
argmax token and the log-probability of the actual next token. Those are sufficient for
top-1 agreement and for both perplexities, and neither is an approximation.

The reference model is measured once, reduced to those artifacts, and unloaded before
any candidate is loaded. The two are never resident at the same time.

---

## 6. Refusal direction

Reproducing the measurement in Arditi et al. (2024).

**Capture.** Forward hooks on each decoder block's output - the residual stream at that
depth. The last prompt-token position is used, which requires left padding; with right
padding the final position is a pad token and the captured vector is meaningless.
`load_tokenizer` sets `padding_side="left"` for exactly this reason.

**Fit.** Per layer, the difference between the harmful and harmless class means,
normalised to unit length.

**Evaluate, on held-out data.** The direction is fitted on a training split and every
reported statistic comes from a test split. Fitting and evaluating on the same prompts
would make "the class means differ along the difference of the class means" close to a
tautology, and it would produce a confident-looking result from pure noise. The unit
test `test_fit_layer_direction_reports_chance_on_noise` pins this down: two
identically-distributed classes must come back near AUROC 0.5.

**Report two scales.** Raw projections onto the unit direction have units of activation
norm, and residual-stream norm grows steeply with depth, so raw magnitudes are not
comparable across layers. Cosine projections divide each activation by its own norm and
are. Cohen's *d* and AUROC are standardised and comparable either way; AUROC is computed
exactly via the Mann-Whitney U statistic with tie correction.

**Check the behaviour first.** A direction separating two prompt sets only says
something about *refusal* if the model's refusal behaviour on those sets actually
differs. The pipeline measures the model's refusal rate on both classes before analysing
anything. The classifier is substring matching over the opening of the completion - the
same crude approach the original paper uses - so reported rates are a lower bound.

**Where this stops.** Measurement only. A direction that separates two prompt classes is
correlational evidence; establishing that the model *uses* it requires intervention -
ablating the direction or steering along it - which is deliberately not implemented
here. Completions to harmful prompts are classified and discarded, never stored.

---

## 7. What gets recorded

Every run writes:

```
results/<experiment>/<timestamp>/
├── meta.json        hardware, driver, CUDA, package versions, git commit + dirty flag,
│                    full config, determinism report, baseline VRAM
├── metrics.json     every cell including failures, capability reports, validation
├── results.csv      flat rows for plotting
├── telemetry/       per-cell GPU time series and event markers
└── run.log          the full DEBUG-level log
```

A cell's `status` is one of:

* `ok` - it ran and produced measurements
* `oom` - the card could not fit it; the allocator state at failure is attached
* `unsupported` - the configuration cannot exist on this machine, with the reason
* `error` - something else went wrong, with the traceback

Nothing is dropped silently and nothing is estimated. A figure is only drawn from cells
with `status: ok`.
