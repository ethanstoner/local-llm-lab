# Local LLM Lab

A reproducible framework for measuring what an open-weight language model actually does
on one consumer GPU - how fast it runs, what quantization costs, and what its internal
activations look like while it decides whether to refuse.

It is not a chat interface and not a wrapper around a serving runtime. The output of
this repository is data: structured result files, telemetry time series, and figures
generated from them. Every number in `results/` and `figures/` was produced by a
measured run on the hardware described below.

---

## 1. Research motivation

Three questions, each with a phase of the project behind it.

**Where does the time actually go?** A single "tokens per second" figure averages two
completely different regimes. Prefill processes the whole prompt in parallel and is
compute-bound; decoding emits one token at a time and is bound by memory bandwidth,
because every generated token requires a pass over every weight in the network. The
first question is what the split looks like across prompt lengths on a real card, when
the measurement is done carefully enough to be believed.

**What does quantization actually buy, and what does it cost?** Storing weights at four
or eight bits rather than sixteen cuts the volume of traffic per token, which in a
bandwidth-bound regime should mean higher throughput. In practice quantized weights must
be dequantized before they can be multiplied, and that costs time. A 4-bit configuration
that uses a third of the memory and runs *slower* than the bf16 baseline is a perfectly
ordinary outcome. The second question is what the trade actually looks like here, in
both memory and output fidelity.

**Is refusal linearly represented in the residual stream?** Arditi et al. (2024) report
that refusal behaviour in chat models is mediated by a single direction in activation
space. The third question is whether that signal is detectable, with what strength, and
at which depth, using only observation - no weight modification and no steering.

---

## 2. Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, compute capability 8.9 |
| Driver | 610.88 (CUDA UMD 13.3) |
| CPU / OS | Windows 11 Pro 26200 |
| Python | 3.12.10 |
| torch | 2.6.0+cu124 |
| transformers | 4.57.0 |
| bitsandbytes | 0.49.0 |

The desktop holds roughly 1.5-2.2 GiB of VRAM before anything is loaded, which is
recorded per run as `gpu.used_by_other_processes_mib`. Headroom figures in this
repository account for it.

**Subject model:** `Qwen/Qwen2.5-7B-Instruct` (7.62B parameters, 28 layers, hidden size
3584, GQA with 4 KV heads, 32768-token context). Chosen because it is ungated on the
Hub - Llama-3.1-8B-Instruct, Gemma-2-9b-it and Ministral-8B all require an interactive
licence acceptance, which makes them unusable for an unattended run - and because it has
genuine refusal behaviour, which Phase 5 needs. `Qwen/Qwen2.5-1.5B-Instruct` is used for
smoke tests.

---

## 3. Architecture

```
src/
├── utils/            config parsing, seeding, run metadata, result IO, prompt datasets
├── monitoring/       NVML sampler thread, PyTorch allocator accounting
├── models/           precision-aware loader, capability registry, OOM handling
├── benchmarks/       timing primitives, sweep driver, metrics records, CLI
├── evaluation/       quantization fidelity metrics, CLI
├── interpretability/ activation hooks, statistics, refusal-direction analysis, CLI
└── visualization/    figure functions, render CLI
configs/              one YAML per experiment
results/              one self-contained directory per run
figures/              rendered PNGs
tests/                119 tests, CPU-only and network-free
docs/methodology.md   the measurement decisions, in detail
```

Four ideas hold it together:

**A run is a directory.** Every experiment writes `meta.json` (hardware, driver, CUDA,
package versions, git commit and dirty flag, full config, determinism settings),
`metrics.json` (every cell, failures included), a flat `.csv` for plotting, a
`telemetry/` time series, and `run.log`. A result is never separated from the conditions
that produced it.

**Unsupported is a result.** `src/models/registry.py` probes each precision before it is
attempted. A configuration that cannot run on this machine is recorded with
`status: "unsupported"` and the reason. Nothing is silently dropped and nothing is
estimated.

**OOM is a result too.** `src/models/oom.py` catches allocation failures, attaches the
allocator state, and lets the sweep continue. Non-OOM exceptions still propagate: a real
bug must not be filed as an experimental outcome.

**Two views of memory, always both.** The PyTorch allocator knows what it handed out;
the driver knows what the device holds, including the CUDA context, cuBLAS workspaces
and other processes. Quoting only the first understates the requirement and only the
second overstates the model's own footprint, so every run reports both.

---

## 4. Methodology

The full version is in [`docs/methodology.md`](docs/methodology.md). The decisions that
most affect whether the numbers mean anything:

* **Every timed region ends with `torch.cuda.synchronize()`.** CUDA work is
  asynchronous; a timer stopped after the launch measures enqueue time, not execution.
* **Prefill and decode are measured separately.** Prompt processing is timed with its
  own dedicated forward pass. Decode throughput is `(new_tokens - 1) / (t_last - t_first)`
  and excludes prefill entirely. End-to-end throughput is reported alongside it.
* **Generation is forced to a fixed token budget** (`min_new_tokens == max_new_tokens`).
  A run that stops early at an end-of-sequence marker produces fewer tokens in less
  time, and the ratio flatters whichever configuration happened to stop soonest.
* **Time to first token is measured two ways** - from a per-token timestamp inside
  `generate`, and from the separate prefill pass - and both are reported, so the two can
  be checked against each other.
* **Per-token synchronization overhead is measured, not assumed.** Stamping a
  synchronized clock once per token is what makes an inter-token latency distribution
  real, but it gives up some CPU/GPU overlap. The harness runs the same configuration
  with and without it as a control and records the difference.
* **One warm-up iteration is discarded** per configuration, then N repeats, reported as
  median and standard deviation.
* **Prompt lengths are exact.** Benchmark prompts are built by tokenising a bundled
  corpus and slicing to a precise token count, with no chat template - a template adds a
  model-dependent number of tokens, which would make "2048 context" mean something
  different for each checkpoint.
* **Determinism is a recorded setting, not an assumption.** Benchmark configs turn
  deterministic algorithms off, because requesting them disables cuDNN autotuning and
  can select slower kernels - that would measure the determinism flag rather than the
  model. Greedy decoding already makes the generated text reproducible. The
  interpretability config turns them on. Either way the run records what took effect.

### Quality metrics (Phase 2)

No LLM judge: that would mean a second, unvalidated model in the loop. Instead, three
deterministic comparisons against the bf16 reference, with the two models never resident
on the card at once:

1. **Greedy continuation agreement** - exact-match rate, token-level agreement, and the
   mean index at which the two models first diverge.
2. **Next-token distribution divergence** - exact KL(reference ‖ candidate) and
   Jensen-Shannon over a fixed prompt set.
3. **Teacher-forced agreement and perplexity** over a fixed corpus with a strided
   window, giving ~1088 scored positions rather than the 32 the prompt comparisons use.
   Each run records the actual count.

### Refusal direction (Phase 5)

Per layer: capture the last-token residual stream for a harmful and a harmless prompt
set, take the difference of the class means, normalise it, and project held-out prompts
onto it.

* The direction is fitted on a **training split** and every reported statistic comes from
  a **held-out split**. Fitting and evaluating on the same prompts would make the result
  close to a tautology.
* Separation is reported as Cohen's *d* and AUROC, both standardised, so layers are
  comparable despite residual-stream norm growing with depth. Raw projections and
  norm-divided (cosine) projections are both recorded.
* A **behavioural check runs first**: the model's actual refusal rate on both sets. A
  direction separating two prompt sets only says something about refusal if the model's
  refusal behaviour on those sets actually differs.

Prompt sets come from `JailbreakBench/JBB-Behaviors`, which ships 100 harmful behaviours
and 100 matched benign ones. A small bundled fallback in `data/` keeps the pipeline
runnable offline. Harmful-prompt completions are classified as refusal or not and then
discarded; only the rate is retained.

---

## 5. Reproducing

```powershell
# 1. Environment (creates venv/, downloads torch, writes requirements.lock.txt)
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1

# 2. Tests - CPU only, no network, no model download
.\venv\Scripts\python.exe -m pytest tests/ -q

# 3. Smoke test on the 1.5B model (~2.9 GB download, under a minute)
.\venv\Scripts\python.exe -m src.benchmarks.run --config configs/smoke.yaml

# 4. Phase 1 - context-length sweep on the 7B model
.\venv\Scripts\python.exe -m src.benchmarks.run --config configs/qwen2.5-7b.yaml

# 5. Phase 2 - precision sweep, then quality comparison
.\venv\Scripts\python.exe -m src.benchmarks.run --config configs/precision_sweep.yaml
.\venv\Scripts\python.exe -m src.evaluation.run  --config configs/precision_sweep.yaml

# 6. Phases 4 and 5 - activation capture and refusal-direction analysis
.\venv\Scripts\python.exe -m src.interpretability.run --config configs/refusal.yaml

# 7. Figures, from whatever runs exist
.\venv\Scripts\python.exe -m src.visualization.render
```

`scripts/run_all.ps1` runs steps 3-7 in order.

Experiments are described entirely by their config file. To change the grid, edit the
YAML; unknown keys and impossible values are rejected at parse time rather than forty
minutes into a sweep.

---

## 6. Results

All figures below come from `results/`. Every number was measured; none is estimated,
and configurations that could not run are reported as such rather than omitted.

### 6.1 Where the time goes (Phase 1)

`Qwen/Qwen2.5-7B-Instruct`, bf16, batch size 1, 128 generated tokens, median of three
repeats after one discarded warm-up.

| prompt tokens | decode tok/s | end-to-end tok/s | TTFT | prefill tok/s | peak VRAM |
|---|---|---|---|---|---|
| 128 | 44.8 | 44.8 | 27 ms | 4999 | 16227 MiB |
| 512 | 43.6 | 43.1 | 57 ms | 8728 | 16311 MiB |
| 2048 | 39.6 | 37.5 | 209 ms | 9903 | 16691 MiB |
| 8192 | 27.4 | 22.8 | 974 ms | 8447 | 17915 MiB |
| 16384 | 19.6 | 14.7 | 2211 ms | 7427 | 20283 MiB |

Three things worth drawing out.

**Decode throughput falls 56% from 128 to 16384 tokens** (and end-to-end falls 67%) while the weights never change.
Nothing about the matrix multiplications got harder; the KV cache got bigger, and every
decode step must now read it. This is the clearest demonstration in the project that
single-stream decoding is bound by memory traffic rather than arithmetic.

**Prefill and decode move in opposite directions over most of the range.** Prefill
throughput *rises* from 4999 to 9903 tok/s between 128 and 2048 tokens, because short
prompts cannot fill the GPU, and then falls back to 7427 at 16384 as attention's
quadratic term begins to tell. Decode falls monotonically throughout. A single
"tokens per second" number would have averaged a rising curve with a falling one.

**Time to first token and the independent prefill pass agree.** At 2048 tokens: TTFT
209 ms against 2048/9903 = 207 ms. At 16384: 2211 ms against 16384/7427 = 2206 ms. Two
separate measurements of nearly the same quantity, taken by different mechanisms and
landing within 1%, which is the evidence that the timing harness does what it claims.

The per-token synchronization control measured **0.62% overhead** (2.854 s with,
2.837 s without). The inter-token latency distribution is therefore close to free.

### 6.2 What quantization costs (Phase 2)

Same model, 512-token prompt, 128 generated tokens.

| precision | decode tok/s | vs bf16 | weights | vs bf16 | peak VRAM |
|---|---|---|---|---|---|
| fp16 | 44.2 | +0.4% | 14526 MiB | - | 16349 MiB |
| bf16 | 44.0 | reference | 14526 MiB | reference | 16311 MiB |
| nf4 | 40.0 | **-9%** | 5191 MiB | **-64%** | 8509 MiB |
| int8 | 13.1 | **-70%** | 8303 MiB | -43% | 10531 MiB |
| fp32 | 2.2 | -95% | 29051 MiB | +100% | 24383 MiB |

**nf4 is the clear winner and int8 is a trap.** nf4 gives back 64% of the memory for a
9% throughput cost, and at a 2048-token prompt it matches bf16 outright (40.3 vs
40.2 tok/s) - by then the smaller weight reads are paying for the dequantization.
int8 saves *less* memory than nf4 and runs 3.4x slower: LLM.int8()'s mixed-precision
decomposition splits each matmul into two passes and cannot be folded into one kernel.
Someone reaching for "8-bit" as the moderate, safe option would get the worst row here.

**fp32 is the interesting failure.** 7.62B parameters at four bytes is 29051 MiB on a
24564 MiB card, so it should not fit - and it did not raise OOM. The Windows display
driver pages VRAM to host memory instead, so the model loaded and ran, at 2.2 tok/s with
a 4.7 s time to first token. At a 2048-token prompt it fell to 0.67 tok/s with a 36 s
TTFT, and the third repeat degraded further to 98 s as paging pressure compounded.

That is the single most useful operational finding here: **on this platform, exceeding
VRAM does not fail, it degrades by a factor of sixty.** A harness that trusted the
absence of an exception would have published those numbers as the model's performance.

### 6.3 Long-context attention on a build without flash

Diagnosing why the first Phase 1 sweep appeared to hang produced a result worth its own
section. Prefill only, bf16:

| prompt tokens | stock `sdpa` | `sdpa_no_gqa` |
|---|---|---|
| 2048 | 0.565 s / 1322 MiB | 0.609 s / 400 MiB |
| 8192 | 140.667 s / 17549 MiB | 0.968 s / 1572 MiB |
| 16384 | did not fit | 2.206 s / 3136 MiB |

Transformers passes `enable_gqa=True` to PyTorch's attention whenever there is no
attention mask, to keep it off the quadratic math kernel. PyTorch's Windows wheels are
not built with flash attention, and the memory-efficient kernel does not implement
`enable_gqa` - so both fast kernels are unavailable and the flag produces exactly the
fallback it was meant to prevent. Expanding the KV heads manually and omitting the flag
restores linear memory. Full diagnosis in `src/models/attention.py` and `PROGRESS.md`.

### 6.4 What quantization costs in fidelity

Against the bf16 reference: 1088 teacher-forced positions, 32 fixed prompts, 64-token
greedy continuations.

| precision | teacher-forced top-1 | perplexity ratio | mean KL (nats) | next-token top-1 | exact continuation |
|---|---|---|---|---|---|
| fp16 | 0.981 | 0.997 | 0.0006 | 1.000 | 0.50 |
| int8 | 0.958 | 1.003 | 0.0077 | 0.969 | 0.09 |
| nf4 | 0.889 | 1.047 | 0.0584 | 0.938 | 0.00 |

Reference perplexity 12.54.

**This is the column that changes the Phase 2 verdict.** On throughput and memory alone
nf4 looked like a straightforward win. It is also the least faithful of the three: 4.7%
higher perplexity, 89% teacher-forced top-1 agreement, and not one of 32 greedy
continuations matching the reference exactly. That is a real behavioural difference, not
a rounding artifact. nf4 remains the right default on a memory-constrained card, but
"9% slower for 64% less memory" is only half the sentence.

**The most useful methodological finding here is about the metric, not the models.**
fp16 and bf16 are numerically near-identical - mean KL of 0.0006 nats and *100%* next-token
top-1 agreement - yet only **half** their greedy continuations match exactly, with the
median divergence at token 49 of 64. Greedy decoding is chaotic: once two models pick
different tokens anywhere, the sequences separate permanently, so exact-match rate mostly
measures how long it takes numerical noise to flip one argmax. It is reported here
because it is what a user would notice, but distribution divergence is the metric that
actually tracks quality. A quantization comparison resting on continuation match alone
would conclude that fp16 "fails" half the time.

### 6.5 Is refusal linearly represented? (Phase 5)

`JailbreakBench/JBB-Behaviors`, 100 harmful and 100 matched benign instructions, split
70/30. The direction is fitted on the training split; everything below is the held-out
split.

**The behavioural precondition holds.** The model refused **36/40 (90%)** of harmful
prompts and **5/40 (12.5%)** of benign ones. Without that gap, a separating direction
would be measuring topic rather than refusal.

| layer | held-out Cohen's *d* | held-out AUROC | fitting-split *d* |
|---|---|---|---|
| 0 | 0.13 | 0.541 | 0.84 |
| 5 | 0.06 | 0.524 | 0.76 |
| 10 | 0.35 | 0.596 | 1.17 |
| 15 | 1.78 | 0.894 | 2.07 |
| 18 | 3.00 | 0.976 | 3.01 |
| **20** | **3.59** | **0.982** | 3.12 |
| 24 | 3.41 | 0.979 | 3.06 |
| 27 | 3.41 | 0.978 | 3.01 |

**A direction is clearly present, and it emerges at a specific depth.** Separation is at
chance through layers 0-10, rises sharply between layers 12 and 19, and plateaus from
layer 18 onward, peaking at layer 20 of 28 - about 71% of the way through the network.
That is consistent with Arditi et al., who report the refusal direction in the middle-to-late
layers rather than at the output.

**The train/test split earned its place.** At layer 0 the fitting split shows *d* = 0.84
while the held-out split shows *d* = 0.13. In a 3584-dimensional space, 70 examples per
class are enough to find a direction that separates the training data by half a standard
deviation of pure noise. Fitting and evaluating on the same prompts would have reported a
refusal signal in the embedding layer.

**"A single direction" needs qualifying.** Cosine similarity between each layer's fitted
direction and layer 20's falls off either side of the peak: 0.60 at layer 18, 0.78 at 19,
0.85 at 21, 0.75 at 22, and 0.39 by layer 27. Near-zero below layer 10. So the directions
found in the high-signal band are *related* but not identical, and this observational
setup cannot distinguish "one feature read imperfectly at several depths" from "several
correlated features". Settling that needs intervention, which is out of scope here.

Residual-stream norm grows from ~10 at layer 0 to ~400 at layer 27 and is
indistinguishable between the two prompt classes, which is why layers are compared with
standardised statistics rather than raw projection magnitudes.

---

## 7. Limitations

* **One GPU, one model, one machine.** Nothing here establishes that these numbers
  generalise to other cards, other architectures or other driver versions.
* **Batch size 1.** Every measurement is single-stream. Server-style throughput with
  continuous batching is a different regime and is not measured.
* **The desktop is not idle.** A browser and other applications hold VRAM and
  occasionally take GPU time. The baseline occupancy is recorded per run and repeats are
  reported with spread, but this is not a quiet benchmarking rig.
* **`transformers.generate`, not an optimised serving stack.** The figures reflect what
  the reference implementation does. vLLM or TensorRT-LLM would produce different, and
  generally better, numbers.
* **The refusal classifier is substring matching.** It is the same crude approach the
  original paper uses for its refusal score. It will miss an unusually phrased refusal,
  so reported refusal rates are a lower bound.
* **The OOM path is untested on this hardware.** `src/models/oom.py` is covered by unit
  tests, but no experiment here ever triggered it: fp32 needed 29051 MiB on a 24564 MiB
  card and the Windows driver paged rather than failing. On a platform that raises
  `CUDA out of memory` properly, that code path is unexercised by these runs.
* **The perplexity corpus is 1089 tokens.** That is ~1088 scored positions - enough for a
  paired comparison between precisions, not enough to characterise the model's absolute
  perplexity on natural text.
* **Exact continuation match is a weak quality signal.** Section 6.4 shows fp16 and bf16
  disagreeing on half of greedy continuations despite being numerically near-identical.
  The metric is reported because it reflects what a user sees, but conclusions should
  rest on the distribution-level numbers.
* **Phase 5 used 100 prompts per class, 30 held out.** Enough to establish that a
  direction exists at an effect size of *d* > 3, not enough to characterise its
  variation across prompt types or categories.
* **Phase 5 is correlational.** A direction that separates two prompt classes is not
  evidence that the model *uses* that direction. Establishing that requires intervention
  - ablating the direction or steering along it - which this project deliberately does
  not do. Section 6.5 also shows the per-layer directions are related but not identical,
  which observation alone cannot resolve.
* **Quantized parameter counts are stored elements, not logical parameters.** 4-bit
  weights are packed into `uint8`, so `param_count` undercounts; the `dtype_histogram`
  in each run's metadata shows the real storage picture.

---

## 8. Future work

* Intervention experiments for Phase 5, done deliberately and with the safety
  implications thought through rather than as an afterthought.
* Batched throughput and a KV-cache memory model validated against measurement.
* A second architecture, to separate model-specific effects from general ones.
* Attention-pattern and per-head analysis, reusing the existing hook infrastructure.
* AWQ and GPTQ alongside the bitsandbytes modes.

---

## References

1. Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024). *Refusal in Language
   Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
2. Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag, Dobriban, Flammarion,
   Pappas, Tramèr, Hassani, Wong (2024). *JailbreakBench: An Open Robustness Benchmark
   for Jailbreaking Large Language Models.* [arXiv:2404.01318](https://arxiv.org/abs/2404.01318)
3. Dettmers, Lewis, Belkada, Zettlemoyer (2022). *LLM.int8(): 8-bit Matrix Multiplication
   for Transformers at Scale.* [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
4. Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023). *QLoRA: Efficient Finetuning of
   Quantized LLMs.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) - source of the
   NF4 data type.
5. Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.*
   [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html) -
   the residual-stream view the activation capture relies on.
6. Qwen Team (2024). *Qwen2.5 Technical Report.* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

## Scope and intent

This repository measures models; it does not modify them. Phase 5 reproduces the
*analysis* in Arditi et al. - does a refusal direction exist, how strong is it, and
where - and stops there. No weight editing, directional ablation or activation steering
is implemented. Harmful-prompt completions are classified and discarded, never stored or
reported.
