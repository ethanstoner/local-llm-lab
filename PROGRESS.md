# Progress log

Working notes for the build: what was done, what was run, what broke, and what is next.
Written as the work happened, so the failures are in here alongside the successes.

---

## Session 1 - 2026-09-22

### Environment inspection (before anything was installed)

| Item | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, compute capability 8.9 |
| Driver / CUDA UMD | 610.88 / 13.3 |
| VRAM held by other processes | ~1.5-2.2 GiB (browser, Roblox Studio, overlays) |
| Python | 3.12.10 (3.10 also present) |
| git | 2.49.0.windows.1 |
| Disk | C: 480.5 GB free, D: 138.8 GB free |
| Hugging Face Hub | reachable, token file present |

Commands used:

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
py -0p
git --version
Get-PSDrive -PSProvider FileSystem
Invoke-WebRequest https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct
```

No system-wide configuration was changed at any point. Everything lives in the
project-local `venv/`.

### Problem 1: the machine's global `transformers` install is broken

Probing the global environment for the `from_pretrained` dtype keyword failed outright:

```
ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this
module, but found huggingface-hub==1.3.3.
```

`transformers` 4.57.0 and `huggingface_hub` 1.3.3 are mutually incompatible, and the
global site-packages has exactly that pair. This is a pre-existing problem on the
machine, not something this project caused - but the first draft of `setup_env.ps1`
pinned the same broken pair, so it would have reproduced the fault inside the venv.

**Fix:** pin `huggingface_hub>=0.34.0,<1.0` and let the resolver pick a compatible
version. The venv resolved to a working combination and imports cleanly.

### Problem 2: pip's download from the PyTorch CDN stalled silently

The first dependency install sat on

```
Downloading .../torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl (2532.3 MB)
```

for 25 minutes. No bytes were written to the pip cache in that window and the python
process accumulated ~3 seconds of CPU total. No timeout fired and no error was raised.

Diagnosis: the CDN was fine. A ranged request pulled 50 MB in 4.0 s (12.4 MB/s), and a
HEAD on the wheel returned `HTTP 200, Content-Length 2532302369`. The stall was in pip's
own connection.

**Fix:** `scripts/setup_env.ps1` now downloads the torch wheel itself with a resumable
`HttpWebRequest` loop that reports progress and retries, then hands the local file to
pip. The download then ran at a steady ~13.6 MB/s and completed in about three minutes.

### Problem 3: `2>$null` on a native command is fatal under `$ErrorActionPreference = "Stop"`

The rewritten setup script died immediately with:

```
python.exe : Traceback (most recent call last):
  + CategoryInfo : NotSpecified: (...) [], RemoteException
  + FullyQualifiedErrorId : NativeCommandError
```

Windows PowerShell 5.1 wraps each stderr line of a native command in an `ErrorRecord`
when stderr is redirected, and `$ErrorActionPreference = "Stop"` promotes that to a
terminating error - even though the command itself succeeded.

**Fix:** the "is torch already installed" probe now reads `__version__` out of
`venv\Lib\site-packages\torch\version.py` with `Select-String`. No native command, no
stderr, no redirect.

### Environment as installed

```
torch 2.6.0+cu124   cuda 12.4   torch.cuda.is_available() True
transformers 4.57.0
bitsandbytes 0.49.0
```

Full pin list in `requirements.lock.txt` (62 packages).

### Pre-download report

Answering the check required before fetching any weights:

| Model | Role | Download | VRAM bf16 | VRAM int8 | VRAM nf4 |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | primary subject | 14.2 GB | ~15.5 GB | ~8.5 GB | ~5.5 GB |
| `Qwen/Qwen2.5-1.5B-Instruct` | smoke tests | 2.89 GB | ~3.1 GB | - | - |

Sizes measured from the Hub API (`/api/models/<id>/tree/main?recursive=true`), not
estimated. Total ~17.1 GB against 480.5 GB free on C:. Peak VRAM ~15.5 GB against
~22.3 GB usable. Both sufficient, so the download proceeded.

Llama-3.1-8B-Instruct, Gemma-2-9b-it and Ministral-8B were all rejected as *gated* on
the Hub: an unattended run would have stalled on a licence click. Qwen3-8B is ungated
but its hybrid thinking mode makes throughput figures ambiguous.

### Built so far

| Area | Modules |
|---|---|
| Config & reproducibility | `src/utils/config.py`, `seed.py`, `env.py`, `io.py`, `datasets.py` |
| GPU telemetry | `src/monitoring/gpu.py` (NVML sampler thread), `memory.py` |
| Model construction | `src/models/loader.py`, `registry.py`, `oom.py` |
| Benchmarks | `src/benchmarks/metrics.py`, `latency.py`, `prompts.py`, `sweep.py`, `run.py` |
| Quality evaluation | `src/evaluation/quality.py`, `run.py` |
| Interpretability | `src/interpretability/hooks.py`, `stats.py`, `refusal.py`, `run.py` |
| Figures | `src/visualization/plots.py`, `render.py` |
| Tests | `tests/` - 109 tests, CPU-only and network-free |

### Test status

```
> .\venv\Scripts\python.exe -m pytest tests/ -q
107 passed in 11.75s
```

Two tests failed on the first run; both were faults in the tests rather than in the code:

* `test_lists_become_tuples` built its YAML by concatenating an unindented block with an
  indented one, so `textwrap.dedent` found a common prefix of `""` and left the second
  block indented. Fixed by building the fragment without `dedent`.
* `test_write_csv_handles_non_finite` expected a bare empty line. Python's `csv` module
  writes a lone empty field as `""` so the row is not an ambiguous blank line - the
  writer was right and the assertion was wrong. Now read back through `csv.reader`.

### Problem 4: Hugging Face downloads stall the same way pip did

`snapshot_download` of the 1.5B model hung at 608 MB of 3.1 GB. Installing `hf_xet`
and retrying hung at the same place. Progress went 11 MB, then 159 MB, then 608 MB,
then nothing for over a minute with no error.

Same diagnosis as Problem 2, and now with a second data point: it is Python's HTTP
stack on this machine, not any particular CDN. A ranged .NET `HttpWebRequest` against
`huggingface.co` sustained 7.86 MB/s on the same file that `huggingface_hub` could not
finish.

**Fix:** `scripts/fetch_model.ps1` downloads a repository's files with the resumable
ranged downloader, and `ModelConfig` gained a `local_path` field so `from_pretrained`
reads from disk. `model.id` is still what gets recorded in run metadata, so provenance
is unaffected, and `_fetch_metadata.json` in each model directory records the resolved
commit. Both models then downloaded cleanly:

```
Qwen2.5-1.5B-Instruct  7 files, 2.89 GB, commit 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
Qwen2.5-7B-Instruct   10 files, 14.19 GB
```

`scripts/fetch_datasets.ps1` does the same for the two JailbreakBench CSVs (44 KB), so
the refusal analysis reads its prompts from disk and cannot be perturbed by a download
stalling partway through a run.

### Problem 5: 8 MiB of cuBLAS workspace pinned 2946 MiB of VRAM

Found by the smoke run. After the sweep unloaded a 2944 MiB model, the log said the
device still held 4560 MiB against a 1528 MiB baseline - nothing had been freed.

Released in isolation the same model came back cleanly, so the first hypothesis was a
lingering Python reference. Instrumentation said otherwise:

```
allocated_mib  8.1
reserved_mib   2946.0
live cuda tensors (gc-tracked): 0
```

Nothing referenced from Python, but 8.1 MiB still allocated. That is cuBLAS's per-stream
workspace. It is allocated *through PyTorch's caching allocator* and held by C++, and the
allocator can only return a segment to the driver when the whole segment is free - so a
few megabytes pinned gigabytes.

**Fix:** `src/monitoring/memory.py` calls `torch._C._cuda_clearCublasWorkspaces()` before
`empty_cache()`. Reserved memory then drops to 0 and the device returns to baseline plus
the ~84 MiB CUDA context, which cannot be freed while the process lives.

This would have OOM'd the precision sweep on its second precision while appearing to
have released the first. Two regression tests now cover it.

### Pipeline validation before the real runs

Rather than discover problems during a 40-minute sweep, each pipeline was exercised on
the 1.5B model first.

**All five precisions load and run** (1.5B, 256-token context, 1 repeat):

| precision | decode tok/s | TTFT ms | weights MiB | peak MiB |
|---|---|---|---|---|
| fp32 | 51.5 | 34.8 | 5889 | 9247 |
| bf16 | 50.8 | 25.5 | 2944 | 6169 |
| fp16 | 47.5 | 25.1 | 2944 | 6203 |
| nf4 | 34.4 | 39.8 | 1070 | 4693 |
| int8 | 11.7 | 109.8 | 1695 | 4981 |

bitsandbytes works on Windows here, and the device returned to ~1.6 GiB between every
precision. The int8 result is not a bug: LLM.int8()'s mixed-precision decomposition is
known to be slow, and at this model size the card is not yet bandwidth-bound, so fp32
being the fastest is consistent rather than surprising. These are smoke numbers on the
wrong model and are not reported as results.

**The interpretability pipeline runs end to end.** On the 1.5B model with the bundled
prompts: refusal rate 15/16 on the harmful set against 1/16 on the harmless set, 28
layers analysed, separation rising sharply between layers 8 and 19. The behavioural
precondition holds, so the direction is measuring something real.

**The figure pipeline produces 10 figures** and each was opened and inspected. One layout
bug was found and fixed: the peak-layer annotation overflowed the axes when the peak sat
near the right edge.

### Problem 6: prefill would have OOM'd at 16k context

Caught by reading rather than by running. `measure_prefill` called the model directly,
and a plain forward pass computes logits for *every* prompt position. At 16384 tokens
with Qwen's 152k vocabulary that is a ~5 GiB tensor on top of 15 GiB of weights.

**Fix:** pass `logits_to_keep=1`. `generate` already does this internally, so the change
also makes the dedicated prefill timing comparable to the TTFT measured inside
`generate` rather than systematically slower than it.

### Problem 7: long-context prefill was 145x slower than it should be

The first Phase 1 sweep on the 7B model produced clean numbers at 128, 512 and 2048
tokens and then appeared to hang at 8192: ten minutes, GPU pinned at 100%, 24079 MiB of
24564 MiB in use, and 12 GB of the process resident in host RAM.

It was not hung. It was paging. Instrumenting prefill alone:

```
ctx=  2048  prefill=  0.902s  peak_alloc=15851 MiB
ctx=  4096  prefill=  1.269s  peak_alloc=19212 MiB
ctx=  8192  prefill=179.043s  peak_alloc=32078 MiB
```

A peak allocation of 32078 MiB on a 24564 MiB card is only possible because the Windows
display driver pages VRAM to host memory instead of failing. **No OOM is raised.** The
run completes and is merely two hundred times slower, which is the dangerous failure
mode: a benchmark that does not check would publish 179 s as the model's prefill
latency.

Activation memory was growing quadratically (1322 -> 4682 -> 17549 MiB for 2048 -> 4096
-> 8192), which means the full attention matrix was being materialised. Four experiments
to find out why:

1. **Is it the attention mask?** No. Identical memory with and without it.
2. **Which SDPA kernels exist?** `UserWarning: Torch was not compiled with flash
   attention`. The Windows wheel has no flash kernel. Memory-efficient works and is
   linear: 56 MiB at 8192 against math's 16832 MiB.
3. **Force the memory-efficient kernel on the model.** `RuntimeError: No available
   kernel` - so transformers is passing something it cannot accept.
4. **Which argument?** `enable_gqa`:

   ```
   enable_gqa=True   mem_efficient: FAIL  No available kernel
   enable_gqa=True   math         : OK    4344 MiB
   enable_gqa=False  mem_efficient: OK      28 MiB
   ```

Transformers' `use_gqa_in_sdpa` passes `enable_gqa=True` whenever there is no attention
mask, and its comment gives the reason: a mask "will fall back to the math kernel". That
holds when a flash kernel exists. Here flash does not exist and memory-efficient does not
implement `enable_gqa`, so the flag causes precisely the fallback it was written to
avoid.

**Fix:** `src/models/attention.py` registers `sdpa_no_gqa` - the stock SDPA path with KV
heads expanded via `repeat_kv` and `enable_gqa` never passed. Measured on
Qwen2.5-7B-Instruct at bf16, prefill only:

| ctx | stock `sdpa` | `sdpa_no_gqa` |
|---|---|---|
| 2048 | 0.565 s / 1322 MiB | 0.609 s / 400 MiB |
| 8192 | 140.667 s / 17549 MiB | 0.968 s / 1572 MiB |
| 16384 | did not fit | 2.206 s / 3136 MiB |

Activation memory is linear again. Configs now use `attn_implementation: auto`, which
probes for a flash kernel and prefers stock `sdpa` where one exists - this is a
workaround for a platform limitation, not an improvement on transformers, and it should
not apply itself where it is not wanted. The probe results are recorded in every run's
metadata.

Ten tests were added, including an equivalence check against an explicit
`softmax(QK^T/sqrt(d))V` reference, so the workaround is verified to compute the same
thing rather than merely to run faster.

### Problem 8: the renderer picked the wrong runs

The first full render selected the *smoke* run - Qwen2.5-1.5B - to supply the headline
throughput figures for a 7B experiment.

`discover_runs` classified runs by which result file they held and kept whichever it
visited last. Directories are walked in sorted order, so "last" meant the
alphabetically-last experiment name. `smoke` sorts after `phase5_refusal_direction`.

**Fix:** rank candidate runs and take the maximum, by how richly a run sweeps the axis
the figure is about, then by timestamp. Context-sweep and precision-sweep runs are now
tracked as separate kinds, so a two-point precision sweep cannot displace the five-point
context sweep's curves. Figures now come from `phase1_context_sweep` and
`phase2_precision_sweep` as intended.

This one is worth dwelling on. The figures looked entirely plausible: correct axes,
sensible curves, a provenance caption. The caption was even correct - it said
`Qwen/Qwen2.5-1.5B-Instruct`, which is exactly the check that caught it.

### All phases complete

| Phase | Run | Result |
|---|---|---|
| 1 | `phase1_context_sweep` | 5/5 cells |
| 2 | `phase2_precision_sweep` | 10/10 cells |
| 2 | `phase2_precision_sweep_quality` | 3 precisions compared to bf16 |
| 4-5 | `phase5_refusal_direction` | 28 layers, 90% vs 12.5% refusal rates |
| 3 | `figures/` | 14 figures, each opened and inspected |

Headline numbers are in README section 6. The findings that surprised me:

1. **nf4 is fast and small but the least faithful.** 64% memory saving, 9% throughput
   cost - and 4.7% higher perplexity with zero exact continuation matches. The
   throughput table alone would have oversold it.
2. **fp16 and bf16 disagree on half their greedy continuations** despite mean KL of
   0.0006 nats and 100% next-token top-1 agreement. Greedy decoding amplifies numerical
   noise until an argmax flips, after which the sequences never re-converge. Exact-match
   rate is therefore a poor quality metric, and this project reports it next to the
   distribution metrics rather than instead of them.
3. **The refusal direction peaks at layer 20 of 28**, with held-out *d* = 3.59 and
   AUROC 0.982, rising from chance at layer 10. Consistent with Arditi et al.
4. **The train/test split was not a formality.** At layer 0 the fitting split shows
   *d* = 0.84 against the held-out split's 0.13. With 70 examples per class in 3584
   dimensions, a direction that separates the training data is available in pure noise.
   Evaluating in-sample would have reported refusal structure in the embedding layer.

### Figure inspection

All 14 were opened and looked at, not just generated. Two layout bugs were found and
fixed that way: the peak-layer annotation overflowed the axes when the peak sat near the
right edge, and `bf16`/`fp16` labels printed on top of each other in the
memory-vs-throughput scatter, since the two points nearly coincide.

Two figures look sparse and are correct that way. `latency_vs_context` shows two lines
almost exactly superimposed - that *is* the cross-check succeeding, since TTFT and the
independently timed prefill pass agree to within 1%. `memory_over_time` is nearly flat
within each run, because 128 generated tokens add little to a KV cache already holding
the prompt.

---

## Summary

The repository is runnable and every phase has produced measured results.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
powershell -ExecutionPolicy Bypass -File .\scriptsetch_model.ps1 -Repo Qwen/Qwen2.5-7B-Instruct
powershell -ExecutionPolicy Bypass -File .\scriptsetch_datasets.ps1
.env\Scripts\python.exe -m pytest tests/ -q          # 119 passed
powershell -ExecutionPolicy Bypass -File .\scripts
un_all.ps1
```

Built: a config-driven experiment harness with NVML telemetry, a precision-aware loader
covering fp32/fp16/bf16/int8/nf4, a timing harness that separates prefill from decode and
validates its own overhead, quantization fidelity metrics against a full-precision
reference, residual-stream activation capture, a layer-wise refusal-direction analysis,
and a figure renderer. 119 tests, CPU-only and network-free.

Eight problems were found and fixed along the way; all are written up above. Four were
platform traps that would have produced plausible-looking wrong numbers rather than
errors:

* cuBLAS workspaces pinning gigabytes of allocator segments after a model was unloaded.
* `enable_gqa` forcing SDPA onto the quadratic math kernel on a build without flash
  attention, making 8k prefill 145x slower.
* The Windows driver paging VRAM to host memory instead of raising OOM, so an fp32 model
  that does not fit "ran" at 1/60th speed.
* The figure renderer selecting a 1.5B smoke run to illustrate a 7B experiment.

That list is the argument for the parts of the design that look like overhead: recording
both allocator and driver memory, probing kernels rather than trusting flags, stamping
provenance onto every figure, and looking at the output.

## Recommended next steps

1. **Intervention for Phase 5.** Everything needed to ablate the layer-20 direction or
   steer along it is in place. That converts a correlational result into a causal one,
   and it is the obvious next experiment - to be done deliberately, with the safety
   implications considered rather than as an afterthought.
2. **A second architecture.** Every finding here is one model on one card. Llama-3.1-8B
   or Gemma-2-9b would separate model-specific effects from general ones. Both are gated,
   so they need an interactive licence acceptance first.
3. **Batched throughput.** All measurements are batch size 1. Server-style throughput
   with continuous batching is a different regime entirely.
4. **A KV-cache memory model** predicted from the config and validated against the
   measured curve in `vram_vs_context.png`.
5. **AWQ and GPTQ** alongside the bitsandbytes modes, now that the harness makes adding
   a precision a registry entry.
6. **Re-run on Linux.** It would take the flash-attention path, which would quantify what
   the `sdpa_no_gqa` workaround costs against a build that does not need it.


---

## Session 2 - 2026-09-22 (afternoon)

Goal: turn a complete but single-model, correlational, batch-1 study into a stronger
piece - a causal test of the Phase 5 result, an analytical account of the benchmark
numbers, and an optimisation that the analysis motivates. What happened instead of the
plan is recorded as carefully as what went to plan.

### Built

| Area | Files |
|---|---|
| Phase 6 causal test | `src/interpretability/intervention.py`, `causal.py`, `intervene.py`, `configs/intervention.yaml` |
| Hardware ceilings | `src/benchmarks/ceilings.py` |
| Paired A/B harness | `src/benchmarks/interleaved.py`, `configs/decode_ab.yaml`, `configs/batch_sweep.yaml` |
| Decode attention path | `sdpa_grouped_decode` and `fp32_reference` in `src/models/attention.py` |
| Roofline analysis | `src/analysis/roofline.py` |
| Cross-scale replication | `configs/refusal_1.5b.yaml`, `configs/intervention_1.5b.yaml` |
| Figures | 7 new figure functions; render pinned to a primary model |
| Tooling | ruff config and fixes; `.github/workflows/tests.yml`; `run_all.ps1` covers every phase |

Tests: 119 -> 169, all CPU-only.

### Problem 9: `src/models/` was never committed

The `.gitignore` rule `models/` - meant for the weights directory - matched `src/models/`
too. The loader, precision registry, OOM guard and attention backends existed only on
this disk. Every commit failed at import from a fresh clone, verified by cloning the
parent of the fix. **Fix:** anchor the rule as `/models/`; add the package exactly as it
stood when the committed results were produced (the attention module was reconstructed
by reversing this session's edits, and the original test suite passed against it from a
fresh clone), then apply this session's changes as separate commits.

### Problem 10: custom attention backends received no padding mask

Found while wiring Phase 6. Transformers keeps attention *functions* and attention
*mask builders* in separate registries; for a name missing from the mask registry,
`masking_utils` passes `None`. `sdpa_no_gqa` was only registered as a function, so every
left-padded batch attended to its pad tokens. On a tiny Llama, a padded prompt's
last-token logits moved by up to 0.249 against the same prompt alone; `eager` and `sdpa`
gave exactly 0.

Impact: batch-1 benchmarks unaffected. Phase 2 prompt-level metrics, Phase 5 capture and
the behaviour check were contaminated and were re-run:

| quantity | before | after |
|---|---|---|
| Phase 5 harmful / benign refusal | 90% / 12.5% | 93% / 5% |
| Phase 5 best layer, held-out d, AUROC | 20, 3.59, 0.982 | 20, 3.83, 0.990 |
| Phase 5 layer 0 held-out d | 0.13 | 0.62 |
| Phase 2 nf4 mean next-token KL | 0.058 | 0.120 |
| Phase 2 fp16 exact continuations | 0.50 | 0.59 |

The old unit test called the function with an explicit mask - a path the model never
takes. The new test runs a padded batch through the real model, and was confirmed to
fail with the registration removed. The in-flight Phase 6 run was killed and deleted.

### Problem 11: a sequential A/B confounded by the desktop

Two sweeps, one per attention backend, were started back to back. The baseline arm
measured 35.5 tok/s at 128 tokens against Phase 1's 44.8 for the identical config - the
desktop had several other applications active, and
batch-1 decode is partly CPU-launch-bound. **Fix:** `src/benchmarks/interleaved.py`,
which loads once, switches backend in place, alternates order each round and reports
paired ratios. The sequential runs and their configs were deleted.

### Problem 12: PyTorch's profiler on Windows, and a profile that paged

`torch.profiler` with CUDA activity raised "Legacy CUDA profiling requires use_cpu=True":
this build has no Kineto, and the fallback attributes device time to CPU ops including
views, so a first profile summed 80 ms of "GPU time" into a 23.6 ms step. The same run's
16k prefill kept full logits (16384 x 152064 bf16 = 5 GB), pushed the card into paging,
and reported 92 ms per step against 54 ms in the benchmark. **Fix:** `logits_to_keep=1`
and CUDA events recorded from module pre/post hooks. That breakdown (hook overhead
included) put attention at 34.2 ms of a 56.5 ms step at 16k on `sdpa_no_gqa`, against
9.8 ms of 30.3 ms with the grouped path.

### Problem 13: bf16 attention scores

The first `sdpa_grouped_decode` rounded scores to bf16, as eager attention does. It was
64% faster at 16k and generated plausible text. The teacher-forced fidelity check
against a float32-attention reference showed mean KL 0.09-0.14 nats with steps above
8 nats, against ~1e-4 for SDPA. **Fix:** float32 scores (~6% of kernel time at 16k), plus
the hybrid dispatch - folded SDPA below ~1.4k cached tokens, grouped fp32 matmul above -
from a per-layer microbenchmark:

| cached tokens | repeat_kv+SDPA | folded SDPA | grouped fp32 |
|---|---|---|---|
| 128 | 43 us | 26 us | 77 us |
| 1024 | 69 us | 44 us | 72 us |
| 2048 | 114 us | 92 us | 91 us |
| 16384 | 995 us | 702 us | 153 us |

### Problem 14: regression tests that passed on the bug

The first bf16-score test passed with the bug re-injected: its inputs were float32. The
second also passed: bf16 had rounded both test keys to 500.0, so 50/50 was correct. The
third uses bf16-exact inputs whose scores (500.0, 500.5) bf16 cannot both represent, and
fails on the old code with 0.500 against 0.378.

### Problem 15: a figure drawing invalid data

The first batch-throughput figure plotted batch 64 as ordinary points although the
harness had flagged it as paging (24054 of 24564 MiB). Flagged cells are now drawn
hollow and labelled, and excluded from the roofline analysis.

### Measured results (session 2)

**Ceilings:** 947 GB/s streaming read; 884-891 GB/s MLP GEMV; 157.5 TFLOP/s bf16 GEMM.

**Phase 7, decode A/B, 5 paired rounds** (grouped vs sdpa_no_gqa): 1.01x at 128, 1.03x at
512 and 1024, 1.02x at 2048, 1.17x at 4k, 1.37x at 8k, 1.57x at 16k (19.2 -> 30.2 tok/s).
KL to the fp32 reference ~1e-4 for both at every length. `auto` now selects it.

**Phase 8, batch sweep, 3 paired rounds:** 1.02x at batch 1, 1.13x at 8, 1.33x at 16,
1.65x at 32 (672 -> 1120 tok/s aggregate). Batch 64 flagged as paging and excluded.

**Roofline:** the old path runs at 0.68-0.80 of its modelled ceiling at every context
and batch size; nf4 at 0.27-0.34; int8 at 0.14-0.16. Prefill MFU 42% at 128 tokens,
85% at 2048, 77% at 16k.

**Phase 6, 7B:** refusal 95% intact; 0-2.5% with the direction from any of layers 14-19
ablated; 95% under all three random directions. Layer 16 ablation: 2.5% refusal, 0.995x
perplexity. Layers 21-24: 30-65% refusal, 1.26-1.36x perplexity. Addition at layer 20:
3% -> 31% (1x) -> 95% (1.5x) -> 100% (2x) on harmless prompts; random vectors 0-2.5%.
The selection rule, fixed before the run, chose layer 27, whose addition is degenerate; reported
as it ran (README section 6).

**Phase 5 and 6, 1.5B:** refusal 100% harmful, 40% JBB benign, 2% bundled benign.
Held-out d peaks at 2.0 (layer 18). Addition at layer 18: 16% -> 76% (1x) -> 100% (2x);
random vectors 12-14%. Ablation: random directions 98-100%; layer 16 19% at +1.6%
perplexity (the rule's choice); layers 14/15/19 reach 0-1% at +27-43% perplexity, and
layer 14's completions score judge NLL 2.5-2.7 - degraded output. Sufficiency replicates; clean necessity does not.

### Commands run

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m src.benchmarks.ceilings
.\venv\Scripts\python.exe -m src.benchmarks.interleaved --config configs/decode_ab.yaml
.\venv\Scripts\python.exe -m src.benchmarks.interleaved --config configs/batch_sweep.yaml
.\venv\Scripts\python.exe -m src.interpretability.run --config configs/refusal.yaml
.\venv\Scripts\python.exe -m src.interpretability.intervene --config configs/intervention.yaml
.\venv\Scripts\python.exe -m src.evaluation.run --config configs/precision_sweep.yaml
.\venv\Scripts\python.exe -m src.interpretability.run --config configs/refusal_1.5b.yaml
.\venv\Scripts\python.exe -m src.interpretability.intervene --config configs/intervention_1.5b.yaml
.\venv\Scripts\python.exe -m src.analysis.roofline
.\venv\Scripts\python.exe -m src.visualization.render
```

CI replay (WSL, fresh clone, CPU torch 2.6.0): ruff clean, 167 passed, 2 GPU tests
deselected. First GitHub Actions run (private repo, 2026-09-22): same result, 1m12s.

### Next steps

1. **A full Arditi-style selection rule** - require the candidate to induce refusal and
   exclude the last fifth of the network - evaluated on a fresh held-out prompt set, since
   the bundled set has now been seen for every layer.
2. **CUDA graphs or a static cache** for batch-1 decode: the roofline puts every
   configuration at ~75% of its bandwidth ceiling, and the remainder is launch overhead.
3. **Remove the DynamicCache `torch.cat`** with a preallocated cache - the grouped
   traffic model charges it two extra passes over the cache per step.
4. **A second architecture family**, if a licence can be accepted interactively.
5. **Decide on public release** - the repo is private on GitHub with CI passing.

---

## Session 3 - 2026-09-22 (evening)

### An external review, and the layer-selection rule fixed

A review of the published repository asked for a shorter landing page, narrower wording
on two claims, an accurate test badge and a license; all done (README is now a landing
page, the full write-up is `docs/results.md`, MIT license added).

The Phase 6 layer-selection rule had applied only one of Arditi et al.'s three criteria
and chose layer 27, whose added direction produces degenerate text. The rule now applies
all three - depth below 80% of the network, ablation within the perplexity budget, and
an addition screen requiring induced refusal above baseline (95% Wilson lower bound) on
the JailbreakBench held-out split. Phase 6 was re-run at both scales:

* **7B:** layers 0-12 fail the addition screen (2-5 of 30 induced against a 2/30
  baseline); layers 14 and 16 pass; **layer 14** is selected. On the bundled prompts it
  removes refusal completely (0/50) with fluent output, and adding it induces refusal on
  38/50 harmless prompts at 1.5x and 48/50 at 2x; random vectors 0/50.
* **1.5B:** the rule selects layer 16, as before.
* Every condition shared with the first run reproduced exactly.

The bundled set had been seen for every layer in the first run, so the corrected rule's
bundled result is reported as a check, not an out-of-sample validation.

Also: CI actions bumped from checkout@v4 / setup-python@v5 to v7 (Node 20 deprecation).
Tests: 169 -> 172.
