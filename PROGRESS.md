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

---

## Next steps

1. Phase 1 sweep on Qwen2.5-7B-Instruct across 128 - 16384 token contexts.
2. Phase 2 precision sweep including `fp32`, which is expected to OOM and should be
   *recorded* as an OOM rather than crash the sweep.
3. Phase 2 quality comparison against the bf16 reference.
4. Render figures and inspect every one of them before claiming they are correct.
5. Phase 5 refusal-direction analysis, analysis only.
6. Finish README with measured results and limitations.
