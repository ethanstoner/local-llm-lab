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
| Tests | `tests/` - 107 tests, CPU-only and network-free |

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

---

## Next steps

1. Phase 1 sweep on Qwen2.5-7B-Instruct across 128 - 16384 token contexts.
2. Phase 2 precision sweep including `fp32`, which is expected to OOM and should be
   *recorded* as an OOM rather than crash the sweep.
3. Phase 2 quality comparison against the bf16 reference.
4. Render figures and inspect every one of them before claiming they are correct.
5. Phase 5 refusal-direction analysis, analysis only.
6. Finish README with measured results and limitations.
