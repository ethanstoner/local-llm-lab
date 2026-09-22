# Runs every experiment in order and renders the figures.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -SkipSmoke
#
# Assumes scripts/setup_env.ps1 and scripts/fetch_model.ps1 have already run.
# Each stage is independent: one failing does not stop the rest, because a partial set
# of real measurements is worth more than an aborted run.

[CmdletBinding()]
param(
    [switch]$SkipSmoke,
    [int]$Device = 0
)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Output "venv not found. Run scripts/setup_env.ps1 first."
    exit 1
}

Set-Location $root
$failures = @()

function Invoke-Stage {
    param([string]$Name, [string[]]$Arguments)

    Write-Output ""
    Write-Output ("=" * 78)
    Write-Output "  $Name"
    Write-Output ("=" * 78)

    $started = Get-Date
    & $py @Arguments
    $code = $LASTEXITCODE
    $elapsed = (Get-Date) - $started

    if ($code -ne 0) {
        Write-Output ("  {0} exited with code {1} after {2:N0}s" -f $Name, $code, $elapsed.TotalSeconds)
        $script:failures += $Name
    } else {
        Write-Output ("  {0} completed in {1:N0}s" -f $Name, $elapsed.TotalSeconds)
    }
}

if (-not $SkipSmoke) {
    Invoke-Stage "Smoke test (1.5B)" @("-m", "src.benchmarks.run", "--config", "configs/smoke.yaml", "--device", $Device)
}

Invoke-Stage "Phase 1: context sweep" @("-m", "src.benchmarks.run", "--config", "configs/qwen2.5-7b.yaml", "--device", $Device)
Invoke-Stage "Phase 2: precision sweep" @("-m", "src.benchmarks.run", "--config", "configs/precision_sweep.yaml", "--device", $Device)
Invoke-Stage "Phase 2: quality comparison" @("-m", "src.evaluation.run", "--config", "configs/precision_sweep.yaml", "--device", $Device)
Invoke-Stage "Phases 4-5: refusal direction" @("-m", "src.interpretability.run", "--config", "configs/refusal.yaml", "--device", $Device)
Invoke-Stage "Phase 6: causal test of the direction" @("-m", "src.interpretability.intervene", "--config", "configs/intervention.yaml", "--device", $Device)
Invoke-Stage "Phase 5 on the 1.5B model" @("-m", "src.interpretability.run", "--config", "configs/refusal_1.5b.yaml", "--device", $Device)
Invoke-Stage "Phase 6 on the 1.5B model" @("-m", "src.interpretability.intervene", "--config", "configs/intervention_1.5b.yaml", "--device", $Device)
Invoke-Stage "Hardware ceilings" @("-m", "src.benchmarks.ceilings", "--device", $Device)
Invoke-Stage "Phase 7: decode attention A/B" @("-m", "src.benchmarks.interleaved", "--config", "configs/decode_ab.yaml", "--device", $Device)
Invoke-Stage "Phase 8: batch sweep" @("-m", "src.benchmarks.interleaved", "--config", "configs/batch_sweep.yaml", "--device", $Device)
Invoke-Stage "Roofline analysis" @("-m", "src.analysis.roofline")
Invoke-Stage "Figures" @("-m", "src.visualization.render")

Write-Output ""
Write-Output ("=" * 78)
if ($failures.Count -gt 0) {
    Write-Output "  Stages that did not exit cleanly: $($failures -join ', ')"
    Write-Output "  Check the run.log in each results/ directory."
    exit 1
}
Write-Output "  All stages completed. See results/ and figures/."
