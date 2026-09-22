# Fetches the JailbreakBench behaviour sets used by the refusal-direction analysis.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\fetch_datasets.ps1
#
# Two small CSVs, ~44 KB total. They are fetched here rather than through
# `datasets.load_dataset` for the same reason the model weights are (see
# scripts/fetch_model.ps1): Python's HTTP stack stalls on this machine.
#
# The files are NOT committed - this repository does not vendor third-party datasets.
# Run this script to reproduce them.
#
# Source: JailbreakBench/JBB-Behaviors, Chao et al. 2024, arXiv:2404.01318 (MIT licence).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "data\jbb"
New-Item -ItemType Directory -Force -Path $target | Out-Null

$base = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data"
$files = @("harmful-behaviors.csv", "benign-behaviors.csv", "../LICENSE")

foreach ($name in $files) {
    $leaf = Split-Path $name -Leaf
    $out = Join-Path $target $leaf
    Write-Output "  $leaf"
    Invoke-WebRequest -Uri "$base/$name" -OutFile $out -UseBasicParsing -TimeoutSec 60
}

Write-Output ""
Write-Output "Fetched to $target"
Get-ChildItem $target | Select-Object Name, Length | Format-Table -AutoSize
