# Creates the project-local virtual environment and installs pinned dependencies.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
#
# Two things here are deliberate rather than incidental:
#
#  * The torch wheel is fetched with a resumable downloader instead of being left to
#    pip. pip's connection to the PyTorch CDN stalled indefinitely during this project's
#    setup - no bytes, no timeout, no error - and a 2.5 GB download needs to be able to
#    report progress and resume.
#  * Installs run strictly sequentially and the venv's own pip is never upgraded. A
#    concurrent pip install against this venv can strip torch's bundled CUDA libraries.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "venv\Scripts\python.exe"

$TorchVersion = "2.6.0+cu124"
$TorchWheel = "torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl"
$TorchUrl = "https://download.pytorch.org/whl/cu124/$TorchWheel"

function Get-FileResumable {
    param([string]$Url, [string]$Destination, [int]$MaxAttempts = 6)

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        $existing = 0
        if (Test-Path $Destination) { $existing = (Get-Item $Destination).Length }

        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.Timeout = 30000
        $request.ReadWriteTimeout = 30000
        if ($existing -gt 0) { $request.AddRange($existing) }
        $file = $null

        try {
            $response = $request.GetResponse()
            $expected = $existing + $response.ContentLength
            $stream = $response.GetResponseStream()
            $mode = if ($existing -gt 0) { [System.IO.FileMode]::Append } else { [System.IO.FileMode]::Create }
            $file = New-Object System.IO.FileStream($Destination, $mode, [System.IO.FileAccess]::Write)

            $buffer = New-Object byte[] 1048576
            $total = $existing
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $lastReport = $existing

            while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $file.Write($buffer, 0, $read)
                $total += $read
                if (($total - $lastReport) -ge 250MB) {
                    $lastReport = $total
                    Write-Output ("  {0:N0} / {1:N0} MB at {2:N1} MB/s" -f ($total / 1MB), ($expected / 1MB), (($total - $existing) / 1MB / $sw.Elapsed.TotalSeconds))
                }
            }

            $file.Close(); $file = $null
            $stream.Close(); $response.Close()

            if ((Get-Item $Destination).Length -ge $expected) {
                Write-Output ("  complete: {0:N0} MB" -f ((Get-Item $Destination).Length / 1MB))
                return
            }
            Write-Output "  short read, resuming (attempt $attempt)"
        } catch {
            if ($file) { try { $file.Close() } catch {} }
            # A 416 means the server says there is nothing left to send.
            if ($_.Exception.Message -match "416") { Write-Output "  already complete"; return }
            Write-Output "  attempt $attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 3
        }
    }
    throw "could not download $Url after $MaxAttempts attempts"
}

if (-not (Test-Path $py)) {
    Write-Output "Creating venv at $root\venv"
    py -3.12 -m venv (Join-Path $root "venv")
}

Write-Output "--- torch $TorchVersion (CUDA 12.4 build) ---"
# Read the version from the installed package rather than running python with its stderr
# redirected. Under Windows PowerShell 5.1, `2>$null` on a native command wraps every
# stderr line in a NativeCommandError, which $ErrorActionPreference = "Stop" makes fatal
# even when the command succeeded.
$already = ""
$torchVersionFile = Join-Path $root "venv\Lib\site-packages\torch\version.py"
if (Test-Path $torchVersionFile) {
    $match = Select-String -Path $torchVersionFile -Pattern "^__version__\s*=\s*['""]([^'""]+)['""]"
    if ($match) { $already = $match.Matches[0].Groups[1].Value }
}
if ($already -eq $TorchVersion) {
    Write-Output "  already installed"
} else {
    $cache = Join-Path $env:TEMP "local-llm-lab-wheels"
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    $wheelPath = Join-Path $cache ($TorchWheel -replace "%2B", "+")
    Get-FileResumable -Url $TorchUrl -Destination $wheelPath
    & $py -m pip install --no-input $wheelPath
}

Write-Output "--- transformers stack ---"
& $py -m pip install --no-input "transformers==4.57.0" "accelerate==1.12.0" "safetensors==0.7.0" "huggingface_hub>=0.34.0,<1.0" "datasets==4.5.0"

Write-Output "--- quantization ---"
& $py -m pip install --no-input "bitsandbytes==0.49.0"

Write-Output "--- analysis and plotting ---"
& $py -m pip install --no-input "numpy==2.2.6" "pandas==2.3.3" "matplotlib==3.10.7" "pyyaml==6.0.2" "nvidia-ml-py"

Write-Output "--- test tooling ---"
& $py -m pip install --no-input "pytest"

Write-Output "--- verification ---"
& $py -c "import torch, transformers, bitsandbytes, pynvml; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('transformers', transformers.__version__); print('bitsandbytes', bitsandbytes.__version__)"

# pip freeze records the locally downloaded torch wheel as a file:// URL into this
# machine's temp directory, which nobody else can install from. Pin it by version
# instead (installable from the cu124 index), and write without a byte-order mark.
$lock = & $py -m pip freeze | ForEach-Object {
    if ($_ -match '^torch @ file:') { 'torch==2.6.0+cu124' } else { $_ }
}
[System.IO.File]::WriteAllLines((Join-Path $root "requirements.lock.txt"), [string[]]$lock)
Write-Output "Wrote requirements.lock.txt"
