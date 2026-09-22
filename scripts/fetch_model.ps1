# Downloads a Hugging Face model repository into a local directory.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model.ps1 -Repo Qwen/Qwen2.5-7B-Instruct
#
# Why this exists rather than `huggingface_hub.snapshot_download`:
#
# On this machine, Python's HTTP stack stalls indefinitely on large streaming downloads.
# pip hung on a 2.5 GB torch wheel for 25 minutes with zero bytes transferred and no
# timeout; huggingface_hub then hung the same way at 608 MB of a 3.1 GB model, both with
# and without hf_xet. A .NET HttpWebRequest to the same URLs sustains 8-13 MB/s. So the
# transfer is done here, with explicit ranged resume and per-attempt timeouts, and
# transformers is pointed at the resulting directory.
#
# Set `model.local_path` in the experiment config to the directory this writes.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Repo,
    [string]$Revision = "main",
    [string]$Destination = "models",
    [int]$MaxAttempts = 8
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$target = Join-Path $root (Join-Path $Destination ($Repo -replace ".*/", ""))
New-Item -ItemType Directory -Force -Path $target | Out-Null

Write-Output "Repository : $Repo@$Revision"
Write-Output "Destination: $target"

# Larger buffers and no expect-continue handshake; this transfer is the whole point.
[System.Net.ServicePointManager]::DefaultConnectionLimit = 16
[System.Net.ServicePointManager]::Expect100Continue = $false

function Invoke-HubApi {
    param([string]$Path)
    $response = Invoke-WebRequest -Uri "https://huggingface.co/api/$Path" -UseBasicParsing -TimeoutSec 30
    return $response.Content | ConvertFrom-Json
}

function Get-FileResumable {
    param([string]$Url, [string]$Destination, [long]$ExpectedSize, [int]$MaxAttempts)

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        $existing = 0
        if (Test-Path $Destination) { $existing = (Get-Item $Destination).Length }
        if ($ExpectedSize -gt 0 -and $existing -ge $ExpectedSize) { return $true }

        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.Timeout = 30000
        $request.ReadWriteTimeout = 30000
        $request.UserAgent = "local-llm-lab/0.1"
        if ($existing -gt 0) { $request.AddRange($existing) }
        $file = $null

        try {
            $response = $request.GetResponse()
            $stream = $response.GetResponseStream()
            $mode = if ($existing -gt 0) { [System.IO.FileMode]::Append } else { [System.IO.FileMode]::Create }
            $file = New-Object System.IO.FileStream($Destination, $mode, [System.IO.FileAccess]::Write)

            $buffer = New-Object byte[] 4194304
            $total = $existing
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $lastReport = $existing

            while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $file.Write($buffer, 0, $read)
                $total += $read
                if (($total - $lastReport) -ge 500MB) {
                    $lastReport = $total
                    Write-Output ("    {0:N0} / {1:N0} MB at {2:N1} MB/s" -f ($total / 1MB), ($ExpectedSize / 1MB), (($total - $existing) / 1MB / $sw.Elapsed.TotalSeconds))
                }
            }

            $file.Close(); $file = $null
            $stream.Close(); $response.Close()

            $size = (Get-Item $Destination).Length
            if ($ExpectedSize -le 0 -or $size -ge $ExpectedSize) { return $true }
            Write-Output ("    short at {0:N0} MB, resuming (attempt {1})" -f ($size / 1MB), $attempt)
        } catch {
            if ($file) { try { $file.Close() } catch {} }
            if ($_.Exception.Message -match "416") { return $true }
            Write-Output "    attempt $attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 3
        }
    }
    return $false
}

$info = Invoke-HubApi "models/$Repo/revision/$Revision"
$sha = $info.sha
Write-Output "Resolved commit: $sha"

$tree = Invoke-HubApi "models/$Repo/tree/$Revision`?recursive=true"
$files = @($tree | Where-Object { $_.type -eq "file" })

# The weights are the point; the rest of the repo is documentation and git plumbing.
$skip = @(".gitattributes", "README.md", "LICENSE", "LICENSE.txt", "NOTICE")
$files = @($files | Where-Object { $skip -notcontains $_.path -and $_.path -notlike "*.md" })

$totalBytes = ($files | Measure-Object -Property size -Sum).Sum
Write-Output ("{0} files, {1:N2} GB total" -f $files.Count, ($totalBytes / 1GB))

$failed = @()
foreach ($file in $files) {
    $filePath = Join-Path $target $file.path
    New-Item -ItemType Directory -Force -Path (Split-Path $filePath) | Out-Null

    if ((Test-Path $filePath) -and (Get-Item $filePath).Length -eq $file.size) {
        Write-Output ("  [have] {0}" -f $file.path)
        continue
    }

    Write-Output ("  [get ] {0} ({1:N1} MB)" -f $file.path, ($file.size / 1MB))
    $url = "https://huggingface.co/$Repo/resolve/$Revision/$($file.path)"
    if (-not (Get-FileResumable -Url $url -Destination $filePath -ExpectedSize $file.size -MaxAttempts $MaxAttempts)) {
        $failed += $file.path
    }
}

if ($failed.Count -gt 0) {
    Write-Output "FAILED: $($failed -join ', ')"
    exit 1
}

# Provenance: the config records the Hub id, this records exactly what was fetched.
$metadata = [ordered]@{
    repo           = $Repo
    revision       = $Revision
    resolved_commit = $sha
    fetched_utc    = (Get-Date).ToUniversalTime().ToString("o")
    n_files        = $files.Count
    total_bytes    = $totalBytes
}
$metadata | ConvertTo-Json | Out-File -FilePath (Join-Path $target "_fetch_metadata.json") -Encoding utf8

Write-Output ""
Write-Output "Complete. Point a config at it with:"
Write-Output "  model:"
Write-Output "    id: $Repo"
Write-Output "    local_path: $Destination/$($Repo -replace '.*/', '')"
