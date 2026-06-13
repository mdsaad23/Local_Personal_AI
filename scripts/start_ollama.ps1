# Starts Ollama with AMD GPU (Vulkan) enabled.
# Persistent env vars (OLLAMA_VULKAN=1, HSA_OVERRIDE_GFX_VERSION=12.0.1) are
# set in HKCU registry — any new process inherits them automatically.
#
# Usage: .\scripts\start_ollama.ps1

# Ensure env vars are set for this session too (belt-and-suspenders)
$env:OLLAMA_VULKAN             = "1"
$env:HSA_OVERRIDE_GFX_VERSION  = "12.0.1"
$env:OLLAMA_MODELS             = "D:\ollama_models"
$env:GGML_VK_VISIBLE_DEVICES   = "0"   # Use only Vulkan0 (RX 9070 XT) — excludes iGPU
# Default request context. The positional-recall (codeneedle) suite stuffs whole
# source files into context (http_server ≈14K tokens); the default 4K would
# truncate them. 32K fits http_server with headroom. Raise for the jquery corpus.
$env:OLLAMA_CONTEXT_LENGTH     = "32768"

# Also write to registry in case they got cleared
[System.Environment]::SetEnvironmentVariable("OLLAMA_VULKAN",            "1",                "User")
[System.Environment]::SetEnvironmentVariable("HSA_OVERRIDE_GFX_VERSION", "12.0.1",           "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS",            "D:\ollama_models", "User")
[System.Environment]::SetEnvironmentVariable("GGML_VK_VISIBLE_DEVICES",  "0",                "User")

# ── Check if already running ────────────────────────────────────────────────────
# NB: use 127.0.0.1, NOT localhost. Ollama binds IPv4 only, but `localhost`
# resolves to ::1 (IPv6) first on Windows — Invoke-RestMethod then times out
# trying IPv6 before falling back, so a healthy server looks dead.
try {
    $existingVersion = (Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2).version
    Write-Output "Ollama $existingVersion is already running — skipping restart."
    Write-Output "GPU: VULKAN=1 — verify with: python scripts/verify_gpu.py"
    exit 0
} catch { }

# ── Kill any existing Ollama instances (tray app + serve) ───────────────────────
$running = Get-Process -Name "ollama*" -ErrorAction SilentlyContinue
if ($running) {
    Write-Output "Stopping existing Ollama processes ($($running.Count) found)..."
    $running | Stop-Process -Force
    Start-Sleep -Seconds 3
}

Write-Output "Starting Ollama with Vulkan GPU support..."

# Use -Environment to explicitly pass vars to the child process (PowerShell 7+)
$env_table = @{}
[System.Environment]::GetEnvironmentVariables().GetEnumerator() | ForEach-Object {
    $env_table[$_.Key] = $_.Value
}
$env_table["OLLAMA_VULKAN"]            = "1"
$env_table["HSA_OVERRIDE_GFX_VERSION"] = "12.0.1"
$env_table["OLLAMA_MODELS"]            = "D:\ollama_models"
$env_table["GGML_VK_VISIBLE_DEVICES"]  = "0"
$env_table["OLLAMA_CONTEXT_LENGTH"]    = "32768"

Start-Process `
    "C:\Users\saadm\AppData\Local\Programs\Ollama\ollama.exe" `
    -ArgumentList "serve" `
    -WindowStyle Hidden `
    -Environment $env_table

# Wait up to 60 seconds for Ollama to respond
$ready = $false
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $version = (Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2).version
        Write-Output "Ollama $version is running (waited ${i}s)"
        $ready = $true
        break
    } catch { }
}

if ($ready) {
    Write-Output "GPU: VULKAN=1 — verify with: python scripts/verify_gpu.py"
} else {
    Write-Output "ERROR: Ollama did not respond after 60s. Check:"
    Write-Output "  $env:LOCALAPPDATA\Ollama\server.log"
}
