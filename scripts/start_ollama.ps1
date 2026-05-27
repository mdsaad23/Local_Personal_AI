# Starts Ollama with AMD GPU (Vulkan) enabled.
# Persistent env vars (OLLAMA_VULKAN=1, HSA_OVERRIDE_GFX_VERSION=12.0.1) are
# set in HKCU registry — any new process inherits them automatically.
#
# Usage: .\scripts\start_ollama.ps1

# Ensure env vars are set for this session too (belt-and-suspenders)
$env:OLLAMA_VULKAN = "1"
$env:HSA_OVERRIDE_GFX_VERSION = "12.0.1"
$env:OLLAMA_MODELS = "D:\ollama_models"

# Also write to registry in case they got cleared
[System.Environment]::SetEnvironmentVariable("OLLAMA_VULKAN", "1", "User")
[System.Environment]::SetEnvironmentVariable("HSA_OVERRIDE_GFX_VERSION", "12.0.1", "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\ollama_models", "User")

# Kill any existing Ollama instances (tray app + serve)
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
$env_table["OLLAMA_VULKAN"] = "1"
$env_table["HSA_OVERRIDE_GFX_VERSION"] = "12.0.1"
$env_table["OLLAMA_MODELS"] = "D:\ollama_models"

Start-Process `
    "C:\Users\saadm\AppData\Local\Programs\Ollama\ollama.exe" `
    -ArgumentList "serve" `
    -WindowStyle Hidden `
    -Environment $env_table

# Wait up to 15 seconds for Ollama to be ready
$ready = $false
for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $version = (Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 2).version
        Write-Output "Ollama $version is running (waited ${i}s)"
        $ready = $true
        break
    } catch { }
}

if ($ready) {
    Write-Output "GPU: VULKAN=1 — verify with: python scripts/verify_gpu.py"
} else {
    Write-Output "ERROR: Ollama did not respond after 15s. Check:"
    Write-Output "  $env:LOCALAPPDATA\Ollama\server.log"
}
