# Start FastAPI dev server (hot-reload)
# Usage: .\scripts\start_api.ps1

$Root = Split-Path $PSScriptRoot -Parent
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "venv not found at $VenvPython — run: python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt"
    exit 1
}

Set-Location $Root

# Load .env
$envFile = Join-Path $Root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
        $parts = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
    }
}

Write-Host "Starting FastAPI on http://localhost:8000" -ForegroundColor Cyan
Write-Host "Vite dev server: cd ui/web && npm run dev  (http://localhost:5173)" -ForegroundColor DarkCyan
Write-Host ""

& $VenvPython -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
