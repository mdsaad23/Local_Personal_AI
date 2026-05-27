# Pull all models in the benchmark matrix.
# Run this before starting the benchmark harness.
# Total download: ~45 GB — run when on a fast connection with time to spare.
#
# Usage: .\scripts\pull_models.ps1
# Skip a model:  comment out its line below.

$models = @(
    # ── Small baseline (fast, ~2-4 GB each) ──────────────────────────────────
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q8_0",

    # ── Mid-tier 7-8B (workhorse tier, ~5-9 GB each) ─────────────────────────
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q8_0",
    "mistral:7b-instruct-v0.3-q4_K_M",

    # ── Upper tier 12-14B (quality tier, ~8-11 GB each) ──────────────────────
    "phi4:14b-q4_K_M",
    "gemma3:12b-it-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",   # production model
    "qwen2.5:14b-instruct-q5_K_M",

    # ── Embedding + vision (always needed) ────────────────────────────────────
    "nomic-embed-text",
    "minicpm-v"
)

$total = $models.Count
$i = 0

foreach ($model in $models) {
    $i++
    Write-Output ""
    Write-Output "[$i/$total] Pulling $model ..."
    $start = Get-Date
    ollama pull $model
    $elapsed = ((Get-Date) - $start).TotalSeconds
    Write-Output "[$i/$total] Done in $([math]::Round($elapsed))s"
}

Write-Output ""
Write-Output "All models pulled. Run verify_gpu.py to confirm Ollama sees them."
ollama list
