"""
Run this before anything else.
Verifies Ollama is running, detects GPU backend, and reports a baseline
inference speed so we know whether GPU acceleration is active.
"""

import subprocess
import sys
import time

import httpx


OLLAMA_URL = "http://localhost:11434"
TEST_MODEL = "llama3.2:3b-instruct-q4_K_M"   # small — pull if absent


def check_ollama_running() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/version", timeout=5)
        version = r.json().get("version", "unknown")
        print(f"[OK] Ollama running — version {version}")
        return True
    except Exception:
        print("[FAIL] Ollama not reachable at http://localhost:11434")
        print("       Start it with: ollama serve")
        return False


def list_gpu_info() -> None:
    """Try rocm-smi (HIP SDK) then fall back to reporting Vulkan only."""
    print("\n── GPU detection ──────────────────────────────────────────")
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("[HIP] rocm-smi found — AMD GPU via HIP backend:")
            for line in result.stdout.strip().splitlines():
                print(f"      {line}")
        else:
            print("[WARN] rocm-smi present but returned an error:")
            print(f"       {result.stderr.strip()}")
    except FileNotFoundError:
        print("[INFO] rocm-smi not found (Linux/HIP tool — not needed on Windows Vulkan path)")
        print("       Using Vulkan backend. GPU speed test below confirms acceleration.")
    except Exception as e:
        print(f"[WARN] rocm-smi check failed: {e}")


def check_ollama_gpu() -> None:
    """Pull a small model and stream one response, measuring tokens/sec."""
    print(f"\n── Inference test ({TEST_MODEL}) ────────────────────────────")

    # Pull model if not present
    print(f"Checking model is available...")
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": TEST_MODEL, "stream": False},
            timeout=300,
        )
        if r.status_code == 200:
            print(f"[OK] Model ready")
    except Exception as e:
        print(f"[WARN] Could not verify model: {e}")
        return

    # Run a timed inference
    prompt = "List three properties of transformer attention mechanisms."
    print(f"Running inference prompt...")
    print(f"Prompt: '{prompt}'\n")

    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    response_text = ""

    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": TEST_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"num_ctx": 2048},
            },
            timeout=120,
        ) as resp:
            import json
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    eval_count    = data.get("eval_count", 0)
                    eval_duration = data.get("eval_duration", 1) / 1e9  # ns → s
                    prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1e9
                    break
                chunk = data.get("response", "")
                if chunk:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
                    response_text += chunk

    except Exception as e:
        print(f"[FAIL] Inference error: {e}")
        return

    total_time = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else -1
    tgs  = eval_count / eval_duration if eval_duration > 0 else 0

    print(f"Response:\n{response_text.strip()}\n")
    print("── Performance ─────────────────────────────────────────────")
    print(f"  TTFT (time to first token) : {ttft:.2f}s")
    print(f"  Token generation speed     : {tgs:.1f} tok/s")
    print(f"  Total time                 : {total_time:.2f}s")
    print(f"  Tokens generated           : {eval_count}")

    print("\n── GPU assessment ──────────────────────────────────────────")
    if tgs >= 40:
        print(f"[OK] {tgs:.1f} tok/s — GPU acceleration confirmed (HIP or Vulkan)")
    elif tgs >= 15:
        print(f"[WARN] {tgs:.1f} tok/s — possible Vulkan or partial GPU use")
        print("       Install AMD HIP SDK to enable full GPU acceleration")
    else:
        print(f"[FAIL] {tgs:.1f} tok/s — CPU only (no GPU acceleration detected)")
        print("       Install AMD HIP SDK from: amd.com/en/developer/rocm.html")

    print("\n── What Ollama reports ─────────────────────────────────────")
    try:
        ps = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=5).json()
        for m in ps.get("models", []):
            print(f"  Model : {m.get('name')}")
            print(f"  Size  : {m.get('size', 0) / 1e9:.1f} GB")
            details = m.get("details", {})
            print(f"  Quant : {details.get('quantization_level', 'unknown')}")
    except Exception:
        pass


def main() -> None:
    print("=" * 60)
    print("  Local AI Assistant - GPU Verification")
    print("=" * 60)

    if not check_ollama_running():
        sys.exit(1)

    list_gpu_info()
    check_ollama_gpu()

    print("\n" + "=" * 60)
    print("  Done. Fix any [FAIL] or [WARN] items before benchmarking.")
    print("=" * 60)


if __name__ == "__main__":
    main()
