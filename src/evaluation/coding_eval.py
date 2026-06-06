"""
Coding evaluation using EvalPlus (HumanEval+).

Generates Python function completions with Ollama, then executes them against
EvalPlus's extended test suite (80× more tests than base HumanEval).

Flow:
  1. Load N problems from HumanEval+ via evalplus.data
  2. Prompt Ollama to complete each function (DIRECT route, no RAG)
  3. Save solutions as JSONL in EvalPlus format
  4. Call evalplus.evaluate subprocess to execute and score
  5. Return pass@1 score

Usage:
  python -m src.evaluation.coding_eval --model qwen3:14b-q4_K_M --n 20
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

from config.settings import OLLAMA_BASE_URL, BENCHMARKS_DIR

logger = logging.getLogger(__name__)


@dataclass
class CodingResult:
    model_id:       str
    dataset:        str
    n_problems:     int
    pass_at_1:      float   # 0.0–1.0
    solved:         int
    duration_s:     float
    error:          str = ""


def _get_problems(dataset: str, n: int) -> dict:
    """Load first N problems from EvalPlus dataset (sorted by task_id for reproducibility)."""
    try:
        if dataset == "humaneval":
            from evalplus.data import get_human_eval_plus
            all_problems = get_human_eval_plus()
        elif dataset == "mbpp":
            from evalplus.data import get_mbpp_plus
            all_problems = get_mbpp_plus()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    except ImportError:
        raise RuntimeError("evalplus not installed. Run: pip install evalplus")

    sorted_ids = sorted(all_problems.keys())[:n]
    return {k: all_problems[k] for k in sorted_ids}


def _generate_solution(model_id: str, prompt: str, timeout_s: int = 60) -> str:
    """Ask Ollama to complete a HumanEval function. Returns the full solution string."""
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert Python programmer. Complete the given function. "
                    "Return ONLY valid Python code — the complete function implementation. "
                    "Do not include any explanation, markdown fences, or extra text. "
                    "Preserve the original function signature and docstring exactly as given."
                ),
            },
            {
                "role": "user",
                "content": f"Complete this Python function:\n\n{prompt}",
            },
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return _clean_solution(content, prompt)
    except Exception as exc:
        logger.error("Ollama generation failed: %s", exc)
        return prompt + "\n    pass\n"


def _clean_solution(raw: str, original_prompt: str) -> str:
    """
    Strip markdown fences and preamble. Return the prompt + body so EvalPlus
    gets a complete, executable function.
    """
    # Remove markdown fences
    raw = re.sub(r"```(?:python)?", "", raw, flags=re.IGNORECASE)
    raw = raw.strip()

    # If the model returned only the body (no def line), prepend the prompt
    if not raw.startswith("def ") and "def " not in raw[:50]:
        return original_prompt + "\n" + raw

    return raw


def _save_solutions(problems: dict, solutions: dict, out_path: Path) -> None:
    """Save solutions in EvalPlus JSONL format."""
    with out_path.open("w", encoding="utf-8") as f:
        for task_id, solution in solutions.items():
            f.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")


def _run_evalplus_evaluate(dataset: str, samples_path: Path) -> float:
    """Call evalplus.evaluate via subprocess. Returns pass@1 (0.0–1.0)."""
    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", dataset,
        "--samples", str(samples_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        return _parse_pass_at_1(output)
    except subprocess.TimeoutExpired:
        logger.error("evalplus.evaluate timed out")
        return 0.0
    except Exception as exc:
        logger.error("evalplus.evaluate failed: %s", exc)
        return 0.0


def _parse_pass_at_1(output: str) -> float:
    """Extract pass@1 from evalplus output lines like 'pass@1: 0.732'."""
    for line in output.split("\n"):
        low = line.lower()
        if "pass@1" in low or "pass_at_1" in low:
            # Try to find a float after colon or equals
            match = re.search(r"[:=]\s*([\d.]+)", line)
            if match:
                val = float(match.group(1))
                return val if val <= 1.0 else val / 100.0
    return 0.0


def run_coding_eval(
    model_id: str,
    dataset: str = "humaneval",
    n_problems: int = 20,
) -> CodingResult:
    """
    Run EvalPlus coding benchmark for one model.

    Args:
        model_id:   Ollama model tag (e.g. "qwen3:14b-q4_K_M")
        dataset:    "humaneval" or "mbpp"
        n_problems: how many problems to sample (default 20 for speed)
    """
    t0 = time.perf_counter()

    try:
        problems = _get_problems(dataset, n_problems)
    except RuntimeError as exc:
        return CodingResult(
            model_id=model_id, dataset=dataset,
            n_problems=0, pass_at_1=0.0, solved=0,
            duration_s=time.perf_counter() - t0, error=str(exc),
        )

    out_dir = BENCHMARKS_DIR / "evalplus" / model_id.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    solutions_path = out_dir / f"{dataset}_solutions.jsonl"

    logger.info("Coding eval | %s | %s | %d problems", model_id, dataset, len(problems))

    solutions: dict[str, str] = {}
    for i, (task_id, problem) in enumerate(problems.items()):
        logger.info("  [%d/%d] %s", i + 1, len(problems), task_id)
        solution = _generate_solution(model_id, problem["prompt"])
        solutions[task_id] = solution

    _save_solutions(problems, solutions, solutions_path)

    pass_at_1 = _run_evalplus_evaluate(dataset, solutions_path)
    solved = round(pass_at_1 * len(problems))

    result = CodingResult(
        model_id=model_id,
        dataset=dataset,
        n_problems=len(problems),
        pass_at_1=round(pass_at_1, 4),
        solved=solved,
        duration_s=round(time.perf_counter() - t0, 1),
    )
    logger.info("Coding result | %s | pass@1=%.3f (%d/%d) | %.1fs",
                model_id, pass_at_1, solved, len(problems), result.duration_s)
    return result


def results_to_dicts(results: list[CodingResult]) -> list[dict]:
    return [asdict(r) for r in results]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:14b-q4_K_M")
    parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_coding_eval(args.model, args.dataset, args.n)
    print(json.dumps(asdict(result), indent=2))
