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
import time
from collections.abc import Callable
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


def _generate_solution(model_id: str, prompt: str, timeout_s: int = 120) -> str:
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


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DEF_RE   = re.compile(r"^def \w+\(", re.MULTILINE)


def _is_valid_python(code: str) -> bool:
    # compile() catches errors ast.parse() doesn't, e.g. `return` outside a
    # function — common when a body-only completion loses its indentation.
    try:
        compile(code, "<solution>", "exec")
        return True
    except SyntaxError:
        return False


def _clean_solution(raw: str, original_prompt: str) -> str:
    """
    Strip <think> traces and markdown fences, then locate the actual code.
    Return the prompt + body so EvalPlus gets a complete, executable function.

    Reasoning models often narrate a solution in prose instead of (or around)
    code — falls back to a `pass` stub (fails tests but is at least valid
    Python) rather than handing EvalPlus a syntax error.
    """
    # Some Ollama templates leave <think>...</think> in `content` verbatim.
    raw = _THINK_RE.sub("", raw).strip()

    # Reasoning models often narrate first, then place the final answer in a
    # fenced code block — prefer the last one if present.
    fences = _FENCE_RE.findall(raw)
    if fences:
        raw = fences[-1].strip()

    # A top-level `def` likely means the model returned a complete, self-
    # contained redefinition — drop any leading prose before it.
    match = _DEF_RE.search(raw)
    if match:
        candidate = raw[match.start():]
        if _is_valid_python(candidate):
            return candidate

    # Otherwise treat raw as the function body to append to the prompt.
    candidate = original_prompt + "\n" + raw
    if _is_valid_python(candidate):
        return candidate

    return original_prompt + "\n    pass\n"


def _save_solutions(problems: dict, solutions: dict, out_path: Path) -> None:
    """Save solutions in EvalPlus JSONL format."""
    with out_path.open("w", encoding="utf-8") as f:
        for task_id, solution in solutions.items():
            f.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")


def _evaluate_solutions_inline(
    problems_subset: dict,
    solutions: dict,
    dataset: str = "humaneval",
) -> float:
    """
    Evaluate solutions using evalplus internals directly (no subprocess).
    Works on Windows — avoids evalplus.evaluate which requires POSIX 'resource' module.
    Uses the cached ground-truth expected outputs; only runs untrusted_check per problem.
    """
    try:
        from evalplus.eval import PASS
        from evalplus.evaluate import check_correctness, get_groundtruth
        if dataset == "humaneval":
            from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
            all_problems = get_human_eval_plus()
            dataset_hash = get_human_eval_plus_hash()
            tasks_only_output_not_none: list = []
        elif dataset == "mbpp":
            from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
            from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
            all_problems = get_mbpp_plus()
            dataset_hash = get_mbpp_plus_hash()
            tasks_only_output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        expected_output = get_groundtruth(all_problems, dataset_hash, tasks_only_output_not_none)
    except Exception as exc:
        logger.error("Inline evaluator setup failed: %s", exc)
        return 0.0

    passed = 0
    total = len(solutions)

    for task_id, solution in solutions.items():
        if task_id not in all_problems:
            logger.warning("Task %s not in full dataset — skipping", task_id)
            total -= 1
            continue

        problem = all_problems[task_id]
        gt = expected_output[task_id]

        try:
            result = check_correctness(
                dataset=dataset,
                completion_id=0,
                problem=problem,
                solution=solution,
                expected_output=gt,
                identifier=f"{task_id}/0",
            )
            base_ok = result["base"][0] == PASS
            plus_ok = result.get("plus", (PASS,))[0] == PASS
            if base_ok and plus_ok:
                passed += 1
        except Exception as exc:
            logger.warning("check_correctness failed for %s: %s", task_id, exc)

    return passed / max(total, 1)


def run_coding_eval(
    model_id: str,
    dataset: str = "humaneval",
    n_problems: int = 20,
    progress: "Callable[[int, int, str], None] | None" = None,
) -> CodingResult:
    """
    Run EvalPlus coding benchmark for one model.

    Args:
        model_id:   Ollama model tag (e.g. "qwen3:14b-q4_K_M")
        dataset:    "humaneval" or "mbpp"
        n_problems: how many problems to sample (default 20 for speed)
        progress:   optional callback(step, total, label) for live UI updates
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

    n_total = len(problems)
    solutions: dict[str, str] = {}
    for i, (task_id, problem) in enumerate(problems.items()):
        logger.info("  [%d/%d] %s", i + 1, n_total, task_id)
        if progress:
            progress(i, n_total, f"problem {i + 1}/{n_total}: {task_id}")
        solution = _generate_solution(model_id, problem["prompt"])
        solutions[task_id] = solution
    if progress:
        progress(n_total, n_total, "evaluating solutions")

    _save_solutions(problems, solutions, solutions_path)

    pass_at_1 = _evaluate_solutions_inline(problems, solutions, dataset)
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
