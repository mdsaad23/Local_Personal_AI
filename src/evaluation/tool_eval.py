"""
Tool / function-calling evaluation.

Tests whether a model supports Ollama's native tools API and correctly
invokes functions. Scores 5 standard tasks across:
  - tool_supported: did the model return a tool_calls field?
  - correct_fn:     did it call the right function?
  - correct_args:   did the arguments contain the expected values?

No external libraries required — uses Ollama /api/chat directly.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict

import httpx

from config.settings import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# ── Task definitions ───────────────────────────────────────────────────────────
# Each task: question + single tool + expected function name + arg check dict
# arg check: {key: required_value_substring | None (just check key present)}

TOOL_TASKS = [
    {
        "id": "weather",
        "question": "What is the weather like in Dubai right now?",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }],
        "expected_fn": "get_weather",
        "expected_args": {"city": "dubai"},
    },
    {
        "id": "calculator",
        "question": "What is 1_847 multiplied by 293? Use the calculator tool.",
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression as a string"},
                    },
                    "required": ["expression"],
                },
            },
        }],
        "expected_fn": "calculate",
        "expected_args": {"expression": None},  # just check key present
    },
    {
        "id": "search",
        "question": "Search for recent research papers on retrieval augmented generation.",
        "tools": [{
            "type": "function",
            "function": {
                "name": "search_papers",
                "description": "Search academic papers by topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }],
        "expected_fn": "search_papers",
        "expected_args": {"query": None},
    },
    {
        "id": "calendar",
        "question": "Schedule a meeting called 'Project Sync' with alice@company.com for tomorrow at 2pm.",
        "tools": [{
            "type": "function",
            "function": {
                "name": "create_event",
                "description": "Create a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "attendees": {"type": "array", "items": {"type": "string"}},
                        "datetime_iso": {"type": "string", "description": "ISO 8601 format"},
                    },
                    "required": ["title", "datetime_iso"],
                },
            },
        }],
        "expected_fn": "create_event",
        "expected_args": {"title": "sync"},  # substring match
    },
    {
        "id": "email",
        "question": "Send an email to bob@example.com with subject 'Quarterly Report' and a short summary body.",
        "tools": [{
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        }],
        "expected_fn": "send_email",
        "expected_args": {"to": "bob@example.com"},
    },
]


@dataclass
class ToolResult:
    model_id:      str
    task_id:       str
    tool_supported: bool   # model returned tool_calls
    correct_fn:    bool    # called the right function
    correct_args:  bool    # args match expectations
    latency_s:     float
    raw_call:      str     # truncated JSON for inspection


def _check_args(args: dict, expected: dict) -> bool:
    for key, val in expected.items():
        if key not in args:
            return False
        if val is not None and val.lower() not in str(args[key]).lower():
            return False
    return True


def run_tool_eval(model_id: str) -> list[ToolResult]:
    """Run all 5 tool-calling tasks for one model."""
    results: list[ToolResult] = []

    for task in TOOL_TASKS:
        logger.info("Tool eval | %s | task=%s", model_id, task["id"])
        t0 = time.perf_counter()

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": task["question"]}],
            "tools": task["tools"],
            "stream": False,
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Tool eval error (%s / %s): %s", model_id, task["id"], exc)
            results.append(ToolResult(
                model_id=model_id, task_id=task["id"],
                tool_supported=False, correct_fn=False, correct_args=False,
                latency_s=round(time.perf_counter() - t0, 3),
                raw_call=str(exc)[:200],
            ))
            continue

        latency = round(time.perf_counter() - t0, 3)
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls") or []

        supported = bool(tool_calls)
        correct_fn = False
        correct_args = False
        raw = ""

        if supported:
            call = tool_calls[0]
            fn_name = call.get("function", {}).get("name", "")
            raw_args = call.get("function", {}).get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            correct_fn   = fn_name == task["expected_fn"]
            correct_args = correct_fn and _check_args(raw_args, task["expected_args"])
            raw = json.dumps({"name": fn_name, "args": raw_args})[:400]

        logger.info("  supported=%s fn_ok=%s args_ok=%s latency=%.2fs",
                    supported, correct_fn, correct_args, latency)
        results.append(ToolResult(
            model_id=model_id, task_id=task["id"],
            tool_supported=supported, correct_fn=correct_fn, correct_args=correct_args,
            latency_s=latency, raw_call=raw,
        ))

    return results


def results_to_dicts(results: list[ToolResult]) -> list[dict]:
    return [asdict(r) for r in results]
