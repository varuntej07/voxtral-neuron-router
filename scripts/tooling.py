"""Shared helpers: paths, the frozen tool snapshot, and prompt renderings of it."""

import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
TOOLS_PATH = ROOT / "data" / "tools.json"
ROWS_PATH = ROOT / "data" / "text" / "rows.jsonl"
RESULTS_DIR = ROOT / "results"

NO_TOOL = "no_tool"
DEFAULT_MODEL = "gpt-4.1-mini"


def _load_dotenv() -> None:
    """Read KEY=value lines from the gitignored .env; a real env var always wins."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


_load_dotenv()


@lru_cache(maxsize=1)
def load_tools() -> list[dict]:
    return json.loads(TOOLS_PATH.read_text(encoding="utf-8"))


def tools_by_name() -> dict[str, dict]:
    return {t["name"]: t for t in load_tools()}


def label_space() -> list[str]:
    return [t["name"] for t in load_tools()] + [NO_TOOL]


def first_sentence(text: str, limit: int = 140) -> str:
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return sentence if len(sentence) <= limit else sentence[: limit - 3] + "..."


def arg_names(tool: dict) -> list[str]:
    return list(tool["parameters"].get("properties", {}))


def compact_catalog() -> str:
    """One line per tool: name, argument names, first sentence. About 370 tokens for 34 tools."""
    lines = []
    for tool in load_tools():
        lines.append(f"- {tool['name']}({', '.join(arg_names(tool))}): {first_sentence(tool['description'])}")
    return "\n".join(lines)


def names_catalog() -> str:
    return "\n".join(f"- {t['name']}" for t in load_tools())


def openai_tools(names: list[str] | None = None) -> list[dict]:
    """Full schemas in OpenAI function-calling shape, in snapshot order so prefixes stay cacheable."""
    wanted = set(names) if names is not None else None
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in load_tools()
        if wanted is None or t["name"] in wanted
    ]


def args_valid(tool_name: str, arguments: dict) -> bool:
    if tool_name == NO_TOOL:
        return arguments == {}
    schema = tools_by_name()[tool_name]["parameters"]
    return not any(Draft202012Validator(schema).iter_errors(arguments))


def read_rows(split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if split is None or r["split"] == split]


def score(rows: list[dict], preds: dict[str, tuple[str, dict]]) -> dict:
    """Routing quality for any router: predictions are {row id: (tool name, arguments)}.

    Model agnostic on purpose. The text stand-in in stress_test.py and Voxtral itself in
    eval_routing.py are scored by this same function, so their numbers are comparable.

    Accuracy alone hides the two failures that matter in a voice product. Firing a tool
    when the user only wanted to talk is a false trigger, and staying silent when they
    asked for something is a missed call. They trade against each other, so both are
    reported next to the accuracy rather than folded into it.
    """
    scored = [(r, preds[r["id"]]) for r in rows if r["id"] in preds]
    gold_tool = [(r, p) for r, p in scored if r["tool"] != NO_TOOL]
    gold_none = [(r, p) for r, p in scored if r["tool"] == NO_TOOL]
    correct = [(r, p) for r, p in scored if p[0] == r["tool"]]
    confusions = Counter(f"{r['tool']} -> {p[0]}" for r, p in scored if p[0] != r["tool"])
    return {
        "n": len(scored),
        "tool_accuracy": len(correct) / max(len(scored), 1),
        "false_trigger_rate": sum(p[0] != NO_TOOL for _, p in gold_none) / max(len(gold_none), 1),
        "missed_call_rate": sum(p[0] == NO_TOOL for _, p in gold_tool) / max(len(gold_tool), 1),
        "arg_keys_match": sum(set(p[1]) == set(r["arguments"]) for r, p in correct) / max(len(correct), 1),
        "top_confusions": confusions.most_common(8),
    }
