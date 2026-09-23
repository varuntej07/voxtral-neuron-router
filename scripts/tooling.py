"""Shared helpers: paths, the frozen tool snapshot, and prompt renderings of it."""

import json
import os
import re
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
