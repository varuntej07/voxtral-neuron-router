"""Snapshot the Aura voice agent's tool schemas into data/tools.json.

Reads the live registries from a local Aura checkout so the label space
matches production exactly. Run once; the demo never imports Aura again.

    python scripts/export_tools.py --aura-backend ../../../../mobileapps/Aura/backend
"""

import argparse
import json
import sys
from pathlib import Path

# speak_only is a speech channel for forced-tool turns, not a user intent.
EXCLUDED = {"speak_only"}


def normalize(name: str, description: str, parameters: dict | None) -> dict:
    return {
        "name": name,
        "description": " ".join(description.split()),
        "parameters": parameters or {"type": "object", "properties": {}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aura-backend", required=True, type=Path)
    parser.add_argument("--out", default=Path(__file__).parent.parent / "data" / "tools.json", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.aura_backend.resolve()))
    import livekit.agents.llm as lk
    import livekit.agents.llm.utils as lk_utils
    from src.agent.buddy_agent import BuddyAgent
    from src.agent.voice.capabilities import VOICE_TOOL_REGISTRY
    from src.shared.tools import TOOL_DEFINITIONS

    voice_names = set(VOICE_TOOL_REGISTRY) - EXCLUDED
    tools: dict[str, dict] = {}

    for spec in TOOL_DEFINITIONS:
        if spec["name"] in voice_names:
            # shared/tools.py uses MCP's camelCase key; an unmatched key silently exports an empty schema.
            params = spec.get("inputSchema") or spec.get("parameters") or spec.get("input_schema")
            tools[spec["name"]] = normalize(spec["name"], spec.get("description", ""), params)

    # Agent methods need a bound instance so `self` drops out of the schema.
    agent = object.__new__(BuddyAgent)
    for tool in lk.find_function_tools(agent):
        if lk.is_raw_function_tool(tool):
            raw = tool.info.raw_schema
            name, desc, params = raw["name"], raw.get("description", ""), raw.get("parameters")
        else:
            fn = lk_utils.build_legacy_openai_schema(tool)["function"]
            name, desc, params = fn["name"], fn.get("description", ""), fn.get("parameters")
        if name in voice_names and name not in tools:
            tools[name] = normalize(name, desc, params)

    missing = sorted(voice_names - set(tools))
    if missing:
        raise SystemExit(f"no schema found for: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ordered = [tools[n] for n in sorted(tools)]
    args.out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"wrote {len(ordered)} tools to {args.out}")


if __name__ == "__main__":
    main()
