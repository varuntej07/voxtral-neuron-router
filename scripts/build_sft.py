"""Turn data/text/rows.jsonl into LoRA-ready chat pairs.

Writes data/sft/{train,val}.jsonl, one conversation per line:

  {"id", "tool", "audio": "data/audio/<id>.wav",
   "messages": [system, user (utterance text), assistant (route JSON)]}

The user turn is text for now. After TTS, the training loader swaps it for the
clip at `audio`, keeping the same id, so the pairs never have to be rebuilt.
The fine-tuned model learns the tools, so its system prompt is small on purpose:
prompt tokens are prefill, and prefill is the latency floor on Inferentia2.

    python scripts/build_sft.py --prompt names
"""

import argparse
import json
from collections import Counter

from tooling import ROOT, compact_catalog, names_catalog, read_rows

SFT_DIR = ROOT / "data" / "sft"
BASE_RULES = "Route the user's spoken turn. Reply with JSON only: {\"tool\": <name or no_tool>, \"arguments\": {...}}."


def system_prompt(kind: str) -> str:
    if kind == "none":
        return BASE_RULES
    catalog = names_catalog() if kind == "names" else compact_catalog()
    return f"{BASE_RULES}\nTools:\n{catalog}"


def to_pair(row: dict, system: str) -> dict:
    target = json.dumps({"tool": row["tool"], "arguments": row["arguments"]}, separators=(",", ":"))
    return {
        "id": row["id"],
        "tool": row["tool"],
        "audio": f"data/audio/{row['id']}.wav",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": row["utterance"]},
            {"role": "assistant", "content": target},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=["names", "compact", "none"], default="names",
                        help="set from the stress test result; names is ~260 tokens")
    args = parser.parse_args()

    system = system_prompt(args.prompt)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = read_rows(split)
        path = SFT_DIR / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(to_pair(r, system)) for r in rows) + "\n", encoding="utf-8")
        counts = Counter(r["tool"] for r in rows)
        print(f"{split}: {len(rows)} pairs, {len(counts)} labels, thinnest {counts.most_common()[-3:]} -> {path}")
    print(f"system prompt: ~{len(system) // 4} tokens ({args.prompt})")


if __name__ == "__main__":
    main()
