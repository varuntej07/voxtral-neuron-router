"""Does the fine-tuned model actually route better than base Voxtral? Audio in, tool out.

This is the number the project is judged on, and nothing else in the repo produces it.
stress_test.py routes with a text LLM standing in for the router, which sizes the prompt
but never touches Voxtral or audio. This runs the real model on the real clips.

One script covers every comparison, so the numbers are comparable by construction:

    # base, before training, on the TTS val split. Run this FIRST or there is no baseline.
    python scripts/eval_routing.py --results results/eval_base_tts.json

    # fine-tuned, same clips
    python scripts/eval_routing.py --adapter checkpoints/lora --results results/eval_lora_tts.json

    # the criterion that counts: human speech, recorded by scripts/record_real_speech.py
    python scripts/eval_routing.py --audio-dir data/audio_real --results results/eval_base_real.json
    python scripts/eval_routing.py --audio-dir data/audio_real --adapter checkpoints/lora \
        --results results/eval_lora_real.json

Gold labels come from data/text/rows.jsonl, which carries `tool` and `arguments` per id.
Scoring is tooling.score, the same function stress_test.py uses, so a Voxtral number and a
text-LLM number can sit in the same table without an asterisk.

Three things worth knowing about what is measured:

  * Accuracy alone is not the product. Firing a tool when the user only wanted to talk is
    a false trigger; staying silent when they asked for something is a missed call. Both
    come back separately, because they trade against each other.
  * A router that emits unparseable JSON has failed even when the tool name inside it is
    right, so json_valid_rate is reported next to accuracy rather than hidden by a lenient
    parser. The parse here is deliberately strict: one json.loads, no regex repair.
  * Predictions stream to <results>.pred.jsonl as they are produced, and a rerun skips ids
    already in it. A 665 row CPU pass takes hours, and losing it to one exception is not
    something to find out at the end.
"""

import argparse
import json
import time
from pathlib import Path

# The loader lives in the reference so the eval and the parity run load weights the same way.
from reference_forward import catalog_text, load_model
from tooling import args_valid, label_space, read_rows

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parse_prediction(text: str) -> tuple[str, dict]:
    """Strict: the model was trained to emit exactly one JSON object and nothing else.

    Repairing malformed output here would inflate the score by hiding a real failure mode,
    so anything that does not parse is <unparseable> and counts against accuracy.
    """
    try:
        out = json.loads(text.strip())
        tool = out["tool"]
        arguments = out.get("arguments") or {}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return "<unparseable>", {}
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return "<unparseable>", {}
    return tool, arguments


def build_processor(model_dir: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_dir)


def prompt_inputs(processor, catalog: str, wav: Path):
    """Catalog first, audio second: the exact order prepare_sft_cache.py trains on.

    Tekken v7 rejects a system role beside audio, so the tool catalog rides in the user
    turn as a text chunk. Text first also keeps the static prefix cacheable at prefill.
    """
    conversation = [{
        "role": "user",
        "content": [{"type": "text", "text": catalog}, {"type": "audio", "path": str(wav)}],
    }]
    return processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True, return_tensors="pt"
    )


def load_done(path: Path) -> dict:
    """Predictions already produced, so an interrupted pass resumes instead of restarting."""
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done[rec["id"]] = (rec["tool"], rec["arguments"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--adapter", default="", help="LoRA directory; omit to score base Voxtral")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xla"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--audio-dir", default="data/audio",
                        help="data/audio_real for the human recorded set")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="targets are median 18 and p95 49, see results/prompt_shape.json. The "
                             "cap matters most for the base model, which was never trained to emit "
                             "one JSON object and stop, so it runs to the cap and pays for every "
                             "token of it. Raise to 128 to cover the 113 token tail.")
    parser.add_argument("--results", default="results/eval_routing.json")
    parser.add_argument("--fresh", action="store_true", help="ignore existing predictions and redo")
    args = parser.parse_args()

    import torch

    audio_dir = ROOT / args.audio_dir
    rows = [r for r in read_rows(args.split) if (audio_dir / f"{r['id']}.wav").exists()]
    skipped = len(read_rows(args.split)) - len(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no clips for split {args.split} in {audio_dir}")

    results_path = Path(args.results)
    pred_path = results_path.with_suffix(".pred.jsonl")
    if args.fresh and pred_path.exists():
        pred_path.unlink()
    preds = load_done(pred_path)
    todo = [r for r in rows if r["id"] not in preds]
    print(f"{len(rows)} rows with audio ({skipped} skipped, no clip), {len(preds)} already done, "
          f"{len(todo)} to run", flush=True)

    model_dir = Path(args.model).expanduser()
    dtype = getattr(torch, args.dtype)
    processor = build_processor(model_dir)
    catalog = catalog_text()

    model = load_model(model_dir, dtype, "eager")
    adapter_check = None
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(Path(args.adapter).expanduser()))
        model = model.eval()
        # PEFT initialises lora_B to zeros, so an adapter that failed to attach, or attached
        # and was never trained, is exactly a no-op: the run would score the base model while
        # claiming to score the tuned one, and present as "fine-tuning did not help". That is
        # the one conclusion that must not be reached by accident, so prove otherwise here
        # rather than inferring it from the accuracy afterwards.
        b_tensors = [p for n, p in model.named_parameters() if "lora_B" in n]
        nonzero = sum(1 for p in b_tensors if p.abs().sum().item() > 0)
        adapter_check = {"lora_B_tensors": len(b_tensors), "nonzero": nonzero}
        if not b_tensors:
            raise SystemExit(f"{args.adapter} attached no LoRA tensors; nothing would be measured")
        if not nonzero:
            raise SystemExit(f"every lora_B in {args.adapter} is zero, so the adapter is a no-op "
                             f"and this run would silently re-score the base model")
        print(f"adapter loaded: {nonzero} of {len(b_tensors)} lora_B tensors are nonzero", flush=True)
    if args.device == "xla":
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    else:
        device = torch.device(args.device)
    model.to(device)

    started = time.time()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(todo, 1):
            inputs = prompt_inputs(processor, catalog, audio_dir / f"{row['id']}.wav")
            # Mel features follow the model dtype; ids stay integer. Anything that is not a
            # tensor is passed through untouched.
            inputs = {
                k: (v if not isinstance(v, torch.Tensor)
                    else v.to(device, dtype) if v.is_floating_point() else v.to(device))
                for k, v in inputs.items()
            }
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False)
            new = generated[0].tolist()[int(inputs["input_ids"].shape[1]):]
            text = processor.tokenizer.decode(new, skip_special_tokens=True)
            tool, arguments = parse_prediction(text)
            preds[row["id"]] = (tool, arguments)
            sink.write(json.dumps({"id": row["id"], "tool": tool, "arguments": arguments,
                                   "raw": text[:400]}) + "\n")
            sink.flush()
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - started) / i
                print(f"{i}/{len(todo)} {rate:.1f}s/row, {rate * (len(todo) - i) / 60:.0f} min left",
                      flush=True)

    from tooling import score

    report = score(rows, preds)
    considered = [preds[r["id"]] for r in rows if r["id"] in preds]
    report["json_valid_rate"] = sum(t != "<unparseable>" for t, _ in considered) / max(len(considered), 1)
    # A name the catalog does not contain is a different failure from a wrong-but-real tool.
    known = set(label_space())
    report["hallucinated_tool_rate"] = sum(
        t not in known and t != "<unparseable>" for t, _ in considered) / max(len(considered), 1)
    report["args_schema_valid_rate"] = sum(
        t in known and args_valid(t, a) for t, a in considered) / max(len(considered), 1)

    record = {
        "model": str(model_dir),
        "adapter": args.adapter or None,
        "is_base": not args.adapter,
        "adapter_check": adapter_check,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "split": args.split,
        "audio_dir": args.audio_dir,
        "audio_is_real_speech": "real" in args.audio_dir,
        "rows_with_audio": len(rows),
        "rows_missing_audio": skipped,
        "seconds_per_row": (time.time() - started) / max(len(todo), 1),
        "scores": report,
    }
    results_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
