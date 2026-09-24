"""Run Voxtral's frozen audio half once, off the training loop, and cache what it produced.

The audio tower and the projector are frozen for the whole fine-tune, so their output is a
function of the clip alone. Computing it inside every training step buys nothing and costs
two things that matter here.

1. Correctness. The Voxtral conv frontend is the one component with an open numeric bug on
   Neuron: at zero transformer depth it still disagrees with the CPU reference at cosine
   0.0975, and the compiler reports PASS both times (see encoder-compiles-but-output-is-wrong
   and depth-0-reproduces-the-whole-encoder-bug in lessons.txt). An encoder inside the
   training graph would feed that into every gradient, the loss would fall anyway, and
   nothing would say so. Precomputing takes it out of the graph entirely.
2. Time. Encoder plus projector is about 2.3 TFLOP per row. On a GPU that is well under a
   second; on the 8 vCPU of a trn1.2xlarge it is tens of seconds, which is why this is meant
   to run somewhere else and be uploaded.

The cache is exact rather than approximate, and that is not an accident of eval mode:
Voxtral hardcodes encoder dropout, layerdrop and activation_dropout to 0.0 and the forward
applies F.dropout(p=0.0), a no-op. A cached embedding is bit-identical to the one the loop
would have computed at the same dtype, model.train() included.

    # on a GPU box, ahead of the trn1 session
    python -u scripts/precompute_audio_embeds.py --split train --device cuda --self-check
    python -u scripts/precompute_audio_embeds.py --split val --device cuda

    # on the trn1, five rows, to prove the uploaded file came from the same function
    python -u scripts/precompute_audio_embeds.py --split val --limit 5 --device cpu \
        --out data/cache/val.probe.npy --results results/precompute_probe.json

Two things about the file it writes.

Storage is bfloat16, held as its uint16 bit pattern because numpy has no bfloat16. The base
model trains in bf16 and the splice narrows to the text embedding dtype anyway, so fp32
storage would be sixteen bits per element discarded on arrival, for twice the disk and twice
the host to device copy. It also means the CPU reference leg and the Neuron leg read the same
bytes, which removes a confound that exists today. The sidecar names the dtype explicitly so
a reader branches on metadata, never on the file.

The rows are bigger than the mels they come from: 375 x 3072 is 2.20 MiB against 1.46 MiB for
a 128 x 3000 fp32 mel. Four times fewer positions, twenty-four times wider. The full train
split is 10.5 GiB, which is what sizes the upload.
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def token_cache_fingerprint(split: str) -> str:
    """Hash of the input_ids the embeds have to line up with.

    Row i of the embeds is only meaningful next to row i of the tokens. Rebuild the token
    cache at a different --limit and that pairing silently breaks, which would train the
    model on somebody else's audio. Hashing a few MB of int32 costs milliseconds and turns a
    silent mismatch into a refusal to start.
    """
    ids = np.load(CACHE_DIR / f"{split}.npz")["input_ids"]
    return hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()


def write_atomic(path: Path, payload: dict) -> None:
    """Temp file then os.replace, so an interrupt mid-write cannot truncate the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def package_versions() -> dict:
    import importlib

    found = {}
    for name in ("torch", "transformers", "numpy"):
        try:
            found[name] = getattr(importlib.import_module(name), "__version__", "present")
        except ImportError:
            found[name] = None
    found["python"] = sys.version.split()[0]
    return found


def audio_embeds(model, mels):
    """(B, 128, 3000) mel to (B, 375, 3072) embeddings.

    get_audio_embeds returns a 2D tensor, not 3D. It reshapes the tower output to
    (-1, audio_config.intermediate_size), and intermediate_size is 5120, which is four times
    d_model 1280: the four-frames-per-token packing is done as a reshape that collapses the
    batch dimension along with it. So the call gives back (B*375, 3072) and the batch has to
    be put back by hand, which is what train_lora.py does at its own call site.
    """
    import torch

    with torch.no_grad():
        flat = model.get_audio_embeds(mels)
    if flat.dim() != 2 or flat.shape[0] % mels.shape[0]:
        raise SystemExit(f"get_audio_embeds returned {tuple(flat.shape)} for a batch of "
                         f"{mels.shape[0]}; this script assumes a flat (B*width, hidden)")
    return flat.reshape(mels.shape[0], flat.shape[0] // mels.shape[0], flat.shape[1])


def self_check(model, mels) -> dict:
    """Is the frozen half deterministic, and is it independent of position in the batch?

    About thirty seconds. The Neuron frontend is currently wrong for reasons nobody has
    explained, so proving the CPU one is neither nondeterministic nor batch dependent is
    worth the time before several GB are written on the strength of it.
    """
    import torch

    first = audio_embeds(model, mels[:1])
    again = audio_embeds(model, mels[:1])
    deterministic = bool(torch.equal(first, again))

    batched = audio_embeds(model, mels)[:1]
    a, b = first.float().flatten(), batched.float().flatten()
    cosine = float(torch.dot(a, b) / (a.norm() * b.norm()))
    return {"deterministic": deterministic, "batch_invariant_cosine": cosine,
            "batch_size": int(mels.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="first N rows; must match the --limit the token cache was built with")
    parser.add_argument("--load-dtype", default="bfloat16", choices=["bfloat16", "float32"],
                        help="what the encoder runs in. A CPU without AMX is often no faster in "
                             "bf16, but fp32 weights are about 19 GB resident")
    parser.add_argument("--store-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--checkpoint-every", type=int, default=32, help="rows between sidecar writes")
    parser.add_argument("--fresh", action="store_true", help="ignore progress and start over")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out", default="", help="override the output .npy path")
    parser.add_argument("--results", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    import torch
    from reference_forward import load_model

    out_path = Path(args.out) if args.out else CACHE_DIR / f"{args.split}.audio_embeds.npy"
    side_path = out_path.with_suffix(".meta.json")
    results_path = Path(args.results) if args.results else \
        ROOT / "results" / f"precompute_audio_embeds_{args.split}.json"

    meta = json.loads((CACHE_DIR / f"{args.split}.meta.json").read_text(encoding="utf-8"))
    width = meta["audio_span_width"]
    features_path = CACHE_DIR / f"{args.split}.features.npy"
    if not features_path.exists():
        raise SystemExit(f"{features_path} is missing; run prepare_sft_cache.py --split {args.split}")
    mels = np.load(features_path, mmap_mode="r")
    rows = len(mels) if not args.limit else min(args.limit, len(mels))

    fingerprint = token_cache_fingerprint(args.split)
    source = {"file": features_path.name, "bytes": features_path.stat().st_size,
              "mtime": features_path.stat().st_mtime}
    numpy_dtype = "uint16" if args.store_dtype == "bfloat16" else "float32"

    # Resume, but only onto a file written from the same inputs in the same layout. Anything
    # else is a different array wearing the right filename.
    done = 0
    arr = None
    hidden = None
    if side_path.exists() and out_path.exists() and not args.fresh:
        old = json.loads(side_path.read_text(encoding="utf-8"))
        mismatch = [k for k, v in (("rows", rows), ("numpy_dtype", numpy_dtype),
                                   ("ids_sha256", fingerprint)) if old.get(k) != v]
        if old.get("source", {}).get("file") != source["file"]:
            mismatch.append("source")
        if mismatch:
            raise SystemExit(f"{side_path.name} disagrees on {', '.join(mismatch)}. Pass --fresh "
                             f"to rebuild; do not resume onto a file built from other inputs.")
        done = int(old.get("rows_done", 0))
        if done:
            arr = np.lib.format.open_memmap(out_path, mode="r+")
            hidden = arr.shape[2]
            print(f"resuming at row {done} of {rows}", flush=True)

    print(f"loading {args.model} on {args.device} in {args.load_dtype}", flush=True)
    model = load_model(Path(args.model).expanduser(), getattr(torch, args.load_dtype), "eager")
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)
    device = torch.device(args.device)
    # Only the frozen half is ever called, so only the frozen half needs to be resident.
    model.audio_tower.to(device)
    model.multi_modal_projector.to(device)

    started = time.time()
    record = {
        "split": args.split, "rows": rows, "rows_done": done, "complete": False,
        "device": args.device, "load_dtype": args.load_dtype, "store_dtype": args.store_dtype,
        "numpy_dtype": numpy_dtype, "batch_size": args.batch_size, "width": width,
        "source": source, "ids_sha256": fingerprint, "model_dir": str(args.model),
        "out": str(out_path), "versions": package_versions(), "notes": args.notes,
        "instance_type": os.environ.get("NEURON_INSTANCE_TYPE"),
        "read_with": "torch.from_numpy(x).view(torch.bfloat16)" if numpy_dtype == "uint16" else None,
    }
    if hidden:
        record["shape"] = [width, hidden]

    def bank(complete: bool = False) -> None:
        record["rows_done"] = done
        record["complete"] = complete
        record["seconds"] = round(time.time() - started, 1)
        if done:
            record["seconds_per_row"] = round(record["seconds"] / done, 3)
            record["rows_per_hour"] = round(3600 * done / max(record["seconds"], 1e-9), 1)
        # Flush the data first. The sidecar is the commit record and must never claim more
        # rows than the array on disk actually holds.
        if arr is not None:
            arr.flush()
        write_atomic(side_path, record)
        write_atomic(results_path, record)

    def on_signal(signum, _frame):
        print(f"\nsignal {signum}, banking {done} rows", flush=True)
        bank()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    bank()  # before anything slow, so a crash at minute one still leaves evidence

    if args.self_check:
        probe = torch.from_numpy(np.array(mels[: max(args.batch_size, 2)])).to(
            device, getattr(torch, args.load_dtype))
        record["self_check"] = self_check(model, probe)
        print(json.dumps(record["self_check"]), flush=True)
        bank()
        if not record["self_check"]["deterministic"]:
            raise SystemExit("the frozen half is not deterministic; do not cache it")

    try:
        while done < rows:
            stop = min(done + args.batch_size, rows)
            batch = torch.from_numpy(np.array(mels[done:stop])).to(
                device, getattr(torch, args.load_dtype))
            out = audio_embeds(model, batch)
            if out.shape[1] != width:
                raise SystemExit(f"got {out.shape[1]} audio tokens, the cache meta says {width}")

            if arr is None:
                hidden = out.shape[2]
                arr = np.lib.format.open_memmap(out_path, mode="w+",
                                                dtype=np.dtype(numpy_dtype),
                                                shape=(rows, width, hidden))
                record["shape"] = [width, hidden]
                gib = rows * width * hidden * arr.dtype.itemsize / 2**30
                print(f"writing {out_path.name}, {rows} x {width} x {hidden} "
                      f"{args.store_dtype}, {gib:.2f} GiB", flush=True)
            if out.shape[2] != hidden:
                raise SystemExit(f"row width changed from {hidden} to {out.shape[2]}")

            host = out.to("cpu")
            if numpy_dtype == "uint16":
                arr[done:stop] = host.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            else:
                arr[done:stop] = host.float().numpy()
            done = stop

            rate = (time.time() - started) / done
            print(f"{done}/{rows} rows  {rate:.2f}s/row  "
                  f"{(rows - done) * rate / 60:.0f} min left", flush=True)
            if done % args.checkpoint_every < args.batch_size:
                bank()
    except BaseException:
        bank()
        raise

    bank(complete=True)
    print(f"done, {done} rows in {record['seconds'] / 60:.1f} min "
          f"({record['seconds_per_row']:.2f}s/row)", flush=True)
    print(f"results written to {results_path}", flush=True)


if __name__ == "__main__":
    main()
