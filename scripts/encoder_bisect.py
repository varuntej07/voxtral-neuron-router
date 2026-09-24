"""Find where the Neuron encoder output stops matching, when neuron_forward_smoke.py says it does not.

Two questions, in the order that makes each answer cheap:

  1. Does plain CPU PyTorch in this venv reproduce results/reference_forward.npz? If not,
     the reference was produced by different code (another transformers version, another
     venv) and the Neuron number is measuring that drift, not the hardware. No compile
     needed, about a minute.
  2. If CPU matches, at what depth does Neuron diverge? The tower is cut to its first k
     layers using its own forward (conv frontend, positions, k layers, final norm) and each
     cut is traced and compared against CPU on the same cut. k=0 isolates the conv frontend.

    source scripts/neuron_env_inf2.sh
    python -u scripts/encoder_bisect.py --cpu-only                       # question 1
    python -u scripts/encoder_bisect.py --layers 0,1,full                # question 2
    python -u scripts/encoder_bisect.py --layers full --with-projector   # the exact smoke graph

The depth list has to reach full depth. A sweep of 0,1,8 that comes back clean says only
that depths 0, 1 and 8 agree; it never traces the graph that produced the mismatch, so it
cannot explain it. Pass 'full' and the real depth is resolved at runtime. The script warns
when the list stops short.

--with-projector matters for the same reason. Without it this script traces the tower alone
and returns last_hidden_state, while neuron_forward_smoke.py traces the tower plus the
reshape and projector. Those are different graphs, so a clean sweep here next to a dirty
smoke run leaves a hole rather than an answer.

Writes results/encoder_bisect.json.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from neuron_forward_smoke import compare
from reference_forward import load_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--layers", default="0,1,full",
                        help="encoder depths to trace and compare. 'full' resolves to the "
                             "tower's real depth at runtime. The list must reach full depth "
                             "or the sweep cannot reproduce the mismatch it exists to explain")
    parser.add_argument("--with-projector", action="store_true",
                        help="include the reshape and projector, so the deepest cut is the "
                             "same graph neuron_forward_smoke.py traces. Without this the "
                             "sweep can come back clean while the smoke run stays dirty, "
                             "because they are not tracing the same thing")
    parser.add_argument("--cpu-only", action="store_true", help="only check CPU against the reference npz")
    parser.add_argument("--compiler-args", default="--target=inf2 --model-type=transformer --auto-cast=none")
    args = parser.parse_args()

    import torch

    dtype = getattr(torch, args.dtype)
    reference = dict(np.load(RESULTS / "reference_forward.npz"))
    ref_facts = json.loads((RESULTS / "reference_forward.json").read_text(encoding="utf-8"))
    features = torch.from_numpy(reference["input_features"]).to(dtype)

    import transformers

    record = {
        "dtype": args.dtype,
        "reference_dtype": ref_facts.get("dtype"),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "compiler_args": args.compiler_args,
    }

    model = load_model(Path(args.model).expanduser(), dtype, args.attn)
    tower, projector = model.audio_tower, model.multi_modal_projector
    intermediate = model.config.audio_config.intermediate_size
    del model.language_model  # the decoder is 3B of weights this script never touches
    gc.collect()

    # Question 1: same code path as reference_forward.py, run here.
    with torch.no_grad():
        hidden = tower(features).last_hidden_state
        embeds = projector(hidden.reshape(-1, intermediate))
    record["cpu_vs_reference"] = [
        compare("encoder_hidden", hidden.float().numpy(), reference["encoder_hidden"]),
        compare("audio_embeds", embeds.float().numpy(), reference["audio_embeds"]),
    ]
    print(json.dumps(record["cpu_vs_reference"], indent=2), flush=True)
    write(record)
    if args.cpu_only:
        return

    import torch_neuronx

    class Tower(torch.nn.Module):
        def __init__(self, tower, projector=None):
            super().__init__()
            self.tower = tower
            self.projector = projector

        def forward(self, input_features):
            hidden = self.tower(input_features).last_hidden_state
            if self.projector is None:
                return hidden
            # Truncating depth does not change the [1, 1500, 1280] output shape, so the
            # reshape still packs cleanly at any depth and a cut graph stays a genuine
            # prefix of the production one.
            return self.projector(hidden.reshape(-1, intermediate))

    all_layers = tower.layers
    full_depth = len(all_layers)
    record["full_depth"] = full_depth
    record["with_projector"] = args.with_projector

    depths = []
    for token in args.layers.split(","):
        token = token.strip()
        depths.append(full_depth if token in ("full", "all") else int(token))
    if max(depths) < full_depth:
        # A sweep that stops short can only ever say "the depths I tried were clean", which
        # is not evidence about the depth that actually failed.
        warning = (
            f"--layers reaches {max(depths)} but the tower is {full_depth} deep, so this "
            f"sweep cannot reproduce the full encoder mismatch. Add {full_depth} or 'full'."
        )
        record["warning"] = warning
        print(f"WARNING: {warning}", flush=True)
    record["depths"] = depths
    write(record)

    record["neuron_vs_cpu_by_depth"] = []
    for depth in depths:
        tower.layers = torch.nn.ModuleList(list(all_layers)[:depth])
        graph = Tower(tower, projector if args.with_projector else None).eval()
        with torch.no_grad():
            want = graph(features).float().numpy()
        started = time.perf_counter()
        traced = torch_neuronx.trace(graph, (features,), compiler_args=args.compiler_args.split())
        got = traced(features).float().numpy()
        tensor = "audio_embeds" if args.with_projector else "encoder_hidden"
        entry = compare(f"{tensor}_depth_{depth}", got, want)
        entry["depth"] = depth
        entry["is_full_depth"] = depth == full_depth
        entry["compile_seconds"] = round(time.perf_counter() - started, 1)
        record["neuron_vs_cpu_by_depth"].append(entry)
        print(json.dumps(entry, indent=2), flush=True)
        write(record)  # after every depth, so a later failure keeps the earlier ones
        del traced
        gc.collect()
    tower.layers = all_layers


def write(record: dict) -> None:
    (RESULTS / "encoder_bisect.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
