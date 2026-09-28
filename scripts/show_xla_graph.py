"""Show how a piece of Voxtral turns from PyTorch into an XLA graph.

This is the step torch_neuronx.trace does before the Neuron compiler ever runs:

    PyTorch module  ->  ATen ops (torch.export)  ->  XLA HLO  ->  neuronx-cc  ->  NEFF on the chip
                        stage 1                     stage 2       only on inf2

The example is Voxtral's audio conv stem, with its real weights read straight from the
checkpoint: conv1, gelu, conv2 (stride 2), gelu, permute, add positions. It is small enough
to read the whole graph, and it is exactly the part that gives wrong numbers on Neuron.

Stages 1 and 2 run on any Linux box with CPU torch_xla. Stage 3 checks that the XLA graph,
run on XLA's own CPU backend, gives the same numbers as plain PyTorch. If it does, the
lowering from PyTorch to HLO is correct and the Neuron bug lives below HLO, in neuronx-cc.

    PJRT_DEVICE=CPU python scripts/show_xla_graph.py --weights ~/models/voxtral-mini-3b

On inf2 the same script runs the graph on a NeuronCore instead:

    source ~/venv-trace/bin/activate
    PJRT_DEVICE=NEURON python scripts/show_xla_graph.py --weights ~/models/voxtral-mini-3b

Writes results/xla_graph/: stem.aten.txt, stem.<device>.hlo.txt, stem.<device>.summary.json.
"""

import argparse
import collections
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "xla_graph"

MEL_BINS, MEL_FRAMES, WIDTH, POSITIONS = 128, 3000, 1280, 1500


class ConvStem(torch.nn.Module):
    """VoxtralEncoder.forward up to the first transformer layer, op for op."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(MEL_BINS, WIDTH, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(WIDTH, WIDTH, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positions", torch.zeros(POSITIONS, WIDTH))

    def forward(self, mel):
        h = torch.nn.functional.gelu(self.conv1(mel))
        h = torch.nn.functional.gelu(self.conv2(h))
        h = h.permute(0, 2, 1)
        return h + self.positions


def load_real_weights(stem: ConvStem, model_dir: Path) -> str:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = {
        "conv1.weight": "audio_tower.conv1.weight", "conv1.bias": "audio_tower.conv1.bias",
        "conv2.weight": "audio_tower.conv2.weight", "conv2.bias": "audio_tower.conv2.bias",
        "positions": "audio_tower.embed_positions.weight",
    }
    state = {}
    for ours, theirs in names.items():
        with safe_open(model_dir / index[theirs], framework="pt") as f:
            state[ours] = f.get_tensor(theirs).float()
    stem.load_state_dict(state)
    return "real Voxtral weights"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="", help="Voxtral snapshot dir; omit for random weights")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    stem = ConvStem().eval()
    source = load_real_weights(stem, Path(args.weights).expanduser()) if args.weights else "random weights"
    mel = torch.randn(1, MEL_BINS, MEL_FRAMES)  # same shape as a real 30 s mel

    # Stage 1: the PyTorch graph, as ATen ops. This is what PyTorch itself thinks it runs.
    exported = torch.export.export(stem, (mel,))
    aten = str(exported.graph_module.graph)
    (OUT / "stem.aten.txt").write_text(aten)
    aten_ops = [line.split("torch.ops.aten.")[1].split("(")[0].rstrip("]")
                for line in aten.splitlines() if "torch.ops.aten." in line]

    # Stage 2: the XLA graph. Moving the module and input to the XLA device makes every op
    # record into a lazy graph instead of running; asking for the HLO text of the output
    # prints that graph. torch_neuronx.trace does the same thing and hands this HLO to
    # neuronx-cc.
    import torch_xla
    import torch_xla.core.xla_model as xm

    # PJRT_DEVICE picks the backend: CPU on a laptop, NEURON on an inf2 box. The graph is
    # the same; stage 3 then says whether that backend computes it correctly.
    device_name = os.environ.get("PJRT_DEVICE", "CPU")
    tag = device_name.lower()
    device = xm.xla_device()
    stem_x, mel_x = stem.to(device), mel.to(device)
    with torch.no_grad():
        out_x = stem_x(mel_x)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([out_x])
    (OUT / f"stem.{tag}.hlo.txt").write_text(hlo)
    hlo_ops = collections.Counter(
        m.group(1) for m in re.finditer(r"=\s*\S+\s+([a-z\-]+)\(", hlo)
    )

    # Stage 3: does the XLA graph compute what PyTorch computes? Same weights, same input.
    started = time.perf_counter()
    got = out_x.cpu().numpy()
    xla_seconds = time.perf_counter() - started
    with torch.no_grad():
        want = stem.cpu()(mel).numpy()
    summary = {
        "weights": source,
        "input_shape": list(mel.shape),
        "output_shape": list(want.shape),
        "aten_ops_in_order": aten_ops,
        "hlo_op_counts": dict(hlo_ops.most_common()),
        "hlo_lines": len(hlo.splitlines()),
        "xla_cpu_vs_pytorch_cosine": round(cosine(got, want), 8),
        "xla_cpu_vs_pytorch_max_abs_diff": float(np.abs(got - want).max()),
        "xla_compile_and_run_seconds": round(xla_seconds, 2),
        "xla_device": device_name,
        "torch": torch.__version__,
        "torch_xla": torch_xla.__version__,
    }
    (OUT / f"stem.{tag}.summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== stage 1: PyTorch as ATen ops ({source}) ===")
    print("  " + "  ->  ".join(aten_ops))
    print(f"\n=== stage 2: XLA HLO, {summary['hlo_lines']} lines, op counts ===")
    for op, n in hlo_ops.most_common():
        print(f"  {op:22s} {n}")
    print(f"\n=== stage 3: XLA on {device_name} vs plain PyTorch on CPU ===")
    print(f"  cosine {summary['xla_cpu_vs_pytorch_cosine']}, max abs diff {summary['xla_cpu_vs_pytorch_max_abs_diff']:.2e}")
    print(f"\nfull graphs in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
