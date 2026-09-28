"""Find which op in the Voxtral encoder's conv stem is wrong on Neuron.

encoder_bisect.py --layers 0 showed that the stem alone (conv1, conv2, positions, final
norm) already scores cosine 0.0975 against CPU while the compiler says PASS. This script
goes one level below that: it traces each op of the stem as its own graph, feeds each one
a known input, and reports where agreement breaks. Every graph compiles in seconds.

    source scripts/neuron_env_inf2.sh
    python -u scripts/conv_stem_probe.py --cpu-only      # laptop: checks the graphs on CPU only
    python -u scripts/conv_stem_probe.py                 # inf2: the real probe

Graphs, in the order the data flows:

    A      gelu(conv1(x))                      [1,128,3000]  -> [1,1280,3000]
    A_mm   same, conv1 written as 3 shifted matmuls, no conv op in the graph
    A_flip same as A, but the graph takes a frames-first input and permutes inside
    B      gelu(conv2(a))  fed CPU output of A [1,1280,3000] -> [1,1280,1500]
    B_mm   same, conv2 (stride 2) as 3 shifted matmuls
    C      A then B
    D      C, permute, + positions              -> [1,1500,1280]
    E      D + layer_norm    (the same ops as encoder_bisect --layers 0, written by hand)
    F      the HF VoxtralEncoder.forward at depth 0, exactly what encoder_bisect traces
    G      E plus the input .to(device=conv1.weight.device) line HF runs first
    H      E with positions read from the live nn.Embedding parameter, as HF does
    I      F with the @check_model_inputs decorator stripped
    J1..J3 the HF body copied by hand: + input move, + dropout, + both
    A_c,C_c A and C with gelu looked up at call time (the AwsNeuronGelu custom call)
    F_fix  F with gelu pinned to the pre-import function while tracing

Two inputs per graph: the real mel from results/reference_forward.npz, and an impulse mel
that is zero except one 1.0 at (bin, frame). conv1's bias makes every output frame nonzero,
so the active-frame readout cannot localize the impulse; the impulse is still a second,
independent parity check.

How to read the result:

    A fails, A_mm passes        conv1d lowering is wrong, and the matmul rewrite is the fix
    A fails, A_mm fails         something below conv, look at A_flip and the impulse readout
    A passes, B fails           stride 2 lowering; B_mm says whether the rewrite fixes it
    A, B pass, C fails          layout between the two convs
    A..D pass, E fails          layer_norm
    A..E pass, F fails          the HF forward wrapper, not an op; G says if it is the device move
    everything passes           the bug is in how the bisect wrapper cut the tower, not the ops

What it found on inf2.xlarge, neuronx-cc 2.26.6360.0: A through E, G, H and J1 to J3 pass
at cosine 0.99999 or better; F and I fail at 0.09748. Their HLO differs from J3's only in
GELU (the AwsNeuronGelu custom call), which decides whether the compiler's permute-then-add
weight layout bug fires. See repro_permute_add.py and neuron_patches.py.

Writes results/conv_stem_probe.json after every graph, so a crash keeps what ran.
"""

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from neuron_forward_smoke import compare
from reference_forward import load_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "conv_stem_probe.json"

PASS_COSINE = 0.999  # bf16 matmul noise sits around 0.9999; a structural bug sits below 0.5


def conv_as_matmul(x, weight, bias, stride):
    """conv1d with kernel 3 and padding 1, written as three shifted matmuls.

    out[:, :, t] = sum_k W[:, :, k] @ x[:, :, stride*t + k - 1] + b
    No conv op reaches the compiler, so if this graph agrees with CPU while the conv graph
    does not, the conv lowering is the bug and this is already a working replacement.
    """
    import torch

    t_out = (x.shape[-1] + 2 - 3) // stride + 1
    xp = torch.nn.functional.pad(x, (1, 1))
    out = None
    for k in range(3):
        xs = xp[:, :, k : k + stride * (t_out - 1) + 1 : stride]  # [1, Cin, t_out]
        y = torch.matmul(weight[:, :, k], xs)  # [Cout, Cin] @ [1, Cin, T] -> [1, Cout, T]
        out = y if out is None else out + y
    return out + bias[None, :, None]


def build_graphs(tower):
    import torch

    gelu = torch.nn.functional.gelu

    class Stem(torch.nn.Module):
        # Every graph owns the stem weights as registered submodules and a buffer.
        # torch_neuronx.trace moves only registered parameters and buffers to the XLA
        # device; a weight reached through a closure stays on CPU and the trace dies
        # with "Expected XLA tensor. Got: CPUBFloat16Type" before neuronx-cc runs.
        def __init__(self):
            super().__init__()
            self.conv1, self.conv2, self.norm = tower.conv1, tower.conv2, tower.layer_norm
            # kept in fp32 by transformers, see _keep_in_fp32_modules_strict
            self.register_buffer("pos", tower.embed_positions.weight.detach().clone())

    class A(Stem):
        def forward(self, x):
            return gelu(self.conv1(x))

    class A_mm(Stem):
        def forward(self, x):
            return gelu(conv_as_matmul(x, self.conv1.weight, self.conv1.bias, 1))

    class A_flip(Stem):
        # Takes [1, 3000, 128]. If this passes while A fails, the compiler mishandles the
        # channels-first layout it was handed and one permute is the fix.
        def forward(self, x):
            return gelu(self.conv1(x.permute(0, 2, 1)))

    class B(Stem):
        def forward(self, a):
            return gelu(self.conv2(a))

    class B_mm(Stem):
        def forward(self, a):
            return gelu(conv_as_matmul(a, self.conv2.weight, self.conv2.bias, 2))

    class C(Stem):
        def forward(self, x):
            return gelu(self.conv2(gelu(self.conv1(x))))

    class D(Stem):
        def forward(self, x):
            h = gelu(self.conv2(gelu(self.conv1(x)))).permute(0, 2, 1)
            return (h + self.pos).to(h.dtype)  # exactly what VoxtralEncoder.forward does

    class E(Stem):
        def forward(self, x):
            h = gelu(self.conv2(gelu(self.conv1(x)))).permute(0, 2, 1)
            return self.norm((h + self.pos).to(h.dtype))

    class F(torch.nn.Module):
        # The HF VoxtralEncoder.forward itself at depth 0, the way encoder_bisect.py traces
        # it. Same ops as E; if this fails and E passes, the wrapper is the bug, not an op.
        def __init__(self):
            super().__init__()
            self.tower = tower

        def forward(self, x):
            return self.tower(x).last_hidden_state

    class G(Stem):
        # E plus the one line HF runs first: move the input to wherever conv1's weight is.
        def forward(self, x):
            x = x.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
            h = gelu(self.conv2(gelu(self.conv1(x)))).permute(0, 2, 1)
            return self.norm((h + self.pos).to(h.dtype))

    class H(torch.nn.Module):
        # E, but positions come from the live nn.Embedding parameter as HF reads them,
        # not from a cloned buffer.
        def __init__(self):
            super().__init__()
            self.tower = tower

        def forward(self, x):
            t = self.tower
            h = gelu(t.conv2(gelu(t.conv1(x)))).permute(0, 2, 1)
            return t.layer_norm((h + t.embed_positions.weight).to(h.dtype))

    class I(F):
        # F with the @check_model_inputs decorator stripped: the bare HF forward body.
        def forward(self, x):
            return type(self.tower).forward.__wrapped__(self.tower, x).last_hidden_state

    class J(F):
        # The bare HF forward body copied line by line, so single lines can be switched on.
        # J1 adds only the input move, J2 only the dropout call, J3 is the whole body.
        def __init__(self, move_input, dropout):
            super().__init__()
            self.move_input, self.use_dropout = move_input, dropout

        def forward(self, x):
            t = self.tower
            if self.move_input:
                x = x.to(dtype=t.conv1.weight.dtype, device=t.conv1.weight.device)
            h = gelu(t.conv1(x))
            h = gelu(t.conv2(h))
            h = h.permute(0, 2, 1)
            h = (h + t.embed_positions.weight).to(h.dtype)
            if self.use_dropout:
                h = torch.nn.functional.dropout(h, p=t.dropout, training=t.training)
            for layer in t.layers:
                h = layer(h, attention_mask=None, layer_head_mask=None)[0]
            return t.layer_norm(h)

    class A_c(Stem):
        # A, but gelu looked up at call time like HF does. After import torch_neuronx that
        # lowers to the AwsNeuronGelu custom call instead of erf arithmetic.
        def forward(self, x):
            return torch.nn.functional.gelu(self.conv1(x))

    class C_c(Stem):
        def forward(self, x):
            f = torch.nn.functional.gelu
            return f(self.conv2(f(self.conv1(x))))

    class F_fix(F):
        # The HF forward, with torch.nn.functional.gelu pinned to the function bound before
        # torch_neuronx was imported, for the duration of the call. The candidate fix.
        def forward(self, x):
            patched = torch.nn.functional.gelu
            torch.nn.functional.gelu = gelu
            try:
                return self.tower(x).last_hidden_state
            finally:
                torch.nn.functional.gelu = patched

    # name -> (module, which input it takes)
    return {
        "A": (A(), "mel"),
        "A_mm": (A_mm(), "mel"),
        "A_flip": (A_flip(), "mel_t"),
        "B": (B(), "a_cpu"),
        "B_mm": (B_mm(), "a_cpu"),
        "C": (C(), "mel"),
        "D": (D(), "mel"),
        "E": (E(), "mel"),
        "F": (F(), "mel"),
        "G": (G(), "mel"),
        "H": (H(), "mel"),
        "I": (I(), "mel"),
        "J1": (J(True, False), "mel"),
        "J2": (J(False, True), "mel"),
        "J3": (J(True, True), "mel"),
        "A_c": (A_c(), "mel"),
        "C_c": (C_c(), "mel"),
        "F_fix": (F_fix(), "mel"),
    }


def channel_cosine(got: np.ndarray, want: np.ndarray, channel_axis: int) -> dict:
    """Cosine per channel. compare() already gives cosine per sequence position; a bug
    that permutes or drops channels shows up here and not there."""
    g = np.moveaxis(got.astype(np.float64)[0], channel_axis - 1, 0).reshape(got.shape[channel_axis], -1)
    w = np.moveaxis(want.astype(np.float64)[0], channel_axis - 1, 0).reshape(want.shape[channel_axis], -1)
    num = (g * w).sum(axis=1)
    den = np.linalg.norm(g, axis=1) * np.linalg.norm(w, axis=1)
    per = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    worst = np.argsort(per)[:5]
    return {
        "mean": round(float(per.mean()), 6),
        "min": round(float(per.min()), 6),
        "channels_below_0.9": int((per < 0.9).sum()),
        "of": int(per.size),
        "worst_channels": {str(int(i)): round(float(per[i]), 4) for i in worst},
    }


def active_frames(y: np.ndarray, time_axis: int, limit: int = 12) -> list:
    """Frames that carry energy. For the impulse input these should be the three frames
    around the impulse (A, B), and read directly where Neuron put them."""
    mag = np.abs(y.astype(np.float64)[0])
    per_frame = mag.sum(axis=0 if time_axis == 2 else 1)
    thresh = per_frame.max() * 1e-3
    idx = np.nonzero(per_frame > thresh)[0]
    return [int(i) for i in idx[:limit]] + (["..."] if len(idx) > limit else [])


def impulse_mel(bin_idx: int, frame_idx: int, dtype):
    import torch

    x = torch.zeros(1, 128, 3000, dtype=dtype)
    x[0, bin_idx, frame_idx] = 1.0
    return x


def write(record: dict) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, OUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--graphs", default="A,A_mm,A_flip,B,B_mm,C,D,E,F,G,H,I,J1,J2,J3,A_c,C_c,F_fix")
    parser.add_argument("--impulse-bin", type=int, default=40)
    parser.add_argument("--impulse-frame", type=int, default=100)
    parser.add_argument("--cpu-only", action="store_true",
                        help="no Neuron: check the matmul rewrites against the conv ops on CPU")
    parser.add_argument("--workdir", default="",
                        help="keep each graph's compiler workdir (HLO and NEFF) under this dir, and save its outputs")
    parser.add_argument("--compiler-args", default="--target=inf2 --model-type=transformer --auto-cast=none")
    args = parser.parse_args()

    import torch
    import transformers

    dtype = getattr(torch, args.dtype)
    reference = dict(np.load(RESULTS / "reference_forward.npz"))
    mel = torch.from_numpy(reference["input_features"]).to(dtype)
    imp = impulse_mel(args.impulse_bin, args.impulse_frame, dtype)

    record = {
        "dtype": args.dtype,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "compiler_args": args.compiler_args,
        "instance": os.environ.get("NEURON_INSTANCE_TYPE"),
        "impulse": {"bin": args.impulse_bin, "frame": args.impulse_frame},
        "pass_cosine": PASS_COSINE,
        "graphs": {},
    }
    try:
        import neuronxcc  # noqa: F401
        record["neuronx_cc"] = neuronxcc.__version__
    except Exception:
        record["neuronx_cc"] = None

    model = load_model(Path(args.model).expanduser(), dtype, "eager")
    tower = model.audio_tower
    del model.language_model, model.multi_modal_projector
    tower.layers = torch.nn.ModuleList()  # depth 0: the stem is all that is left
    gc.collect()
    graphs = build_graphs(tower)

    # CPU truth for every graph, on both inputs. B and B_mm take the CPU output of A.
    with torch.no_grad():
        inputs = {
            "mel": {"real": mel, "impulse": imp},
            "mel_t": {"real": mel.permute(0, 2, 1).contiguous(), "impulse": imp.permute(0, 2, 1).contiguous()},
        }
        inputs["a_cpu"] = {k: graphs["A"][0](v) for k, v in inputs["mel"].items()}
        # Only the requested graphs: bf16 conv on CPU is slow, and 18 graphs x 2 inputs took
        # longer than a whole Neuron run.
        requested = {g.strip() for g in args.graphs.split(",") if g.strip()} | {"A", "A_mm", "B", "B_mm"}
        want = {name: {k: g(v) for k, v in inputs[src].items()} for name, (g, src) in graphs.items() if name in requested}

    # The one CPU claim this script makes: the matmul rewrites equal the conv ops. If they
    # do not, a PASS from A_mm on Neuron would mean nothing.
    record["cpu_checks"] = {
        "A_mm_vs_A": compare("A_mm_vs_A", want["A_mm"]["real"].float().numpy(), want["A"]["real"].float().numpy()),
        "B_mm_vs_B": compare("B_mm_vs_B", want["B_mm"]["real"].float().numpy(), want["B"]["real"].float().numpy()),
        "impulse_A_active_frames_cpu": active_frames(want["A"]["impulse"].float().numpy(), time_axis=2),
    }
    for k in ("A_mm_vs_A", "B_mm_vs_B"):
        c = record["cpu_checks"][k]["cosine_similarity"]
        print(f"cpu check {k}: cosine {c}", flush=True)
        if c < PASS_COSINE:
            print("  the matmul rewrite does not match the conv op on CPU; fix that before trusting it on Neuron", flush=True)
    write(record)
    if args.cpu_only:
        print(f"wrote {OUT}", flush=True)
        return

    import torch_neuronx

    first_fail = None
    for name in [g.strip() for g in args.graphs.split(",") if g.strip()]:
        graph, src = graphs[name]
        graph = graph.eval()
        example = inputs[src]["real"]
        entry = {"input": src, "input_shape": list(example.shape)}
        record["graphs"][name] = entry
        write(record)
        started = time.perf_counter()
        try:
            extra = {"compiler_workdir": str(Path(args.workdir) / name)} if args.workdir else {}
            traced = torch_neuronx.trace(graph, (example,), compiler_args=args.compiler_args.split(), **extra)
        except Exception as exc:  # keep the failure, keep going
            entry["compiled"] = False
            entry["error"] = str(exc)[-2000:]
            entry["compile_seconds"] = round(time.perf_counter() - started, 1)
            print(f"{name}: COMPILE FAILED after {entry['compile_seconds']} s", flush=True)
            write(record)
            continue
        entry["compiled"] = True
        entry["compile_seconds"] = round(time.perf_counter() - started, 1)

        for which in ("real", "impulse"):
            x = inputs[src][which]
            with torch.no_grad():
                got = traced(x).float().numpy()
            if args.workdir:
                np.save(Path(args.workdir) / name / f"out_{which}.npy", got)
                np.save(Path(args.workdir) / name / f"want_{which}.npy", want[name][which].float().numpy())
            w = want[name][which].float().numpy()
            stats = compare(f"{name}_{which}", got, w)
            # output layout is [1,C,T] for A..C and A_flip, [1,T,C] for D and E
            channel_axis = 2 if name in ("D", "E", "F", "G", "H", "I", "J1", "J2", "J3", "F_fix") else 1
            time_axis = 1 if name in ("D", "E", "F", "G", "H", "I", "J1", "J2", "J3", "F_fix") else 2
            if got.shape == w.shape:
                stats["cosine_by_channel"] = channel_cosine(got, w, channel_axis)
                if which == "impulse":
                    stats["active_frames_cpu"] = active_frames(w, time_axis)
                    stats["active_frames_neuron"] = active_frames(got, time_axis)
            entry[which] = stats
            write(record)

        cos = entry["real"].get("cosine_similarity", 0.0)
        entry["pass"] = bool(cos >= PASS_COSINE)
        if not entry["pass"] and first_fail is None:
            first_fail = name
        print(
            f"{name}: {'PASS' if entry['pass'] else 'FAIL'}  cosine {cos:.5f}  "
            f"compile {entry['compile_seconds']} s  "
            f"impulse frames cpu {entry['impulse'].get('active_frames_cpu')} "
            f"neuron {entry['impulse'].get('active_frames_neuron')}",
            flush=True,
        )
        write(record)
        del traced
        gc.collect()

    # None only means "all passed" if something compiled; otherwise the probe learned nothing.
    compiled = [n for n, e in record["graphs"].items() if e.get("compiled")]
    record["first_failing_graph"] = first_fail
    record["graphs_compiled"] = len(compiled)
    write(record)
    if not compiled:
        errors = {e.get("error", "") for e in record["graphs"].values()}
        print(f"no graph compiled; {len(errors)} distinct error(s), first one:\n{next(iter(errors))[-1500:]}", flush=True)
    else:
        print(f"first failing graph: {first_fail}  ({len(compiled)} compiled)", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
