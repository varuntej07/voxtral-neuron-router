"""Does Voxtral's forward pass compile and run on Inferentia2, and does it match the reference.

Two graphs, traced and compared separately:

    encoder   input_features [1, 128, 3000] -> (encoder_hidden, audio_embeds)
    decoder   inputs_embeds  [1, S, 3072]   -> last position logits [1, vocab]

They are separate on purpose. The splice between them, writing 375 audio embeddings into
the text embedding sequence, is a boolean masked assignment in modeling_voxtral, which is
a data dependent shape. reference_forward.py proves the audio span is one contiguous run
at a constant offset, so on Neuron it is a static slice and it belongs on the host, not in
either graph. Splitting also means a failure tells you which half failed, and it keeps two
smaller compilations instead of one large one, which matters because inf2.xlarge has 4
vCPU and 16 GB of host memory and the compiler runs on the host.

Shape decisions, since Neuron compiles one graph per shape:

  * mel is [1, 128, 3000] always. VoxtralEncoder raises unless the mel length is exactly
    max_source_positions * 2, so the 30 s window is not a choice, it is the contract. A
    4 s clip and a 29 s clip are the same graph, which is why prompt_shape.json shows 375
    audio tokens regardless of clip length.
  * the decoder sequence is whatever reference_forward.py measured, not a padded bucket.
    The prompt is the frozen catalog plus a fixed 375 audio tokens plus fixed turn
    markers, so the prefill length is a constant and there is nothing to pad. The causal
    mask is therefore a plain lower triangle, built on the host and carried as a graph
    constant rather than an input. It has to be handed in rather than left to the model:
    transformers builds masks with torch.vmap and torch.jit.trace cannot capture that.
  * --auto-cast=none, so the compiler does not silently change precision. The weights are
    loaded in bf16 and the reference is computed in bf16, so any difference that shows up
    is the hardware and the compiler, not a dtype the compiler chose on its own.

    source scripts/neuron_env_inf2.sh
    python -u scripts/neuron_forward_smoke.py --stage encoder
    python -u scripts/neuron_forward_smoke.py --stage decoder

Use python -u. A full compile is opaque for minutes at a time and buffered output makes it
look like a hang.

Debug graphs, for when the question is why a compile fails rather than how fast it runs:

    python -u scripts/neuron_forward_smoke.py --stage encoder --layers 1
    python -u scripts/neuron_forward_smoke.py --stage decoder --seq-len 64 --layers 2
    python -u scripts/neuron_forward_smoke.py --stage encoder --layers 1 --compile-only

--layers and --seq-len cut the graph down so an iteration costs a fraction of a full
compile. A cut run compares against a CPU run of the same cut rather than against the
stored full length reference, is marked truncated, and is written under its own key, so it
cannot be mistaken for or overwrite an acceptance number. --compile-only stops before the
graph executes, which is what lets a host with no Neuron device be useful: tracing and
compiling are host CPU work and only running the NEFF needs a chip.

Writes results/neuron_parity.json after every phase, not once at the end. A three hour
session once produced no file at all because this script wrote only on the way out, and on
failure only after torch_neuronx.analyze, which recompiles operator by operator and can run
longer than the compile that just failed. analyze is now opt in behind --analyze and the
error reaches disk before it runs. Ctrl-C at any point leaves a record naming the phase it
died in.
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Both scripts have to load the model and ask for logits the same way, so the loader lives
# in one place and the reference is the place it lives.
from reference_forward import load_model, logits_keep_kwarg

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COMPARE = {
    "encoder": ("encoder_hidden", "audio_embeds"),
    "decoder": ("prefill_logits",),
}

# The record being built, at module scope so the signal handler can bank it. A three hour
# session once ended with an empty results/ because this script only wrote on the way out:
# on success after the latency loop, on failure only after torch_neuronx.analyze, which can
# run longer than the compile that failed. Anything that ends the process now leaves a file
# behind saying which phase it died in.
_RECORD: dict = {}
_STARTED = time.perf_counter()


def checkpoint(phase: str | None = None, **fields) -> None:
    """Update the in-flight record and put it on disk immediately.

    Called after every phase rather than at the end, because the whole value of this script
    on a long compile is that it says something while it is still running.
    """
    if phase is not None:
        _RECORD["phase"] = phase
        elapsed = round(time.perf_counter() - _STARTED, 1)
        _RECORD["phase_elapsed_seconds"] = elapsed
        print(f"[{_RECORD.get('stage', '?')}] {phase} at {elapsed}s", flush=True)
    _RECORD.update(fields)
    if _RECORD.get("stage"):
        write(_RECORD)


def _bank_and_exit(signum, _frame):
    """Ctrl-C during a multi hour compile used to discard everything. Now it writes."""
    checkpoint(f"interrupted by signal {signum} during {_RECORD.get('phase', 'startup')}")
    _RECORD["interrupted"] = True
    if _RECORD.get("stage"):
        write(_RECORD)
    sys.exit(130)


def compare(name: str, got: np.ndarray, want: np.ndarray) -> dict:
    """Absolute and relative error, plus the thing that actually decides the routing
    output: whether the argmax moved.
    """
    got, want = got.astype(np.float64), want.astype(np.float64)
    if got.shape != want.shape:
        return {"tensor": name, "shape_mismatch": [list(got.shape), list(want.shape)]}
    diff = np.abs(got - want)
    denom = np.linalg.norm(want) or 1.0
    flat_got, flat_want = got.reshape(-1), want.reshape(-1)
    cosine = float(flat_got @ flat_want / ((np.linalg.norm(flat_got) or 1.0) * (np.linalg.norm(flat_want) or 1.0)))
    stats = {
        "tensor": name,
        "shape": list(got.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "relative_frobenius_error": float(np.linalg.norm(got - want) / denom),
        "cosine_similarity": round(cosine, 8),
        "reference_abs_max": float(np.abs(want).max()),
        "neuron_abs_max": float(np.abs(got).max()),
    }
    if got.ndim == 3 and got.shape[0] == 1:
        # Where along the sequence the agreement breaks tells you what kind of bug it is.
        # Near 1.0 at position 0 decaying with position points at positional embeddings or
        # the attention mask. A flat low value at every position points at channel layout
        # or the conv frontend. Both are structural, and neither looks like precision.
        a, b = got[0], want[0]
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        per_pos = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        probe = [0, 1, 2, len(per_pos) // 4, len(per_pos) // 2, len(per_pos) - 1]
        stats["cosine_by_position"] = {
            str(i): round(float(per_pos[i]), 6) for i in sorted(set(probe)) if i < len(per_pos)
        }
        stats["cosine_by_position_mean"] = round(float(per_pos.mean()), 6)
        stats["cosine_by_position_min"] = round(float(per_pos.min()), 6)
    if name == "prefill_logits":
        top_got = np.argsort(-flat_got)[:5]
        top_want = np.argsort(-flat_want)[:5]
        stats["top1_agrees"] = bool(top_got[0] == top_want[0])
        stats["top5_overlap"] = int(len(set(top_got.tolist()) & set(top_want.tolist())))
        stats["neuron_top5_ids"] = [int(i) for i in top_got]
        stats["reference_top5_ids"] = [int(i) for i in top_want]
    return stats


def causal_mask(seq_len: int, dtype):
    """An additive lower triangular mask, built on the host and baked into the graph.

    transformers builds attention masks with torch.vmap, and torch.jit.trace, which
    torch_neuronx.trace runs first, cannot capture vmap: it fails with an unhelpful
    "invalid unordered_map key" from functorch. masking_utils returns any already 4D mask
    untouched, so handing one in skips the mask builder entirely. The sequence is exactly
    full, so the correct mask is a plain lower triangle with no padding to account for.
    """
    import torch

    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1)[None, None]


def build_graph(stage: str, model_dir: Path, dtype, attn: str, inputs, layers: int | None = None):
    """Return (module, output_names, want, full_depth). The wrapper keeps only the
    submodules the stage needs so the rest of the 3B can be freed before compiling.

    `want` is the CPU output of this exact module on this exact input, computed here rather
    than read from the reference npz. That matters whenever `layers` truncates the graph: a
    one layer Neuron output compared against a 32 layer stored reference produces a number
    that looks like parity and means nothing. It costs no extra peak memory, because
    torch.jit.trace, which torch_neuronx.trace runs first, already does this same CPU
    forward internally.
    """
    import torch

    model = load_model(model_dir, dtype, attn)

    if stage == "encoder":
        intermediate = model.config.audio_config.intermediate_size
        tower = model.audio_tower
        full_depth = len(tower.layers)
        if layers is not None:
            if layers > full_depth:
                raise SystemExit(f"--layers {layers} exceeds the encoder depth {full_depth}")
            tower.layers = torch.nn.ModuleList(list(tower.layers)[:layers])

        class EncoderGraph(torch.nn.Module):
            def __init__(self, tower, projector):
                super().__init__()
                self.audio_tower = tower
                self.projector = projector

            def forward(self, input_features):
                hidden = self.audio_tower(input_features).last_hidden_state
                return hidden, self.projector(hidden.reshape(-1, intermediate))

        graph = EncoderGraph(tower, model.multi_modal_projector).eval()
        names = COMPARE["encoder"]
    else:
        language_model = model.language_model
        keep = logits_keep_kwarg(language_model)
        decoder_layers = language_model.model.layers
        full_depth = len(decoder_layers)
        if layers is not None:
            if layers > full_depth:
                raise SystemExit(f"--layers {layers} exceeds the decoder depth {full_depth}")
            # The final norm and lm_head stay, so the output is still [1, vocab] and the
            # top1_agrees branch of compare() keeps working.
            language_model.model.layers = torch.nn.ModuleList(list(decoder_layers)[:layers])

        class DecoderPrefill(torch.nn.Module):
            def __init__(self, language_model, mask):
                super().__init__()
                self.language_model = language_model
                # A buffer, not an input: the mask is the same tensor on every call, so it
                # belongs in the graph as a constant and the graph keeps one input.
                self.register_buffer("causal_mask", mask, persistent=False)

            def forward(self, inputs_embeds):
                extra = {keep: 1} if keep else {}
                return self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=self.causal_mask,
                    use_cache=False,
                    **extra,
                ).logits[:, -1, :]

        # The mask is sized from the input that will actually be traced, so --seq-len needs
        # no change here: slice the input and the [1, 1, N, N] mask follows.
        graph = DecoderPrefill(language_model, causal_mask(int(inputs[0].shape[1]), dtype)).eval()
        names = COMPARE["decoder"]

    # The CPU truth for this exact module on this exact input, before the model is freed.
    with torch.no_grad():
        out = graph(*inputs)
    out = (out,) if isinstance(out, torch.Tensor) else tuple(out)
    want = {name: out[i].float().numpy() for i, name in enumerate(names)}

    del model, out
    gc.collect()
    return graph, names, want, full_depth


def example_input(stage: str, reference: dict, dtype, seq_len: int | None = None):
    """The traced input, optionally cut short for a cheap debug compile.

    Slicing the decoder prefix is exact rather than approximate. The mask is a plain lower
    triangle with no padding, so position i depends only on positions up to i, which makes
    the last logit of an N token prefix the same function the full run computes at position
    N-1. That is what allows a truncated run to be a real parity test rather than a smoke
    signal. It has to be compared against a CPU run of the same cut, though, not against the
    stored full length reference.
    """
    import torch

    key = "input_features" if stage == "encoder" else "inputs_embeds"
    tensor = torch.from_numpy(reference[key]).to(dtype)
    if seq_len is not None:
        if stage == "encoder":
            # VoxtralEncoder raises unless the mel length is exactly max_source_positions*2,
            # so the 30 s window is the contract, not a choice. Say so here rather than let
            # transformers fail further in with a less obvious message.
            raise SystemExit(
                "--seq-len does not apply to the encoder: the mel is fixed at [1, 128, 3000] "
                "and VoxtralEncoder rejects any other length. Use --layers to make the "
                "encoder graph cheaper."
            )
        if seq_len > tensor.shape[1]:
            raise SystemExit(
                f"--seq-len {seq_len} exceeds the reference length {tensor.shape[1]}"
            )
        tensor = tensor[:, :seq_len, :].contiguous()
    return (tensor,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["encoder", "decoder"])
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--artifacts", default="~/neuron_artifacts")
    parser.add_argument("--reuse", action="store_true", help="load a saved .pt instead of compiling")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compiler-args",
                        default="--target=inf2 --model-type=transformer --auto-cast=none",
                        help="passed to torch_neuronx.trace, which is what decides rather "
                             "than NEURON_CC_FLAGS, so the inf2 target is named here too")
    parser.add_argument("--analyze", action="store_true",
                        help="run torch_neuronx.analyze after a failed trace. Off by default: "
                             "it recompiles the graph operator by operator, which on a 3B "
                             "model can take longer than the compile that just failed, and "
                             "it used to run before the error was written to disk")
    parser.add_argument("--layers", type=int, default=None,
                        help="trace only the first N transformer layers. A cheap debug graph; "
                             "parity is then computed against a CPU run of the same cut, and "
                             "the record is marked truncated")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="decoder only: trace an N token prefix instead of the full 601. "
                             "Exact, not approximate, because the mask is unpadded lower "
                             "triangular")
    parser.add_argument("--compile-only", action="store_true",
                        help="stop after the compile. This is what makes a host with no "
                             "Neuron device useful: compiling is a host CPU job, only "
                             "executing the NEFF needs a chip")
    parser.add_argument("--dump-outputs", action="store_true",
                        help="save the raw Neuron tensors so the next hypothesis about a "
                             "parity gap costs no recompile")
    args = parser.parse_args()

    if args.seq_len is not None and args.stage == "encoder":
        raise SystemExit(
            "--seq-len does not apply to the encoder: the mel is fixed at [1, 128, 3000]. "
            "Use --layers instead."
        )
    signal.signal(signal.SIGINT, _bank_and_exit)
    signal.signal(signal.SIGTERM, _bank_and_exit)

    import torch
    import torch_neuronx

    dtype = getattr(torch, args.dtype)
    npz = RESULTS / "reference_forward.npz"
    if not npz.exists():
        raise SystemExit("run scripts/reference_forward.py first; there is nothing to compare against")
    reference = dict(np.load(npz))
    ref_facts = json.loads((RESULTS / "reference_forward.json").read_text(encoding="utf-8"))

    truncated = args.layers is not None or args.seq_len is not None
    suffix = ""
    if args.layers is not None:
        suffix += f"_L{args.layers}"
    if args.seq_len is not None:
        suffix += f"_S{args.seq_len}"

    artifacts = Path(args.artifacts).expanduser()
    artifacts.mkdir(parents=True, exist_ok=True)
    # The depth and length ride in the filename so a toy compile cannot overwrite the real
    # artifact, and so --reuse cannot silently load a toy in place of it.
    saved = artifacts / f"voxtral_{args.stage}_{args.dtype}{suffix}.pt"

    record = _RECORD
    record.update({
        "stage": args.stage,
        "dtype": args.dtype,
        "attn_implementation": args.attn,
        "compiler_args": args.compiler_args,
        "layers": args.layers,
        "seq_len_override": args.seq_len,
        "truncated": truncated,
        "baseline": "cpu_same_graph" if truncated else "reference_npz",
        "compile_only": args.compile_only,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_dtype": ref_facts.get("dtype"),
        "reference_prefill_tokens": ref_facts.get("prefill_tokens"),
        "wav_is_synthetic": ref_facts.get("wav_is_synthetic"),
        "instance": os.environ.get("NEURON_INSTANCE_TYPE", "unrecorded"),
        "artifact_path": str(saved),
        "compiler_workdir": str(artifacts / f"workdir_{args.stage}{suffix}"),
    })
    if truncated:
        record["truncation_note"] = (
            "a debug graph, not an acceptance run: parity is against a CPU run of the same "
            "cut, and these numbers must not be quoted as the Neuron result"
        )
    # On disk before anything slow starts, so even an immediate kill leaves a record.
    checkpoint("started")
    if ref_facts.get("dtype") != args.dtype:
        record["warning"] = (
            f"reference is {ref_facts.get('dtype')} but this run is {args.dtype}; the "
            "difference below includes the dtype change, not just Neuron"
        )

    inputs = example_input(args.stage, reference, dtype, seq_len=args.seq_len)
    checkpoint("input ready", input_shape=list(inputs[0].shape))

    names = COMPARE[args.stage]
    want = None
    if args.reuse and saved.exists():
        traced = torch.jit.load(str(saved))
        checkpoint("artifact reused", compiled=True, reused_artifact=str(saved))
        if truncated:
            raise SystemExit(
                "--reuse with --layers/--seq-len has no CPU baseline to compare against; "
                "drop --reuse so the truncated graph is rebuilt and its baseline computed"
            )
    else:
        checkpoint(f"loading the model and building the {args.stage} graph")
        graph, names, want, full_depth = build_graph(
            args.stage, Path(args.model).expanduser(), dtype, args.attn, inputs, args.layers
        )
        checkpoint("cpu baseline done", full_depth=full_depth)
        print(f"    compiler workdir: {record['compiler_workdir']}", flush=True)
        print("    tail -f that directory's log to watch neuronx-cc while this call is opaque",
              flush=True)
        checkpoint("compiling")
        started = time.perf_counter()
        try:
            traced = torch_neuronx.trace(
                graph,
                inputs,
                compiler_workdir=record["compiler_workdir"],
                compiler_args=args.compiler_args.split(),
            )
            torch.jit.save(traced, str(saved))
            checkpoint(
                "compiled",
                compiled=True,
                compile_seconds=round(time.perf_counter() - started, 1),
                artifact=str(saved),
                artifact_mb=round(saved.stat().st_size / 1e6, 1),
            )
        except Exception:
            # The error goes to disk before anything else is attempted. analyze used to run
            # here first, recompiling the graph operator by operator, and a ctrl-C during it
            # threw away the traceback this whole run existed to produce.
            checkpoint(
                "trace failed",
                compiled=False,
                compile_seconds=round(time.perf_counter() - started, 1),
                error=traceback.format_exc()[-4000:],
            )
            if args.analyze:
                checkpoint("analyzing (opt in; this recompiles operator by operator)")
                try:
                    record["analysis"] = str(torch_neuronx.analyze(graph, inputs))[:4000]
                except Exception as exc:
                    record["analysis_error"] = f"{type(exc).__name__}: {exc}"
                checkpoint("analyze done")
            raise SystemExit(f"{args.stage}: trace failed, see results/neuron_parity.json")

    if args.compile_only:
        checkpoint("done, compile only: no execution, no parity, no latency")
        print(json.dumps(record, indent=2), flush=True)
        return

    checkpoint("executing on device")
    outputs = traced(*inputs)
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    got = {name: outputs[i].float().numpy() for i, name in enumerate(names)}

    # A truncated graph is compared against the CPU run of that same cut. Comparing a one
    # layer output against the stored 32 layer reference would produce a number that looks
    # like parity and means nothing.
    if truncated:
        assert want is not None, "a truncated run always builds its own CPU baseline"
        baseline = want
    else:
        baseline = reference
    record["parity"] = [compare(name, got[name], baseline[name]) for name in names]
    checkpoint("parity done")

    if args.dump_outputs:
        out_path = RESULTS / f"neuron_{args.stage}{suffix}_out.npz"
        bundle = {f"got_{k}": got[k] for k in names}
        bundle.update({f"want_{k}": np.asarray(baseline[k]) for k in names})
        np.savez(str(out_path), **bundle)  # type: ignore[arg-type]
        checkpoint("outputs dumped", dumped_outputs=str(out_path))

    # Graph latency on one NeuronCore, which is not the end to end turn latency. It is a
    # floor for it, and it is the number that says whether the compile is worth keeping.
    checkpoint("timing")
    for _ in range(3):
        traced(*inputs)
    samples = []
    for _ in range(args.iters):
        started = time.perf_counter()
        traced(*inputs)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    latency = {
        "p50": round(samples[len(samples) // 2], 2),
        "p95": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 2),
        "min": round(samples[0], 2),
        "iters": args.iters,
        "note": "single graph, one NeuronCore, batch 1, host to device copy included",
    }
    if truncated:
        latency["note"] += "; TRUNCATED GRAPH, not the servable latency"
    checkpoint("done", graph_latency_ms=latency)
    print(json.dumps(record, indent=2), flush=True)


def record_key(record: dict) -> str:
    """The entry name in neuron_parity.json.

    A full run keys on the stage alone, so "encoder" always means the real thing. A
    truncated debug run gets its own key, because keying everything on the stage meant a
    30 second --layers 1 toy could silently overwrite a 286 second full compile and the
    latency that went with it.
    """
    key = record["stage"]
    if record.get("layers") is not None:
        key += f"@L{record['layers']}"
    if record.get("seq_len_override") is not None:
        key += f"@S{record['seq_len_override']}"
    return key


def write(record: dict) -> None:
    """One file, one entry per stage, so running the stages on different days still leaves
    a single readable record.

    Written atomically: this is called after every phase now, and a ctrl-C landing between
    the read and the write would otherwise leave a truncated JSON file and destroy the
    earlier stage's results along with the current one.
    """
    path = RESULTS / "neuron_parity.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Never let one corrupt file stop a run from banking what it has.
            existing = {}
    existing[record_key(record)] = record
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_cli() -> None:
    """main() wrapped in the last safety net.

    BaseException rather than Exception, so a KeyboardInterrupt arriving outside the signal
    handler still banks what the run had reached. Separate from __main__ so the net itself
    can be tested.
    """
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        if _RECORD.get("stage"):
            _RECORD["fatal"] = f"{type(exc).__name__}: {exc}"
            _RECORD["traceback"] = traceback.format_exc()[-4000:]
            write(_RECORD)
            print(f"banked a partial record at {_RECORD.get('phase')}", flush=True)
        raise


if __name__ == "__main__":
    run_cli()
