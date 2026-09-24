"""The golden forward pass for one routing turn, run in plain PyTorch.

This is the thing the Neuron run is compared against. Without it, a Neuron run only
tells you that a graph executed, not that it executed correctly, and "matches the
reference within 1 point" is an explicit success criterion.

It saves every tensor at the three seams that matter:

    input_features  -> audio_tower       -> encoder_hidden   (the Whisper style encoder)
    encoder_hidden  -> reshape+projector -> audio_embeds      (375 language model tokens)
    inputs_embeds   -> language_model    -> prefill_logits    (the first token decision)

inputs_embeds is saved too, so the decoder stage of the Neuron script needs nothing
from the audio tower and the two stages can be traced independently. That matters on
inf2.xlarge, where 16 GB of host memory is the binding constraint during compilation.

Audio: by default this synthesizes a fixed clip from a closed form formula, so the
reference and the Neuron box agree byte for byte without moving a file between them.
Numeric parity does not care what the audio says. Once the TTS clips exist, rerun with
--wav data/audio/<id>.wav to also get a meaningful greedy decode.

    python scripts/reference_forward.py --model ~/models/voxtral-mini-3b
    python scripts/reference_forward.py --model ~/models/voxtral-mini-3b \
        --wav data/audio/fd24691781c2.wav --dtype float32

Writes results/reference_forward.json and results/reference_forward.npz.
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
DEFAULT_SECONDS = 4.0


def deterministic_wav(path: Path, seconds: float) -> Path:
    """A clip defined by a formula rather than by a file, so any two machines produce
    identical samples. Three chirps plus a syllable rate envelope, which gives the mel
    frontend something with structure across both time and frequency instead of a tone.
    """
    import soundfile as sf

    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float64) / SAMPLE_RATE
    wave = np.zeros_like(t)
    for start_hz, end_hz, amp in ((110.0, 180.0, 0.5), (320.0, 240.0, 0.3), (900.0, 1500.0, 0.15)):
        sweep = start_hz * t + 0.5 * (end_hz - start_hz) / seconds * t**2
        wave += amp * np.sin(2 * np.pi * sweep)
    wave *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    wave = wave / np.abs(wave).max() * 0.9
    sf.write(path, wave.astype(np.float32), SAMPLE_RATE)
    return path


def catalog_text() -> str:
    """The frozen tool catalog, read from the SFT pairs so it cannot drift from training."""
    first = json.loads((ROOT / "data/sft/train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return first["messages"][0]["content"]


def build_inputs(model_dir: Path, wav: Path, catalog: str):
    """The catalog first and the audio second, matching probe_prompt_shape.py: it keeps
    the static prefix cacheable and it puts the audio span at a constant offset.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": catalog}, {"type": "audio", "path": str(wav)}],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True, return_tensors="pt"
    )
    return processor, inputs


def load_model(model_dir: Path, dtype, attn: str):
    from transformers import VoxtralForConditionalGeneration

    # The keyword was renamed from torch_dtype to dtype, and both live in **kwargs rather
    # than in the signature, so asking politely is not possible: try the new name and fall
    # back. The Neuron DLAMI does not necessarily ship the same transformers as the box the
    # reference was built on, so this has to work either way.
    try:
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_dir, attn_implementation=attn, dtype=dtype
        )
    except TypeError:
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_dir, attn_implementation=attn, torch_dtype=dtype
        )
    return model.eval()


def logits_keep_kwarg(module) -> str:
    """logits_to_keep was called num_logits_to_keep before 4.51. Asking for one position
    instead of all of them is the difference between a 1.5 MB output and a 300 MB one.
    """
    params = inspect.signature(module.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--wav", default=None, help="real clip; default is a synthetic one")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                        help="eager is what the Neuron trace uses, so keep them the same")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    import torch

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    model_dir = Path(args.model).expanduser()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    if args.wav:
        wav = Path(args.wav).expanduser()
        if not wav.exists():
            raise SystemExit(f"no such clip: {wav}")
    else:
        wav = results / "reference_clip.wav"
        deterministic_wav(wav, args.seconds)

    catalog = catalog_text()
    processor, inputs = build_inputs(model_dir, wav, catalog)
    model = load_model(model_dir, dtype, args.attn)
    config = model.config

    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"].to(dtype)
    if input_features.shape[0] != 1:
        raise SystemExit(
            f"clip produced {input_features.shape[0]} 30 s windows; the fixed shape plan "
            "assumes one, so either shorten the clip or decide on a window count first"
        )

    facts = {
        "model_dir": str(model_dir),
        "wav": str(wav),
        "wav_is_synthetic": args.wav is None,
        "dtype": args.dtype,
        "attn_implementation": args.attn,
        "prefill_tokens": int(input_ids.shape[1]),
        "mel_shape": list(input_features.shape),
    }

    with torch.no_grad():
        encoder_hidden = model.audio_tower(input_features).last_hidden_state
        intermediate = config.audio_config.intermediate_size
        audio_embeds = model.multi_modal_projector(encoder_hidden.reshape(-1, intermediate))

        inputs_embeds = model.get_input_embeddings()(input_ids).clone()
        audio_mask = (input_ids == config.audio_token_id)[0]
        positions = torch.nonzero(audio_mask).flatten()
        # The splice in modeling_voxtral is a boolean masked assignment, which is a
        # data dependent shape and will not trace. It only needs to be, because the mask
        # is treated as unknown. Here we prove the span is one contiguous run at a fixed
        # offset, which turns the splice into a static slice on Neuron.
        contiguous = bool(positions.numel() and positions[-1] - positions[0] + 1 == positions.numel())
        facts["audio_span"] = {
            "start": int(positions[0]),
            "count": int(positions.numel()),
            "contiguous": contiguous,
        }
        if not contiguous:
            raise SystemExit("audio token positions are not contiguous; the static slice plan is wrong")
        if positions.numel() != audio_embeds.shape[0]:
            raise SystemExit(
                f"{positions.numel()} audio placeholders but {audio_embeds.shape[0]} audio embeddings"
            )
        start = int(positions[0])
        inputs_embeds[:, start : start + audio_embeds.shape[0], :] = audio_embeds.to(inputs_embeds.dtype)

        # The mask is built here and handed in, rather than letting the model build it.
        # transformers builds masks with torch.vmap, which torch.jit.trace cannot capture,
        # so the Neuron graph has to pass a ready made 4D mask: masking_utils returns any
        # 4D mask untouched. The reference has to use the identical mask or the comparison
        # is measuring two different masks. The sequence is exactly full, so this is a
        # plain lower triangle and it is a constant, not an input.
        seq = inputs_embeds.shape[1]
        causal = torch.full((seq, seq), torch.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype)
        causal = torch.triu(causal, diagonal=1)[None, None]

        keep = logits_keep_kwarg(model.language_model)
        lm_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": causal, "use_cache": False}
        if keep:
            lm_kwargs[keep] = 1
        prefill_logits = model.language_model(**lm_kwargs).logits[:, -1, :]
    facts["causal_mask"] = {"shape": list(causal.shape), "built_on_host": True}

    top = torch.topk(prefill_logits.float(), 5, dim=-1)
    facts["first_token"] = {
        "top5_ids": [int(i) for i in top.indices[0]],
        "top5_logits": [round(float(v), 4) for v in top.values[0]],
        "top1_text": processor.tokenizer.decode([int(top.indices[0][0])]),
    }

    if not args.no_generate:
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids, input_features=input_features,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        new_tokens = generated[0].tolist()[int(input_ids.shape[1]):]
        facts["greedy_text"] = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        facts["greedy_new_tokens"] = len(new_tokens)

    # bfloat16 has no numpy equivalent, so everything is stored as float32. The widening
    # is exact, so the stored file is still the exact reference.
    npz = results / "reference_forward.npz"
    np.savez(
        npz,
        input_ids=input_ids.numpy().astype(np.int64),
        input_features=input_features.float().numpy(),
        encoder_hidden=encoder_hidden.float().numpy(),
        audio_embeds=audio_embeds.float().numpy(),
        inputs_embeds=inputs_embeds.float().numpy(),
        prefill_logits=prefill_logits.float().numpy(),
    )
    facts["npz"] = str(npz.relative_to(ROOT))
    facts["npz_mb"] = round(npz.stat().st_size / 1e6, 1)
    facts["tensor_shapes"] = {
        "encoder_hidden": list(encoder_hidden.shape),
        "audio_embeds": list(audio_embeds.shape),
        "inputs_embeds": list(inputs_embeds.shape),
        "prefill_logits": list(prefill_logits.shape),
    }

    (results / "reference_forward.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
