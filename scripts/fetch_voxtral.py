"""Download Voxtral Mini 3B once and print the numbers the Neuron work depends on.

Run this on the trn1 box (or any CPU box with the HF token) before touching Neuron.
It writes the snapshot to --out and prints, from the real config rather than memory:
the mel frame count a 30 s window produces, how many audio tokens that becomes,
the module names LoRA will attach to, and the parameter counts that follow.

Those numbers decide the fixed shapes, and Neuron compiles one graph per shape,
so they belong in results/ before any compile is attempted.

    export HF_TOKEN=...                       # the repo is gated, accept the terms first
    pip install "transformers>=4.55" "huggingface_hub>=0.34" mistral_common soundfile
    python scripts/fetch_voxtral.py --out ~/models/voxtral-mini-3b
"""

import argparse
import json
from pathlib import Path

REPO = "mistralai/Voxtral-Mini-3B-2507"
# Everything except the duplicate formats: consolidated.safetensors is the Mistral-format
# copy of the same weights, and *.pth / *.gguf are not what transformers loads.
IGNORE = ["consolidated*", "*.pth", "*.gguf", "original/*"]

LORA_HINTS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def summarize(path: Path) -> dict:
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(path)
    audio, text = config.audio_config, config.text_config
    processor = AutoProcessor.from_pretrained(path)
    extractor = processor.feature_extractor

    # The encoder halves the time axis in conv2 (stride 2), then the projector packs
    # `intermediate_size / hidden_size` encoder frames into one language-model token.
    mel_frames = audio.max_source_positions * 2
    encoder_frames = audio.max_source_positions
    frames_per_token = audio.intermediate_size // audio.hidden_size
    audio_tokens = encoder_frames // frames_per_token
    window_seconds = mel_frames * extractor.hop_length / extractor.sampling_rate

    return {
        "repo": REPO,
        "sampling_rate": extractor.sampling_rate,
        "mel_bins": extractor.feature_size,
        "hop_length": extractor.hop_length,
        "window_seconds": window_seconds,
        "mel_frames_per_window": mel_frames,
        "encoder_frames_per_window": encoder_frames,
        "frames_packed_per_audio_token": frames_per_token,
        "audio_tokens_per_window": audio_tokens,
        "audio_token_id": config.audio_token_id,
        "audio_hidden_size": audio.hidden_size,
        "audio_layers": audio.num_hidden_layers,
        "text_hidden_size": text.hidden_size,
        "text_layers": text.num_hidden_layers,
        "text_heads": text.num_attention_heads,
        "text_kv_heads": getattr(text, "num_key_value_heads", text.num_attention_heads),
        "text_vocab_size": text.vocab_size,
        "text_max_position_embeddings": text.max_position_embeddings,
    }


def module_report(path: Path) -> dict:
    """Load on meta device so this runs on a laptop: names and shapes, no weights."""
    import torch
    from transformers import AutoConfig, VoxtralForConditionalGeneration

    config = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = VoxtralForConditionalGeneration._from_config(config)

    targets, totals = set(), {"audio_tower": 0, "multi_modal_projector": 0, "language_model": 0}
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if name.startswith("language_model") and leaf in LORA_HINTS:
            targets.add(leaf)
    for name, param in model.named_parameters():
        for part in totals:
            if name.startswith(part):
                totals[part] += param.numel()
    return {"lora_target_modules": sorted(targets), "params_by_part": totals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="~/models/voxtral-mini-3b")
    parser.add_argument("--skip-download", action="store_true", help="just re-read a snapshot")
    args = parser.parse_args()

    out = Path(args.out).expanduser()
    if not args.skip_download:
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, local_dir=out, ignore_patterns=IGNORE)

    facts = summarize(out)
    facts.update(module_report(out))
    facts["local_dir"] = str(out)

    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)
    (results / "voxtral_shapes.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
