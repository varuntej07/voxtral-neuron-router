"""Measure the token shape of one routing turn, before anything is compiled.

Neuron compiles one graph per input shape, so the sequence length, the audio span
position and the audio span width have to be known and constant before training.
This measures all three from the real tokenizer instead of assuming them, and writes
results/prompt_shape.json.

It needs only tekken.json from the model repo, not the weights:

    curl -sSLO --output-dir ~/models/voxtral-mini-3b \\
        https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/resolve/main/tekken.json
    pip install "mistral_common[audio]" soundfile
    python scripts/probe_prompt_shape.py
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
AUDIO_TOKEN_ID = 24  # config.json: audio_token_id
PROBE_SECONDS = (1, 2, 3, 7, 15, 30)


def build_tokenizer(tekken: Path):
    from mistral_common.protocol.instruct.validator import ValidationMode
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    # finetuning mode lets the assistant turn be last, which serving mode rejects.
    return MistralTokenizer.from_file(str(tekken), mode=ValidationMode.finetuning)


def encode(tokenizer, catalog: str, target: str, wav: Path) -> list[int]:
    from mistral_common.protocol.instruct.messages import AssistantMessage, AudioChunk, TextChunk, UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.audio import Audio

    audio = Audio.from_file(str(wav), strict=False)
    # The catalog goes first and the audio second so the static prefix stays cacheable
    # as a prefill prefix, and so the audio span lands at a constant offset.
    request = ChatCompletionRequest(
        messages=[
            UserMessage(content=[TextChunk(text=catalog), AudioChunk(input_audio=audio.to_base64("wav"))]),
            AssistantMessage(content=target),
        ]
    )
    return tokenizer.encode_chat_completion(request).tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tekken", default="~/models/voxtral-mini-3b/tekken.json")
    args = parser.parse_args()

    tokenizer = build_tokenizer(Path(args.tekken).expanduser())
    pairs = [json.loads(l) for l in (ROOT / "data/sft/train.jsonl").read_text(encoding="utf-8").splitlines() if l]
    catalog = pairs[0]["messages"][0]["content"]

    tmp = Path(tempfile.mkdtemp())
    by_seconds = {}
    for seconds in PROBE_SECONDS:
        wav = tmp / f"silence_{seconds}s.wav"
        sf.write(wav, np.zeros(SAMPLE_RATE * seconds, dtype=np.float32), SAMPLE_RATE)
        ids = encode(tokenizer, catalog, pairs[0]["messages"][2]["content"], wav)
        span = [i for i, t in enumerate(ids) if t == AUDIO_TOKEN_ID]
        by_seconds[seconds] = {"total": len(ids), "audio_tokens": len(span), "audio_start": span[0]}

    # The target JSON is what the loss is computed on, so its length sets logits_to_keep.
    targets = [len(tokenizer.instruct_tokenizer.tokenizer.encode(p["messages"][2]["content"], bos=False, eos=False))
               for p in pairs]

    shapes = {n["total"] for n in by_seconds.values()}
    facts = {
        "tokenizer_version": "v7",
        "audio_token_id": AUDIO_TOKEN_ID,
        "by_clip_seconds": by_seconds,
        "constant_regardless_of_clip_length": len(shapes) == 1,
        "audio_tokens": by_seconds[PROBE_SECONDS[0]]["audio_tokens"],
        "audio_span_start": by_seconds[PROBE_SECONDS[0]]["audio_start"],
        "target_tokens": {"max": max(targets), "p95": int(np.percentile(targets, 95)), "median": int(np.median(targets))},
        "rows": len(pairs),
        "note": "system role plus audio is rejected by tokenizer v7, so the tool catalog "
                "rides in the user turn as a text chunk ahead of the audio",
    }
    longest = max(n["total"] for n in by_seconds.values()) - min(targets) + max(targets)
    facts["suggested_seq_bucket"] = ((longest + 63) // 64) * 64

    out = ROOT / "results" / "prompt_shape.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
