"""Tokenize and mel-extract the SFT pairs once, into fixed-shape arrays Neuron can eat.

Neuron compiles one graph per input shape, so every example has to arrive at the same
shape. This does all the variable-length work on CPU, ahead of time, and writes three
arrays per split to data/cache/<split>.npz:

    input_features  (n, 128, 3000)    one 30 s mel window, padded with silence
    input_ids       (n, seq)          right-padded with the pad token
    labels          (n, seq)          -100 everywhere but the assistant JSON

Two measured facts drive the layout (see results/prompt_shape.json, written by
scripts/probe_prompt_shape.py):

  - Tokenizer v7 rejects a system message in the same conversation as audio, so the
    tool catalog rides in the user turn as a text chunk ahead of the audio. The
    text-first order also keeps the static prefix cacheable at prefill.
  - The audio span is 375 tokens starting at index 225, for any clip from 1 s to 30 s.
    That constant span is what lets the training forward splice audio in with a cat
    instead of the boolean scatter stock Voxtral uses.

Tokens come from mistral_common and mels from the model's own feature extractor,
which is exactly what VoxtralProcessor does internally, split apart so the prompt and
target boundary is checkable rather than guessed.

Audio is optional on purpose. While the clips are still being generated, --silence
builds the cache from the text pairs with silent audio, which is enough to compile the
graph, measure step time, and shake out the loader.

    python scripts/prepare_sft_cache.py --split train --silence      # today
    python scripts/prepare_sft_cache.py --split train                # once the wavs land
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "data" / "sft"
CACHE_DIR = ROOT / "data" / "cache"

SAMPLE_RATE = 16000
AUDIO_TOKEN_ID = 24
WINDOW_SECONDS = 30  # the encoder accepts exactly one 30 s window, nothing shorter


def build_tokenizers(model_dir: Path) -> tuple:
    """Two views of the same tokenizer.

    mistral_common validates message structure per mode: finetuning insists the last
    turn is the assistant, serving insists it is not. The full sequence needs the
    first, the prompt-only sequence needs the second.
    """
    from mistral_common.protocol.instruct.validator import ValidationMode
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    path = str(model_dir / "tekken.json")
    return (
        MistralTokenizer.from_file(path, mode=ValidationMode.finetuning),
        MistralTokenizer.from_file(path, mode=ValidationMode.serving),
    )


def encode(tokenizers, catalog: str, target: str, wav: Path) -> tuple[list[int], int]:
    """Full token sequence plus the index where the assistant target starts."""
    from mistral_common.protocol.instruct.messages import AssistantMessage, AudioChunk, TextChunk, UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.audio import Audio

    for_training, for_prompt = tokenizers
    audio = Audio.from_file(str(wav), strict=False)
    turn = UserMessage(content=[TextChunk(text=catalog), AudioChunk(input_audio=audio.to_base64("wav"))])

    # Serving mode wants a model name even though nothing is served here.
    prompt = for_prompt.encode_chat_completion(ChatCompletionRequest(messages=[turn], model="voxtral")).tokens
    full = for_training.encode_chat_completion(
        ChatCompletionRequest(messages=[turn, AssistantMessage(content=target)])
    ).tokens

    if full[: len(prompt)] != prompt:
        raise ValueError("the prompt-only encoding is not a prefix of the full one")
    return full, len(prompt)


def mel_window(extractor, audio: np.ndarray) -> np.ndarray:
    """One 30 s mel window. The encoder asserts this exact length, so no padding choices."""
    want = SAMPLE_RATE * WINDOW_SECONDS
    if len(audio) > want:
        raise ValueError(f"clip is {len(audio) / SAMPLE_RATE:.1f}s, longer than the {WINDOW_SECONDS}s window")
    padded = np.pad(audio, (0, want - len(audio)))
    features = extractor(padded, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"]
    return features[0]


def read_clip(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {rate} Hz, expected {SAMPLE_RATE}")
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--seq", type=int, default=768, help="from results/prompt_shape.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--silence", action="store_true", help="skip the wavs, use silence")
    args = parser.parse_args()

    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    model_dir = Path(args.model).expanduser()
    tokenizers = build_tokenizers(model_dir)
    extractor = WhisperFeatureExtractor.from_pretrained(model_dir)

    pairs = [json.loads(l) for l in (SFT_DIR / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    if args.limit:
        pairs = pairs[: args.limit]

    # Tokenization needs a file, so silence mode writes one and reuses it for every row.
    tmp = Path(tempfile.mkdtemp())
    silence_wav = tmp / "silence.wav"
    sf.write(silence_wav, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    ids_rows, mel_rows, prompt_lens = [], [], []
    for pair in pairs:
        catalog = pair["messages"][0]["content"]
        target = pair["messages"][2]["content"]
        wav = silence_wav if args.silence else ROOT / pair["audio"]
        ids, prompt_len = encode(tokenizers, catalog, target, wav)
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32) if args.silence else read_clip(wav)
        ids_rows.append(ids)
        mel_rows.append(mel_window(extractor, audio))
        prompt_lens.append(prompt_len)

    longest = max(len(r) for r in ids_rows)
    if longest > args.seq:
        raise SystemExit(f"--seq {args.seq} is shorter than the longest row ({longest})")

    pad_id = tokenizers[0].instruct_tokenizer.tokenizer.pad_id
    input_ids = np.full((len(ids_rows), args.seq), pad_id, dtype=np.int32)
    labels = np.full((len(ids_rows), args.seq), -100, dtype=np.int32)
    for i, (row, prompt_len) in enumerate(zip(ids_rows, prompt_lens)):
        input_ids[i, : len(row)] = row
        labels[i, prompt_len : len(row)] = row[prompt_len:]

    # The user turn is byte-identical in every row, so the target starts at a fixed
    # index. That is what lets the loss read a constant slice of the logits instead of
    # running the 131k-wide lm_head over the whole padded sequence.
    if len(set(prompt_lens)) != 1:
        raise SystemExit(f"prompt length varies between rows: {sorted(set(prompt_lens))[:5]}")

    spans = [np.flatnonzero(row == AUDIO_TOKEN_ID) for row in input_ids]
    starts = {int(s[0]) for s in spans}
    widths = {len(s) for s in spans}
    if len(starts) != 1 or len(widths) != 1:
        raise SystemExit(f"audio span moves between rows: starts {sorted(starts)[:5]}, widths {sorted(widths)[:5]}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_DIR / f"{args.split}.npz",
        input_ids=input_ids,
        labels=labels,
        input_features=np.stack(mel_rows).astype(np.float32),
    )
    meta = {
        "split": args.split,
        "rows": len(ids_rows),
        "seq": args.seq,
        "longest_row": longest,
        "audio_span_start": starts.pop(),
        "audio_span_width": widths.pop(),
        # Loss only touches the tail, so the forward can drop the rest of the logits.
        "prompt_len": prompt_lens[0],
        "target_tokens_max": max(len(r) - p for r, p in zip(ids_rows, prompt_lens)),
        "mel_shape": list(mel_rows[0].shape),
        "pad_token_id": int(pad_id),
        "silence": args.silence,
    }
    (CACHE_DIR / f"{args.split}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
