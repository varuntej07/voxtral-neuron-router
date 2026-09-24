"""Speak the user turns of data/text/rows.jsonl as 16 kHz clips on a GPU instance (EC2).

Same output and checks as colab_synthesize_audio.py: data/audio/<id>.wav plus voices.jsonl,
seeded Piper/Kokoro voices, Whisper large-v3 exact-match check, misses redone with a new voice.
Built for speed: TTS runs in --workers CPU processes with one ONNX thread each, and Whisper
checks finished chunks on the GPU while the workers keep speaking.

    pip install "piper-tts>=1.8" "kokoro-onnx>=0.6.1" soundfile librosa jiwer transformers
    python scripts/synthesize_audio.py --limit 64      # smoke test: prints text vs heard
    python scripts/synthesize_audio.py                 # everything; rerun resumes
"""

import argparse
import json
import logging
import os
import random
import re
import time
import urllib.request
from itertools import islice
from multiprocessing import get_context
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000  # Voxtral's audio encoder input rate
PIPER_SHARE = 0.7
SPEED_RANGE = (0.9, 1.1)
MAX_ATTEMPTS = 4
PIPER_SPEAKERS = 904  # en_US-libritts_r-medium; checked against the model config at startup

KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella",
    "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
MODEL_FILES = {
    "en_US-libritts_r-medium.onnx": PIPER_URL,
    "en_US-libritts_r-medium.onnx.json": PIPER_URL,
    "kokoro-v1.0.onnx": KOKORO_URL,
    "voices-v1.0.bin": KOKORO_URL,
}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
EMAIL_SYMBOLS = {".": " dot ", "_": " underscore ", "-": " dash ", "+": " plus ", "@": " at "}
TYPOGRAPHY = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": ", "}
# Whisper spells filler sounds many ways ("Hmm" came back as "Hum", "And"; "Uh" as "Ah").
# The route never depends on them, so they are dropped from both sides before comparing.
FILLERS = {"hmm", "hm", "hum", "uh", "uhm", "um", "ah", "ahh", "oh", "er", "erm", "eh", "mm", "huh"}


def spoken(text: str) -> str:
    """Write the utterance the way a person says it: emails letter by letter, & as 'and'.

    The route target keeps the written form (client@example.com); only the audio is spoken.
    """
    for mark, plain in TYPOGRAPHY.items():
        text = text.replace(mark, plain)
    text = EMAIL.sub(lambda m: " " + re.sub(r"[._+@-]", lambda s: EMAIL_SYMBOLS[s.group()], m.group()) + " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def pick_voice(clip_id: str, attempt: int) -> dict:
    """Same id and attempt always give the same voice, so a rerun reproduces the dataset."""
    rng = random.Random(f"{clip_id}:{attempt}")
    speed = round(rng.uniform(*SPEED_RANGE), 2)
    if rng.random() < PIPER_SHARE:
        return {"engine": "piper", "voice": rng.randrange(PIPER_SPEAKERS), "speed": speed}
    return {"engine": "kokoro", "voice": rng.choice(KOKORO_VOICES), "speed": speed}


def download_models(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, base_url in MODEL_FILES.items():
        if not (models_dir / name).exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(base_url + name, models_dir / f"{name}.part")
            (models_dir / f"{name}.part").replace(models_dir / name)


# ---------------------------------------------------------------- TTS worker processes

_piper = _kokoro = None


def init_worker(models_dir: str) -> None:
    """One ONNX thread per process: many short clips scale across processes, not threads."""
    global _piper, _kokoro
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    from piper import PiperVoice

    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    def session(name: str):
        return ort.InferenceSession(str(Path(models_dir) / name), sess_options=options,
                                    providers=["CPUExecutionProvider"])

    _piper = PiperVoice.load(str(Path(models_dir) / "en_US-libritts_r-medium.onnx"))
    _piper.session = session("en_US-libritts_r-medium.onnx")
    _kokoro = Kokoro.from_session(session("kokoro-v1.0.onnx"), str(Path(models_dir) / "voices-v1.0.bin"))


def synthesize(text: str, voice: dict) -> np.ndarray:
    from piper import SynthesisConfig

    if voice["engine"] == "piper":
        config = SynthesisConfig(speaker_id=voice["voice"], length_scale=1 / voice["speed"])
        chunks = list(_piper.synthesize(text, syn_config=config))
        audio, rate = np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate
    else:
        lang = "en-gb" if voice["voice"].startswith("b") else "en-us"
        audio, rate = _kokoro.create(text, voice=voice["voice"], speed=voice["speed"], lang=lang)
    audio = librosa.resample(audio.astype(np.float32), orig_sr=rate, target_sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=40)
    pad = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)  # short silence, like a real end of speech
    audio = np.concatenate([pad, audio, pad])
    return audio * (0.9 / max(float(np.abs(audio).max()), 1e-6))


def speak(task: dict) -> tuple:
    """Write one clip and hand the audio back for the Whisper check."""
    path = Path(task["path"])
    if task["reuse"] and path.exists():  # written before an interruption, never checked
        audio, _ = sf.read(path, dtype="float32")
    else:
        audio = synthesize(task["text"], task["voice"])
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    return task, audio


# ---------------------------------------------------------------- main process

def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def save_manifest(path: Path, manifest: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text("\n".join(json.dumps(r) for r in manifest.values()) + "\n", encoding="utf-8")
    temp.replace(path)


def show(records: list) -> None:
    for rec in records:
        print(f"{'OK  ' if rec['ok'] else 'MISS'} {rec['id']}  {rec['engine']}:{rec['voice']}  "
              f"x{rec['speed']}  try {rec['attempt']}")
        print(f"     text : {rec['text']}")
        print(f"     heard: {rec['heard']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=ROOT / "data" / "text" / "rows.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "audio")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="TTS processes; leave a couple of cores for Whisper and I/O")
    parser.add_argument("--chunk", type=int, default=128, help="clips per Whisper pass and manifest save")
    parser.add_argument("--whisper-batch", type=int, default=16)
    parser.add_argument("--limit", type=int, help="only the first N rows, for a smoke test")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[:args.limit] if args.limit else rows
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "voices.jsonl"
    manifest = load_manifest(manifest_path)
    download_models(args.models)
    speakers = json.loads((args.models / "en_US-libritts_r-medium.onnx.json").read_text())["num_speakers"]
    assert speakers == PIPER_SPEAKERS, f"voice has {speakers} speakers, expected {PIPER_SPEAKERS}"

    # Workers start before torch touches CUDA; spawn keeps them free of the GPU context.
    pool = get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(str(args.models),))

    import torch
    import transformers
    from transformers import pipeline

    transformers.logging.set_verbosity_error()  # otherwise repeats a max_length notice every batch
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                   dtype=torch.float16, device="cuda:0")

    def normalize(text: str) -> str:
        """Whisper's English normalizer: case, punctuation, and 'six PM' == '6 pm'.

        It leaves 'p.m.' as 'p m' and '3:30' as '3 30' while 'three thirty' becomes '330',
        so those two forms are folded first.
        """
        text = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", spoken(text), flags=re.IGNORECASE)
        text = re.sub(r"(\d):(\d\d)\b", r"\1\2", text)
        return asr.tokenizer.normalize(text)

    def matches(text: str, heard: str) -> bool:
        expected, got = normalize(text).split(), normalize(heard).split()
        if expected == got:
            return True
        if not FILLERS & set(re.findall(r"[a-z]+", text.lower())):
            return False
        expected, got = [w for w in expected if w not in FILLERS], [w for w in got if w not in FILLERS]
        if text.lower().split()[0].strip(",.") in FILLERS and got[:1] == ["and"] and expected[:1] != ["and"]:
            got = got[1:]  # a leading filler heard as "and"
        return expected == got

    def transcribe(audios: list) -> list:
        # The longest utterance is 42 words (~60 tokens); 128 stops runaway repeats and caps memory.
        with torch.inference_mode():
            outputs = asr([{"raw": a, "sampling_rate": SAMPLE_RATE} for a in audios],
                          batch_size=args.whisper_batch,
                          generate_kwargs={"language": "english", "task": "transcribe", "max_new_tokens": 128})
        return [o["text"].strip() for o in outputs]

    import jiwer

    for rec in manifest.values():  # re-score earlier checks under the current rule, no re-speaking
        rec["ok"] = matches(rec["text"], rec["heard"])
    if manifest:
        save_manifest(manifest_path, manifest)
    print(f"{len(rows)} rows, {args.workers} TTS workers, output {args.out}")
    started, previewed = time.time(), False
    while True:
        todo = [r for r in rows
                if not manifest.get(r["id"], {}).get("ok") and manifest.get(r["id"], {}).get("attempt", 0) < MAX_ATTEMPTS]
        if not todo:
            break
        print(f"\n{len(todo)} clips to speak or redo")
        tasks = []
        for row in todo:
            previous = manifest.get(row["id"])
            attempt = previous["attempt"] + 1 if previous else 1
            tasks.append({"id": row["id"], "text": spoken(row["utterance"]), "voice": pick_voice(row["id"], attempt),
                          "attempt": attempt, "reuse": previous is None,
                          "path": str(args.out / f"{row['id']}.wav")})
        results = pool.imap(speak, tasks, chunksize=2)  # workers run ahead while Whisper checks
        done_now = 0
        while batch := list(islice(results, args.chunk)):
            records = [{"id": t["id"], "text": t["text"], **t["voice"], "attempt": t["attempt"],
                        "seconds": round(len(a) / SAMPLE_RATE, 2)} for t, a in batch]
            for rec, heard in zip(records, transcribe([a for _, a in batch])):
                expected, got = normalize(rec["text"]), normalize(heard)
                rec.update(heard=heard, ok=matches(rec["text"], heard), wer=round(jiwer.wer(expected, got), 3))
                manifest[rec["id"]] = rec
            save_manifest(manifest_path, manifest)

            if not previewed:
                print("\nFirst clips: the text it was given vs what Whisper heard")
                show(records[:8])
                previewed = True
            done_now += len(records)
            elapsed = time.time() - started
            rate = done_now / elapsed
            misses = [r for r in records if not r["ok"]]
            print(f"  {done_now}/{len(todo)} this round, {len(records) - len(misses)}/{len(records)} exact, "
                  f"{rate:.1f} clips/s, ~{(len(todo) - done_now) / rate / 60:.0f} min left in this round")
            show(misses)
    pool.close()

    records = list(manifest.values())
    passed = [r for r in records if r["ok"]]
    failed = [r for r in records if not r["ok"]]
    print(f"\n{len(passed)}/{len(rows)} clips match their text exactly, "
          f"{sum(r['seconds'] for r in records) / 3600:.1f} h of audio, {(time.time() - started) / 60:.0f} min")
    if failed:
        print(f"\n{len(failed)} clips still differ after {MAX_ATTEMPTS} voices; listen before keeping them:")
        show(failed)


if __name__ == "__main__":
    main()
