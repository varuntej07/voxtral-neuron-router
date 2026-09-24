"""Speak the user turns of data/text/rows.jsonl as 16 kHz clips, on Colab.

Paste this whole file into one Colab cell (Runtime > Change runtime type > T4 GPU) and run it.
If rows.jsonl is not already in DRIVE_DIR, the cell asks you to upload it.

Each clip gets a seeded random voice: PIPER_SHARE from Piper libritts_r (904 speakers), the rest
from Kokoro. Whisper large-v3 then transcribes every clip back, and a clip passes only if the
transcript matches its text exactly after normalization. Misses are redone with a new voice, up
to MAX_ATTEMPTS, and whatever still misses is printed with a player so you can listen to it.

Engines: piper-tts 1.8 and kokoro-onnx 0.6.1 (Kokoro-82M v1.0 on onnxruntime). The `kokoro`
package itself caps Python below 3.13, which Colab no longer has. Both TTS engines run on CPU,
and Whisper runs on the T4.

Clips, voices.jsonl and the model files all live on Google Drive and are saved after each
chunk, so after a disconnect just run the cell again and it picks up where it stopped. When it
finishes, download DRIVE_DIR/audio.zip and unzip it into data/audio/ (clip names are
<row id>.wav, matching data/sft/*.jsonl).
"""

import subprocess
import sys


def pip_install(*args: str) -> bool:
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    if result.returncode:
        print(f"pip install {' '.join(args)} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result.returncode == 0


if not pip_install("piper-tts>=1.8", "kokoro-onnx>=0.6.1", "soundfile", "librosa", "jiwer"):
    raise SystemExit("pip failed; the error is printed above")

import gc
import json
import logging
import random
import re
import shutil
import urllib.request
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
import transformers
from google.colab import drive, files
from IPython.display import Audio, display
from kokoro_onnx import Kokoro
from piper import PiperVoice, SynthesisConfig
from transformers import pipeline

# phonemizer warns "words count mismatch" on nearly every line; the Whisper check covers real errors.
logging.getLogger("phonemizer").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()  # otherwise repeats a max_length notice every batch

DRIVE_DIR = Path("/content/drive/MyDrive/voxtral-audio")
AUDIO_DIR = DRIVE_DIR / "audio"
MODELS_DIR = DRIVE_DIR / "models"
MANIFEST = AUDIO_DIR / "voices.jsonl"
ROWS = DRIVE_DIR / "rows.jsonl"

SAMPLE_RATE = 16000  # Voxtral's audio encoder input rate
PIPER_SHARE = 0.7
SPEED_RANGE = (0.9, 1.1)
MAX_ATTEMPTS = 4
CHUNK = 64
PREVIEW = 8

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

# ---------------------------------------------------------------- text as it should be spoken

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
EMAIL_SYMBOLS = {".": " dot ", "_": " underscore ", "-": " dash ", "+": " plus ", "@": " at "}
TYPOGRAPHY = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": ", "}


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


# ---------------------------------------------------------------- engines

drive.mount("/content/drive")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
if not ROWS.exists():
    print("Upload data/text/rows.jsonl")
    shutil.move(next(iter(files.upload())), ROWS)
rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"{len(rows)} rows")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
for name, base_url in MODEL_FILES.items():
    if not (MODELS_DIR / name).exists():
        print(f"downloading {name}")
        urllib.request.urlretrieve(base_url + name, MODELS_DIR / f"{name}.part")
        (MODELS_DIR / f"{name}.part").replace(MODELS_DIR / name)

piper = PiperVoice.load(str(MODELS_DIR / "en_US-libritts_r-medium.onnx"))
PIPER_SPEAKERS = piper.config.num_speakers

kokoro = Kokoro(str(MODELS_DIR / "kokoro-v1.0.onnx"), str(MODELS_DIR / "voices-v1.0.bin"))
missing = sorted(set(KOKORO_VOICES) - set(kokoro.get_voices()))
KOKORO_VOICES = [v for v in KOKORO_VOICES if v not in missing]
print(f"piper speakers: {PIPER_SPEAKERS}, kokoro voices: {len(KOKORO_VOICES)}"
      + (f" (not in this model file: {missing})" if missing else ""))

# Rerunning the cell in the same session: drop the previous Whisper and the crashed run's
# traceback (which keeps its GPU tensors alive) before loading again.
globals().pop("asr", None)
sys.last_type = sys.last_value = sys.last_traceback = None
gc.collect()
torch.cuda.empty_cache()
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


# Whisper spells filler sounds many ways ("Hmm" came back as "Hum", "And"; "Uh" as "Ah").
# The route never depends on them, so they are dropped from both sides before comparing.
FILLERS = {"hmm", "hm", "hum", "uh", "uhm", "um", "ah", "ahh", "oh", "er", "erm", "eh", "mm", "huh"}


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


def pick_voice(clip_id: str, attempt: int) -> dict:
    """Same id and attempt always give the same voice, so a rerun reproduces the dataset."""
    rng = random.Random(f"{clip_id}:{attempt}")
    speed = round(rng.uniform(*SPEED_RANGE), 2)
    if rng.random() < PIPER_SHARE:
        return {"engine": "piper", "voice": rng.randrange(PIPER_SPEAKERS), "speed": speed}
    return {"engine": "kokoro", "voice": rng.choice(KOKORO_VOICES), "speed": speed}


def synthesize(text: str, voice: dict) -> np.ndarray:
    if voice["engine"] == "piper":
        config = SynthesisConfig(speaker_id=voice["voice"], length_scale=1 / voice["speed"])
        chunks = list(piper.synthesize(text, syn_config=config))
        audio, rate = np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate
    else:
        lang = "en-gb" if voice["voice"].startswith("b") else "en-us"
        audio, rate = kokoro.create(text, voice=voice["voice"], speed=voice["speed"], lang=lang)
    audio = librosa.resample(audio.astype(np.float32), orig_sr=rate, target_sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=40)
    pad = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)  # short silence, like a real end of speech
    audio = np.concatenate([pad, audio, pad])
    return audio * (0.9 / max(float(np.abs(audio).max()), 1e-6))


def transcribe(audios: list) -> list:
    # batch 16 with Whisper's default 448-token limit ran out of the T4's 15 GB. The longest
    # utterance is 42 words (~60 tokens), so 128 tokens is plenty and stops runaway repeats.
    with torch.inference_mode():
        outputs = asr([{"raw": a, "sampling_rate": SAMPLE_RATE} for a in audios], batch_size=4,
                      generate_kwargs={"language": "english", "task": "transcribe", "max_new_tokens": 128})
    torch.cuda.empty_cache()
    return [o["text"].strip() for o in outputs]


# ---------------------------------------------------------------- progress on Drive

def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, MANIFEST.read_text(encoding="utf-8").splitlines())}


def save_manifest(manifest: dict) -> None:
    temp = MANIFEST.with_suffix(".tmp")
    temp.write_text("\n".join(json.dumps(r) for r in manifest.values()) + "\n", encoding="utf-8")
    temp.replace(MANIFEST)


def show(records: list) -> None:
    for rec in records:
        print(f"{'OK  ' if rec['ok'] else 'MISS'} {rec['id']}  {rec['engine']}:{rec['voice']}  "
              f"x{rec['speed']}  try {rec['attempt']}")
        print(f"     text : {rec['text']}")
        print(f"     heard: {rec['heard']}")


# ---------------------------------------------------------------- speak, check, redo misses

manifest = load_manifest()
for rec in manifest.values():  # re-score earlier checks under the current rule, no re-speaking
    rec["ok"] = matches(rec["text"], rec["heard"])
if manifest:
    save_manifest(manifest)
previewed = False
while True:
    todo = [r for r in rows
            if not manifest.get(r["id"], {}).get("ok") and manifest.get(r["id"], {}).get("attempt", 0) < MAX_ATTEMPTS]
    if not todo:
        break
    print(f"\n{len(todo)} clips to speak or redo")
    for start in range(0, len(todo), CHUNK):
        audios, records = [], []
        for row in todo[start:start + CHUNK]:
            path = AUDIO_DIR / f"{row['id']}.wav"
            previous = manifest.get(row["id"])
            attempt = previous["attempt"] + 1 if previous else 1
            text, voice = spoken(row["utterance"]), pick_voice(row["id"], attempt)
            if previous is None and path.exists():  # written before a disconnect, never checked
                audio, _ = sf.read(path, dtype="float32")
            else:
                audio = synthesize(text, voice)
                sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
            audios.append(audio)
            records.append({"id": row["id"], "text": text, **voice, "attempt": attempt,
                            "seconds": round(len(audio) / SAMPLE_RATE, 2)})
        for rec, heard in zip(records, transcribe(audios)):
            expected, got = normalize(rec["text"]), normalize(heard)
            rec.update(heard=heard, ok=matches(rec["text"], heard), wer=round(jiwer.wer(expected, got), 3))
            manifest[rec["id"]] = rec
        save_manifest(manifest)

        if not previewed:
            print("\nFirst clips: the text it was given vs what Whisper heard")
            show(records[:PREVIEW])
            for rec in records[:3]:
                display(Audio(str(AUDIO_DIR / f"{rec['id']}.wav")))
            previewed = True
        misses = [r for r in records if not r["ok"]]
        done = sum(r["ok"] for r in manifest.values())
        print(f"  chunk {start // CHUNK + 1}: {len(records) - len(misses)}/{len(records)} exact, "
              f"{done}/{len(rows)} verified overall")
        show(misses)

# ---------------------------------------------------------------- report and package

records = list(manifest.values())
passed = [r for r in records if r["ok"]]
failed = [r for r in records if not r["ok"]]
hours = sum(r["seconds"] for r in records) / 3600
print(f"\n{len(passed)}/{len(rows)} clips match their text exactly, {hours:.1f} h of audio")
print(f"engines among passing clips: piper {sum(r['engine'] == 'piper' for r in passed)}, "
      f"kokoro {sum(r['engine'] == 'kokoro' for r in passed)}")
if failed:
    print(f"\n{len(failed)} clips still differ after {MAX_ATTEMPTS} voices; listen before keeping them:")
    show(failed)
    for rec in failed[:10]:
        display(Audio(str(AUDIO_DIR / f"{rec['id']}.wav")))

shutil.make_archive(str(DRIVE_DIR / "audio"), "zip", AUDIO_DIR)
print(f"\nDownload {DRIVE_DIR / 'audio.zip'} from Google Drive and unzip it into data/audio/")
