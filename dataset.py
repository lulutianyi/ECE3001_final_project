from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import random
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

# RAVDESS filename convention:
# Modality - VocalChannel - Emotion - EmotionalIntensity - Statement - Repetition - ActorID.wav
# Example: 03-01-01-01-01-01-01.wav
# For speaker identification we ONLY need ActorID (01-24).

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

MODALITY_MAP = {"01": "full-AV", "02": "video-only", "03": "audio-only"}
VOCAL_CHANNEL_MAP = {"01": "speech", "02": "song"}

def parse_ravdess_filename(name: str) -> Dict[str, str]:
    """Parse RAVDESS filename into metadata.

    TODO(student):
      - Split by '-' and validate there are 7 fields.
      - Return a dict that includes 'actor' (ActorID) at minimum.
    """
    # ===== TODO(student) START =====
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected filename format: {name}")
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    return {
        "modality": modality,
        "modality_str": MODALITY_MAP.get(modality, modality),
        "vocal_channel": vocal_channel,
        "vocal_channel_str": VOCAL_CHANNEL_MAP.get(vocal_channel, vocal_channel),
        "emotion": emotion,
        "emotion_str": EMOTION_MAP.get(emotion, emotion),
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
    }
    # ===== TODO(student) END =====

def _find_actor_root(dataset_root: str | Path) -> Path:
    """Auto-detect where Actor_01 ... Actor_24 live.

    Accept either:
      A) <root>/Actor_01 ... Actor_24
      B) <root>/audio_speech_actors_01-24/Actor_01 ... Actor_24

    TODO(student): implement the detection logic.
    """
    # ===== TODO(student) START =====
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"DATASET_ROOT not found: {root}")

    nested = root / "audio_speech_actors_01-24"
    if nested.exists() and any(nested.glob("Actor_*")):
        return nested

    if any(root.glob("Actor_*")):
        return root

    for p in root.iterdir():
        if p.is_dir() and any(p.glob("Actor_*")):
            return p

    raise FileNotFoundError(
        """Could not find Actor_* folders. Expected either:
 - <root>/Actor_01 ... Actor_24
 - <root>/audio_speech_actors_01-24/Actor_01 ... Actor_24"""
    )
    # ===== TODO(student) END =====

def list_wavs(dataset_root: str | Path) -> List[Path]:
    actor_root = _find_actor_root(dataset_root)
    return sorted(actor_root.glob("Actor_*/**/*.wav"))

def make_splits(
    dataset_root: str | Path,
    out_json: str | Path,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, Any]:
    """Create per-actor stratified splits (train/val/test) and save as JSON."""
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1
    actor_root = _find_actor_root(dataset_root)
    rng = random.Random(seed)

    actor_dirs = sorted([p for p in actor_root.iterdir() if p.is_dir() and p.name.startswith("Actor_")])
    actors = [d.name.split("_")[1] for d in actor_dirs]
    actors_sorted = sorted(actors)

    splits = {"seed": seed, "dataset_root": str(actor_root), "actors": actors_sorted, "train": [], "val": [], "test": []}

    for actor_id in actors_sorted:
        files = sorted((actor_root / f"Actor_{actor_id}").glob("*.wav"))
        files = [f for f in files if f.is_file()]
        rng.shuffle(files)

        n = len(files)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = max(1, n - n_test - n_val)

        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:n_train+n_val+n_test]

        def rel(p: Path) -> str:
            return str(p.relative_to(actor_root))

        splits["train"].extend([{"path": rel(p), "actor": actor_id} for p in train_files])
        splits["val"].extend([{"path": rel(p), "actor": actor_id} for p in val_files])
        splits["test"].extend([{"path": rel(p), "actor": actor_id} for p in test_files])

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits

@dataclass
class AudioConfig:
    target_sr: int = 16000
    clip_seconds: float = 3.0
    mono: bool = True
    normalize: bool = True

class RAVDESSpeakerID(torch.utils.data.Dataset):
    def __init__(self, dataset_root: str | Path, split_json: str | Path, split: str, audio_cfg: AudioConfig, is_train: bool = False) -> None:
        self.actor_root = _find_actor_root(dataset_root)
        self.audio_cfg = audio_cfg
        self.is_train = is_train

        data = json.loads(Path(split_json).read_text(encoding="utf-8"))
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of train/val/test")
        self.items = data[split]

        self.actors = sorted({it["actor"] for it in self.items})
        self.actor_to_index = {a: i for i, a in enumerate(self.actors)}
        self.target_len = int(audio_cfg.target_sr * audio_cfg.clip_seconds)

    def __len__(self) -> int:
        return len(self.items)

    def _load_wav(self, rel_path: str) -> torch.Tensor:
        """Load wav and return waveform [1, target_len].

        TODO(student): implement
          - torchaudio.load
          - mono
          - resample to target_sr
          - normalize
          - crop/pad
        """
        # ===== TODO(student) START =====
        # Normalize separators so split files generated on Windows/Linux both work.
        rel_path_norm = rel_path.replace("\\", "/")
        wav_path = self.actor_root / rel_path_norm

        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception:
            # Fallback for environments where torchaudio backend is unavailable.
            sr, wav_np = wavfile.read(str(wav_path))
            if wav_np.ndim == 1:
                wav_np = wav_np[:, np.newaxis]
            if np.issubdtype(wav_np.dtype, np.integer):
                wav_np = wav_np.astype(np.float32) / np.iinfo(wav_np.dtype).max
            else:
                wav_np = wav_np.astype(np.float32)
            # scipy returns [T, C], convert to [C, T]
            waveform = torch.from_numpy(wav_np).transpose(0, 1).contiguous()

        if self.audio_cfg.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.audio_cfg.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.audio_cfg.target_sr)

        if self.audio_cfg.normalize:
            peak = waveform.abs().max().clamp(min=1e-8)
            waveform = waveform / peak

        T = waveform.shape[1]
        if T >= self.target_len:
            if self.is_train:
                start = random.randint(0, T - self.target_len)
            else:
                start = (T - self.target_len) // 2
            waveform = waveform[:, start:start + self.target_len]
        else:
            pad = self.target_len - T
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform
        # ===== TODO(student) END =====

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, str]]:
        it = self.items[idx]
        rel_path = it["path"]
        actor_id = it["actor"]
        y = self.actor_to_index[actor_id]
        x = self._load_wav(rel_path)
        meta = parse_ravdess_filename(Path(rel_path).name)
        return x, y, meta
