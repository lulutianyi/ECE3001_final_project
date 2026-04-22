from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset import RAVDESSpeakerID, AudioConfig
from fe import STFTConfig, STFTSpectrogram
from models import create_model

def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)

@torch.no_grad()
def evaluate_on_split(dataset_root: str, split_json: str, split: str, ckpt_path: str, batch_size: int = 64, num_workers: int = 2, device: str = "cpu") -> Dict[str, Any]:
    device = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    audio_cfg = AudioConfig(**ckpt["audio_cfg"])

    ds = RAVDESSpeakerID(dataset_root, split_json, split, audio_cfg, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feat = ckpt["feature"]
    fe = STFTSpectrogram(STFTConfig(n_fft=int(feat["n_fft"]), hop_length=int(feat["hop_length"]), win_length=int(feat["win_length"]))).to(device)
    fe.eval()

    model = create_model(ckpt["model_name"], num_classes=int(ckpt["num_classes"]), in_channels=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true, y_pred = [], []
    for wav, y, _meta in loader:
        wav = wav.to(device)
        x = fe(wav)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return {"acc": acc, "confusion_matrix": cm, "report": report, "actors": ds.actors}
