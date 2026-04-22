from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RAVDESSpeakerID, AudioConfig
from fe import STFTConfig, STFTSpectrogram
from models import create_model
from utils import set_seed, ensure_dir, top1_accuracy

def _get_device(device_pref: str) -> torch.device:
    if device_pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_pref)
    return torch.device("cpu")

def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def _build_class_weights(train_ds: RAVDESSpeakerID, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = Counter(train_ds.actor_to_index[it["actor"]] for it in train_ds.items)
    num_classes = len(train_ds.actors)
    freq = torch.tensor([counts.get(i, 1) for i in range(num_classes)], dtype=torch.float32)
    if mode == "inv_freq":
        weights = 1.0 / freq.clamp(min=1.0)
    elif mode == "sqrt_inv_freq":
        weights = 1.0 / torch.sqrt(freq.clamp(min=1.0))
    else:
        raise ValueError("class_weight_mode must be one of: none, inv_freq, sqrt_inv_freq")

    # Keep expected scale stable across options.
    weights = weights / weights.mean().clamp(min=1e-8)
    return weights.to(device)

def _save_checkpoint(
    save_path: Path,
    cfg: Any,
    model: torch.nn.Module,
    num_classes: int,
    audio_cfg: AudioConfig,
    train_ds: RAVDESSpeakerID,
    history: Dict[str, list],
    epoch: int,
) -> None:
    torch.save({
        "model_name": str(_get(cfg, "model_name")),
        "num_classes": num_classes,
        "state_dict": model.state_dict(),
        "audio_cfg": audio_cfg.__dict__,
        "feature": {
            "type": "log-stft",
            "n_fft": int(_get(cfg, "n_fft")),
            "hop_length": int(_get(cfg, "hop_length")),
            "win_length": int(_get(cfg, "win_length")),
        },
        "actor_to_index": train_ds.actor_to_index,
        "history": history,
        "epoch": epoch,
    }, save_path)

def run_training(cfg: Any) -> Dict[str, Any]:
    """Train a speaker ID classifier (best checkpoint by val accuracy)."""
    set_seed(int(_get(cfg, "seed", 42)))
    out_dir = ensure_dir(_get(cfg, "out_dir"))
    device = _get_device(str(_get(cfg, "device", "cuda")))
    print(f"Device: {device}")

    audio_cfg = AudioConfig(
        target_sr=int(_get(cfg, "target_sr", 16000)),
        clip_seconds=float(_get(cfg, "clip_seconds", 3.0)),
        mono=True,
        normalize=True,
    )

    train_ds = RAVDESSpeakerID(_get(cfg, "dataset_root"), _get(cfg, "split_json"), "train", audio_cfg, is_train=True)
    val_ds   = RAVDESSpeakerID(_get(cfg, "dataset_root"), _get(cfg, "split_json"), "val", audio_cfg, is_train=False)
    num_classes = len(train_ds.actors)

    train_loader = DataLoader(train_ds, batch_size=int(_get(cfg, "batch_size", 32)), shuffle=True,
                              num_workers=int(_get(cfg, "num_workers", 2)), pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=int(_get(cfg, "batch_size", 32)), shuffle=False,
                            num_workers=int(_get(cfg, "num_workers", 2)), pin_memory=(device.type=="cuda"))

    fe = STFTSpectrogram(STFTConfig(
        n_fft=int(_get(cfg, "n_fft", 512)),
        hop_length=int(_get(cfg, "hop_length", 160)),
        win_length=int(_get(cfg, "win_length", 400)),
    )).to(device)
    fe.eval()

    model = create_model(str(_get(cfg, "model_name", "resnet18")), num_classes=num_classes, in_channels=1, pretrained=bool(_get(cfg, "pretrained", False))).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(_get(cfg, "lr", 1e-3)), weight_decay=float(_get(cfg, "weight_decay", 1e-4)))
    scheduler = None
    if str(_get(cfg, "lr_scheduler", "none")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=max(1, int(_get(cfg, "epochs", 10))),
            eta_min=float(_get(cfg, "min_lr", 1e-5)),
        )

    class_weight_mode = str(_get(cfg, "class_weight_mode", "none")).lower()
    class_weights = _build_class_weights(train_ds, class_weight_mode, device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(_get(cfg, "label_smoothing", 0.0)),
    )

    best_val_acc = -1.0
    best_path = out_dir / "best.pt"
    best_metric = float("-inf")
    val_ema = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_acc_ema": [], "lr": []}

    for epoch in range(1, int(_get(cfg, "epochs", 10)) + 1):
        # ---- train ----
        model.train()
        tr_loss, tr_acc, nb = 0.0, 0.0, 0
        for wav, y, _meta in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                x = fe(wav)

            logits = model(x)
            loss = criterion(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            tr_loss += float(loss.item())
            tr_acc += top1_accuracy(logits.detach(), y)
            nb += 1

        train_loss = tr_loss / max(1, nb)
        train_acc = tr_acc / max(1, nb)

        # ---- val ----
        model.eval()
        va_loss, va_acc, vb = 0.0, 0.0, 0
        with torch.no_grad():
            for wav, y, _meta in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                wav = wav.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = fe(wav)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += float(loss.item())
                va_acc += top1_accuracy(logits, y)
                vb += 1

        val_loss = va_loss / max(1, vb)
        val_acc = va_acc / max(1, vb)
        if val_ema is None:
            val_ema = val_acc
        else:
            alpha = float(_get(cfg, "val_ema_alpha", 0.6))
            val_ema = alpha * val_ema + (1.0 - alpha) * val_acc

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_acc_ema"].append(val_ema)
        history["lr"].append(float(optim.param_groups[0]["lr"]))

        print(
            f"[Epoch {epoch}] train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} ema={val_ema:.4f} lr={history['lr'][-1]:.2e}"
        )

        if bool(_get(cfg, "save_every_epoch", True)):
            _save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", cfg, model, num_classes, audio_cfg, train_ds, history, epoch)

        selection_metric = str(_get(cfg, "selection_metric", "val_acc_ema")).lower()
        if selection_metric == "val_loss":
            current_metric = -val_loss
        elif selection_metric == "val_acc":
            current_metric = val_acc
        elif selection_metric == "val_acc_ema":
            current_metric = val_ema
        else:
            raise ValueError("selection_metric must be one of: val_acc, val_acc_ema, val_loss")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if current_metric > best_metric:
            best_metric = current_metric
            _save_checkpoint(best_path, cfg, model, num_classes, audio_cfg, train_ds, history, epoch)
            print(f"Saved best checkpoint to: {best_path} (val_acc={best_val_acc:.4f})")

        if bool(_get(cfg, "save_last", True)):
            _save_checkpoint(out_dir / "last.pt", cfg, model, num_classes, audio_cfg, train_ds, history, epoch)

        if scheduler is not None:
            scheduler.step()

    return {"best_val_acc": best_val_acc, "best_path": str(best_path), "history": history}
