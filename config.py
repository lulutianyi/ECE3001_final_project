from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TrainConfig:
    dataset_root: str = "./data/ravdess"
    split_json: str = "./splits/split_seed42.json"
    out_dir: str = "./outputs/exp1"

    target_sr: int = 16000
    clip_seconds: float = 3.0

    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400

    model_name: str = "resnet18"
    pretrained: bool = False

    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # none | cosine
    min_lr: float = 1e-5

    # Mitigate class-level prediction bias.
    class_weight_mode: str = "sqrt_inv_freq"  # none | inv_freq | sqrt_inv_freq
    label_smoothing: float = 0.05

    # Small validation set can be noisy; use EMA-smoothed val metric for model selection.
    val_ema_alpha: float = 0.6
    selection_metric: str = "val_acc_ema"  # val_acc | val_acc_ema | val_loss

    # Save multiple checkpoints for later comparison/ensembling.
    save_every_epoch: bool = True
    save_last: bool = True

    device: str = "cuda"
