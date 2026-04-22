from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class STFTConfig:
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    power: float = 2.0
    log_eps: float = 1e-10
    center: bool = True

class STFTSpectrogram(torch.nn.Module):
    """STFT log-spectrogram.
    Input: waveform [B, 1, T] or [B, T]
    Output: spec [B, 1, F, TT]
    """
    def __init__(self, cfg: STFTConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute log power spectrogram.

        TODO(student): implement with torch.stft(return_complex=True)
        """
        # ===== TODO(student) START =====
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        if wav.dim() != 2:
            raise ValueError(f"Expected [B,T] or [B,1,T], got {wav.shape}")

        device = wav.device
        window = self.window.to(device=device, dtype=wav.dtype)

        stft = torch.stft(
            wav,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=self.cfg.center,
            return_complex=True,
        )
        mag = stft.abs()
        spec = mag.pow(self.cfg.power) if self.cfg.power != 1.0 else mag
        spec = torch.log(spec + self.cfg.log_eps)
        return spec.unsqueeze(1)
        # ===== TODO(student) END =====
