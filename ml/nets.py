"""Network definitions shared by the Colab notebook and the backend.

  MultiChannelIntensityNet - EfficientNet-B0 on IR + WV + PMW with availability masks
  SeqIntensityNet          - ConvLSTM over the last 8 IR/WV frames (21 h) -> 24 h wind change + RI
  BasinDetector            - ResNet-18 FPN heatmap detector (CenterNet-style) for TC / depression centres
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MC_CHANNELS = ["ir", "wv", "pmw", "wv_mask", "pmw_mask"]
DET_CLASSES = ["cyclone", "depression"]  # IMD >= 34 kt, 17-33 kt


class MultiChannelIntensityNet(nn.Module):
    def __init__(self, n_cls=7, pretrained=False, arch="efficientnet_b0", drop=0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=pretrained, in_chans=len(MC_CHANNELS), num_classes=0)
        d = self.backbone.num_features
        self.drop = nn.Dropout(drop)
        self.head_v, self.head_t, self.head_c = nn.Linear(d, 1), nn.Linear(d, 1), nn.Linear(d, n_cls)

    def heads(self, pooled):
        f = self.drop(pooled)
        return self.head_v(f).squeeze(1), self.head_t(f).squeeze(1), self.head_c(f)

    def forward(self, x):
        return self.heads(self.backbone(x))


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, ch, k=3):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(cin + ch, 4 * ch, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class SeqIntensityNet(nn.Module):
    """Input (B, T, 2, 64, 64) IR/WV frames + (B, 2) scalars [current wind/100, past-24h change/30].
    Output: 24 h wind change (kt/20) and RI logit."""

    def __init__(self, cin=2, hidden=64, n_scalar=2, use_images=True):
        super().__init__()
        self.use_images = use_images
        self.enc = nn.Sequential(
            nn.Conv2d(cin, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU())
        self.cell = ConvLSTMCell(64, hidden)
        self.post = nn.Sequential(nn.Conv2d(hidden, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        d = (64 if use_images else 0) + n_scalar
        self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, frames, scalars):
        feats = [scalars]
        if self.use_images:
            b, t = frames.shape[:2]
            z = self.enc(frames.flatten(0, 1)).unflatten(0, (b, t))
            h = torch.zeros(b, self.cell.ch, *z.shape[-2:], device=z.device)
            c = torch.zeros_like(h)
            for k in range(t):
                h, c = self.cell(z[:, k], (h, c))
            feats.insert(0, self.post(h))
        out = self.head(torch.cat(feats, 1))
        return out[:, 0], out[:, 1]


class BasinDetector(nn.Module):
    """Heatmap detector: (B, 2, H, W) IR/WV scene -> (B, n_cls, H/4, W/4) centre logits."""

    def __init__(self, n_cls=len(DET_CLASSES), pretrained=False):
        super().__init__()
        import timm
        self.body = timm.create_model("resnet18", pretrained=pretrained, in_chans=2, features_only=True, out_indices=(1, 2, 3, 4))
        chs = self.body.feature_info.channels()  # strides 4, 8, 16, 32
        self.lat = nn.ModuleList([nn.Conv2d(c, 96, 1) for c in chs])
        self.head = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.GELU(), nn.Conv2d(96, n_cls, 1))
        self.head[-1].bias.data.fill_(-2.2)  # CenterNet prior

    def forward(self, x):
        fs = self.body(x)
        y = self.lat[-1](fs[-1])
        for f, l in zip(reversed(fs[:-1]), reversed(self.lat[:-1])):
            y = F.interpolate(y, size=f.shape[-2:], mode="nearest") + l(f)
        return self.head(y)


def heatmap_peaks(logits, thr=0.3, k=3):
    """Local maxima above thr. logits (C, H, W) -> list of (cls, y, x, score)."""
    p = torch.sigmoid(logits)
    keep = (F.max_pool2d(p[None], k, 1, k // 2)[0] == p) & (p >= thr)
    idx = keep.nonzero().tolist()
    return [(c, y, x, float(p[c, y, x])) for c, y, x in idx]
