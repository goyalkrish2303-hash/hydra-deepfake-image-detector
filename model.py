"""Hydra: 3-head deepfake / AI-image detector.
Head 1 (pixels)    : ResNet-50 transfer learning
Head 2 (frequency) : small CNN on the log-FFT spectrum
Head 3 (file)      : EXIF / Content-Credentials check (rule-based, see file_check.py)
Heads 1+2 are fused (Dense 512 + Dropout -> Sigmoid).
"""
import torch
import torch.nn as nn
from torchvision import models

IMG = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def fft_map(x):
    """x: (B,3,H,W) normalised -> (B,1,H,W) standardised log-magnitude spectrum."""
    with torch.autocast(device_type=x.device.type, enabled=False):  # cuFFT needs fp32 for 224x224
        g = x.float().mean(1, keepdim=True)
        f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1))
        m = torch.log1p(f.abs())
        m = (m - m.mean(dim=(-2, -1), keepdim=True)) / (m.std(dim=(-2, -1), keepdim=True) + 1e-6)
    return m


class SpatialHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        r = models.resnet50(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.layer4 = r.layer4
        for p in self.stem.parameters():
            p.requires_grad = False

    def forward(self, x):
        fmap = self.layer4(self.stem(x))
        return fmap, fmap.mean((2, 3))


class FreqHead(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 128),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(fft_map(x))


class HydraNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.spatial = SpatialHead(pretrained)
        self.freq = FreqHead()
        self.fusion = nn.Sequential(nn.Linear(2048 + 128, 512), nn.ReLU(inplace=True),
                                    nn.Dropout(0.4), nn.Linear(512, 1))

    def forward(self, x, return_fmap=False):
        fmap, s = self.spatial(x)
        q = self.freq(x)
        logit = self.fusion(torch.cat([s, q], 1)).squeeze(1)
        return (logit, fmap) if return_fmap else logit

    def head_probs(self, x):
        """Diagnostic ablation: shared fusion score with the other feature stream zeroed."""
        with torch.no_grad():
            _, s = self.spatial(x)
            q = self.freq(x)
            z0s, z0q = torch.zeros_like(s), torch.zeros_like(q)
            ps = torch.sigmoid(self.fusion(torch.cat([s, z0q], 1))).squeeze(1)
            pf = torch.sigmoid(self.fusion(torch.cat([z0s, q], 1))).squeeze(1)
        return ps, pf
