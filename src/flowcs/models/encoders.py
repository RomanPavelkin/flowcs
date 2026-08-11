"""U-Net encoder/decoder models used both as task autoencoders and as
multiscale feature extractors that condition the flow-matching mask
generator (CondFlow)."""
import torch
from torch import nn
import torch.nn.functional as F


class DigitFeatureEncoder(nn.Module):
    """
    U-Net encoder/decoder for MNIST digit reconstruction and feature extraction.

    Extracts multiscale features at different resolutions:
    - skip1: 32 channels at 28x28 (fine details)
    - skip2: 64 channels at 14x14 (medium features)
    - skip3: 128 channels at 7x7 (coarse features)
    - bottleneck: 256 channels at 3x3 (highest-level features)

    These features are later used by CondFlow for task-aware sampling.
    """

    def __init__(self):
        super().__init__()

        # === ENCODER PATH ===
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # === BOTTLENECK ===
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        # === DECODER PATH ===
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(32, 1, 1)

    def freeze_parameters(self):
        """Freeze all parameters (used when this encoder serves as a frozen
        feature extractor for CondFlow after pre-training)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """Unfreeze all parameters for training or fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 1, 28, 28]
        Returns:
            Reconstructed image [B, 1, 28, 28]
        """
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)

        x = self.bottleneck(skip3)

        x = self.up1(x)
        x = F.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3(x)

        x = self.out_conv(x)
        return x


class CelebAFeatureEncoder(nn.Module):
    """Same U-Net architecture as DigitFeatureEncoder, adapted for 3-channel
    RGB CelebA images."""

    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(32, 3, 1)

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, 128, 128]
        Returns:
            Reconstructed image [B, 3, 128, 128]
        """
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)

        x = self.bottleneck(skip3)

        x = self.up1(x)
        x = F.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3(x)

        x = self.out_conv(x)
        return x
