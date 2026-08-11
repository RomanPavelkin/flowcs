"""Task-aware flow-matching models that generate compressed-sensing
subsampling masks conditioned on a frozen, pre-trained feature encoder.

- `CondFlow` is used for the MNIST and CelebA image experiments, where the
  conditioning signal `y` is an image and the frozen encoder is a
  DigitFeatureEncoder / CelebAFeatureEncoder.
- `FlowMatchingMaskGenerator` is used for the MRI experiment, where the
  conditioning signal is the low-frequency-sampled image and the frozen
  encoder is the CNN_denoiser inside the pre-trained MoDL model.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .embeddings import TimeEmbedding, MoDLTimeEmbedding


class CondFlow(nn.Module):
    """Flow model that generates a subsampling mask with a U-Net,
    conditioned on multiscale features from a pre-trained, frozen image
    encoder (DigitFeatureEncoder or CelebAFeatureEncoder)."""

    def __init__(self, encoder, norm_type='batch', use_batch_norm=None):
        """
        Args:
            encoder: Pre-trained encoder for feature extraction from the
                conditioning input (its enc1/enc2/enc3/bottleneck submodules
                are used directly).
            norm_type (str): 'batch'/'bn', 'layer'/'ln', 'group'/'gn', or
                'none'/None. Default: 'batch'.
            use_batch_norm (bool, optional): Deprecated. If provided,
                overrides norm_type (True -> 'batch', False -> 'none').
        """
        super().__init__()

        if use_batch_norm is not None:
            import warnings
            warnings.warn("use_batch_norm is deprecated. Use norm_type parameter instead.", DeprecationWarning)
            norm_type = 'batch' if use_batch_norm else 'none'

        self.norm_type = norm_type.lower() if isinstance(norm_type, str) else 'none'

        self.time_emb = TimeEmbedding(64)

        # Frozen, pre-trained U-Net encoder used for feature extraction from y
        self.y_encoder = encoder

        self.time_proj_bottleneck = nn.Linear(64, 256)

        self._build_encoder_decoder()

    def _get_norm_layer(self, num_channels):
        if self.norm_type in ['batch', 'bn']:
            return nn.BatchNorm2d(num_channels)
        elif self.norm_type in ['layer', 'ln']:
            return nn.GroupNorm(1, num_channels)  # GroupNorm(1, C) == LayerNorm over (C,H,W)
        elif self.norm_type in ['group', 'gn']:
            num_groups = min(8, num_channels)
            while num_channels % num_groups != 0:
                num_groups -= 1
            return nn.GroupNorm(num_groups, num_channels)
        else:
            return nn.Identity()

    def _build_encoder_decoder(self):
        # === Z ENCODER PATH (mirrors the frozen encoder's structure) ===
        self.z_enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            self._get_norm_layer(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            self._get_norm_layer(32),
            nn.LeakyReLU(0.2)
        )
        self.z_enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            self._get_norm_layer(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            self._get_norm_layer(64),
            nn.LeakyReLU(0.2)
        )
        self.z_enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            self._get_norm_layer(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            self._get_norm_layer(128),
            nn.LeakyReLU(0.2)
        )
        self.z_bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            self._get_norm_layer(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            self._get_norm_layer(256),
            nn.LeakyReLU(0.2)
        )

        # === CONDITIONING PROJECTION LAYERS (project y features onto z) ===
        self.y_cond_skip1 = nn.Conv2d(32, 32, 1)
        self.y_cond_skip2 = nn.Conv2d(64, 64, 1)
        self.y_cond_skip3 = nn.Conv2d(128, 128, 1)
        self.y_cond_bottleneck = nn.Conv2d(256, 256, 1)

        # === Z DECODER PATH ===
        self.z_up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.z_dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            self._get_norm_layer(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            self._get_norm_layer(128),
            nn.LeakyReLU(0.2)
        )
        self.z_up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.z_dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            self._get_norm_layer(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            self._get_norm_layer(64),
            nn.LeakyReLU(0.2)
        )
        self.z_up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.z_dec3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            self._get_norm_layer(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            self._get_norm_layer(32),
            nn.LeakyReLU(0.2)
        )

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, z, t, y):
        batch_size = z.shape[0]

        t_emb = self.time_emb(t)

        h, w = y.shape[2:]
        z_img = z.view(batch_size, 1, h, w)

        # === EXTRACT MULTISCALE FEATURES FROM Y (conditioning input) ===
        y_skip1 = self.y_encoder.enc1(y)
        y_skip2 = self.y_encoder.enc2(y_skip1)
        y_skip3 = self.y_encoder.enc3(y_skip2)
        y_bottleneck = self.y_encoder.bottleneck(y_skip3)

        # === ENCODE Z, CONDITIONED BY Y AT EACH SCALE ===
        z_skip1 = self.z_enc1(z_img)
        z_skip1 = z_skip1 + self.y_cond_skip1(y_skip1)

        z_skip2 = self.z_enc2(z_skip1)
        z_skip2 = z_skip2 + self.y_cond_skip2(y_skip2)

        z_skip3 = self.z_enc3(z_skip2)
        z_skip3 = z_skip3 + self.y_cond_skip3(y_skip3)

        z_bn = self.z_bottleneck(z_skip3)

        h_bn, w_bn = z_bn.shape[2:]
        z_bn = z_bn + self.y_cond_bottleneck(y_bottleneck)

        t_emb_bottleneck = self.time_proj_bottleneck(t_emb)
        t_emb_spatial = t_emb_bottleneck.view(batch_size, 256, 1, 1).expand(-1, -1, h_bn, w_bn)
        z_bn = z_bn + t_emb_spatial

        # === DECODE Z BACK TO ORIGINAL DIMENSIONS ===
        x = self.z_up1(z_bn)
        x = F.interpolate(x, size=z_skip3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, z_skip3], dim=1)
        x = self.z_dec1(x)

        x = self.z_up2(x)
        x = torch.cat([x, z_skip2], dim=1)
        x = self.z_dec2(x)

        x = self.z_up3(x)
        x = torch.cat([x, z_skip1], dim=1)
        x = self.z_dec3(x)

        x = self.out_conv(x)
        return x


class CNN_denoiser_feature_wrapper(nn.Module):
    """Wraps a pre-trained CNN_denoiser to expose its intermediate layer
    features for use as conditioning signals."""

    def __init__(self, cnn_denoiser):
        super().__init__()
        self.denoiser = cnn_denoiser

        total_layers = len(self.denoiser.nw)
        # (n_layers - 1) conv_blocks * 3 layers + 2 final layers = total_layers
        self.n_layers = (total_layers - 2) // 3 + 1

        first_conv = None
        for module in self.denoiser.nw:
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
        self.features_dim = first_conv.out_channels if first_conv else 64
        self.in_ch = first_conv.in_channels if first_conv else 2

        last_conv = None
        for module in reversed(list(self.denoiser.nw)):
            if isinstance(module, nn.Conv2d):
                last_conv = module
                break
        self.out_ch = last_conv.out_channels if last_conv else 2

    def forward(self, x, return_features=False):
        """
        Args:
            x: Input tensor [B, C, H, W]
            return_features: If True, also return the list of intermediate
                feature maps (one per conv block, excluding the output layer).
        Returns:
            output, or (output, features) if return_features=True.
        """
        if not return_features:
            return self.denoiser(x)

        idt = x
        features = []
        h = x

        layer_idx = 0
        conv_block_size = 3

        for _ in range(self.n_layers - 1):
            for _ in range(conv_block_size):
                h = self.denoiser.nw[layer_idx](h)
                layer_idx += 1
            features.append(h)

        h = self.denoiser.nw[layer_idx](h)      # Conv2d
        h = self.denoiser.nw[layer_idx + 1](h)  # BatchNorm2d

        output = h + idt
        return output, features


class FlowMatchingMaskGenerator(nn.Module):
    """
    Flow-matching model that generates 1D k-space sampling masks conditioned
    on features pooled from a pre-trained, frozen CNN_denoiser.

    Image -> CNN encoder -> pooled multiscale features -> MLP([x_t, features,
    time embedding, positional encoding]) -> 1D velocity field.
    """

    def __init__(self, mask_size, cnn_denoiser, hidden_dim=512, time_emb_dim=64, pos_emb_dim=64):
        """
        Args:
            mask_size: Size of the (flattened) mask vector.
            cnn_denoiser: Pre-trained CNN_denoiser instance used as a frozen
                encoder.
            hidden_dim: Hidden dimension of the MLP layers.
            time_emb_dim: Dimension fed into the sinusoidal time embedding.
            pos_emb_dim: Dimension of the positional embedding.
        """
        super().__init__()

        self.cnn_encoder = cnn_denoiser

        # Locate all Conv2d layers; the CNN structure is:
        # [conv_block1, conv_block2, ..., final_Conv2d, final_BatchNorm2d]
        # where each conv_block is [Conv2d, BatchNorm2d, ReLU].
        conv_layers = []
        conv_indices = []  # index right after each Conv2d block (post-activation)

        for i, module in enumerate(self.cnn_encoder.nw):
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
                if i + 2 < len(self.cnn_encoder.nw) and isinstance(self.cnn_encoder.nw[i + 2], nn.ReLU):
                    conv_indices.append(i + 3)  # after Conv2d, BatchNorm, ReLU
                else:
                    conv_indices.append(i + 1)  # final layer: after Conv2d only

        # Extract features from all intermediate layers (exclude final output layer)
        self.conv_indices = conv_indices[:-1]

        self.cnn_features_dim = sum(layer.out_channels for layer in conv_layers[:-1])

        self.mask_size = mask_size
        self.hidden_dim = hidden_dim
        self.pos_emb_dim = pos_emb_dim

        self.time_emb = MoDLTimeEmbedding(dim=time_emb_dim)

        self.register_buffer("pos_encoding", self._create_positional_encoding(mask_size, pos_emb_dim))

        input_dim = mask_size + self.cnn_features_dim + time_emb_dim + pos_emb_dim

        self.mlp_layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.mlp_layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.mlp_layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.mlp_layer4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.mlp_output = nn.Linear(hidden_dim, mask_size)

    def _create_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x_t, t, image):
        """
        Args:
            x_t: Noisy mask at time t, shape [B, mask_size]
            t: Time parameter, shape [B, 1] or [B]
            image: Conditioning image batch, shape [B, C, H, W]
        Returns:
            v_t: Predicted velocity field, shape [B, mask_size]
        """
        batch_size = x_t.shape[0]

        with torch.no_grad():
            h = image
            intermediate_features = []
            prev_idx = 0

            for checkpoint_idx in self.conv_indices:
                for i in range(prev_idx, checkpoint_idx):
                    h = self.cnn_encoder.nw[i](h)
                pooled = h.mean(dim=(-2, -1))
                intermediate_features.append(pooled)
                prev_idx = checkpoint_idx

        pooled_features = torch.cat(intermediate_features, dim=-1)

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_emb(t)

        pos_emb = self.pos_encoding.mean(dim=0).unsqueeze(0).expand(batch_size, -1)

        mlp_input = torch.cat([x_t, pooled_features, t_emb, pos_emb], dim=-1)

        h = self.mlp_layer1(mlp_input)
        h = self.mlp_layer2(h)
        h = self.mlp_layer3(h)
        h = self.mlp_layer4(h)

        v_t = self.mlp_output(h)
        return v_t
