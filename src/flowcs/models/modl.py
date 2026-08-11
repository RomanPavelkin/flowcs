"""CNN denoiser and unrolled MoDL (Model-based Deep Learning) reconstruction
network for accelerated single-coil MRI (Experiment 3)."""
import torch
from torch import nn

from ..mri.ops import A_forward, A_adj


def conv_block(in_channels, out_channels):
    """Basic convolutional block: Conv -> BatchNorm -> ReLU."""
    return [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]


class CNN_denoiser(nn.Module):
    """Simple CNN denoiser with a residual (identity) connection, used both
    as the MoDL regularizer and, once pre-trained, as the frozen encoder for
    the flow-matching mask generator."""

    def __init__(self, n_layers=5, in_ch=2, out_ch=2, features=64):
        """
        Args:
            n_layers (int): Total number of convolutional layers.
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            features (int): Number of hidden features.
        """
        super().__init__()
        layers = []
        layers += conv_block(in_ch, features)
        for _ in range(n_layers - 2):
            layers += conv_block(features, features)
        layers += [
            nn.Conv2d(features, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        ]
        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x
        dw = self.nw(x) + idt
        return dw


def matvec_AHA_plus_lambda(x, mask, lam):
    """Applies M(x) = A^H A x + lam * x for the CG solve below."""
    k = A_forward(x, mask)
    Ahk = A_adj(k, mask)
    return Ahk + lam * x


def cg_solve(b, x0, mask, lam, cg_iters=8, tol=1e-9):
    """
    Differentiable conjugate-gradient solver for the symmetric
    positive-definite system M x = b, where M is given implicitly by
    matvec_AHA_plus_lambda.
    """
    x = x0.clone()
    x = x.to(b.dtype).requires_grad_(True)
    r = b - matvec_AHA_plus_lambda(x, mask, lam)
    p = r.clone()

    rsold = (r * r).sum(dim=(-3, -2, -1), keepdim=True)

    for _ in range(cg_iters):
        Ap = matvec_AHA_plus_lambda(p, mask, lam)
        denom = (p * Ap).sum(dim=(-3, -2, -1), keepdim=True)

        alpha = rsold / (denom + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = (r * r).sum(dim=(-3, -2, -1), keepdim=True)
        if torch.sqrt(rsnew).mean() < tol:
            break

        p = r + (rsnew / (rsold + 1e-12)) * p
        rsold = rsnew

    return x


class MoDL_SingleCoilMRI_acceleration(nn.Module):
    """Unrolled MoDL reconstruction network for single-coil accelerated MRI:
    alternates a learned CNN denoiser with a data-consistency CG solve."""

    def __init__(self, denoiser=None, num_iters=6, cg_iters=8, lam_init=0.01,
                 depth=5, in_ch=2, out_ch=2, features=32):
        super().__init__()

        if denoiser is not None:
            self.denoiser = denoiser
        else:
            self.denoiser = CNN_denoiser(in_ch=in_ch, out_ch=out_ch, features=features, depth=depth)

        self.num_iters = num_iters
        self.cg_iters = cg_iters
        self.lams = nn.ParameterList([nn.Parameter(torch.tensor(lam_init)) for _ in range(num_iters)])

    def forward(self, k0, mask):
        """
        Args:
            k0: Measured (undersampled) k-space, shape (..., 2, H, W)
            mask: Sampling mask, shape (..., H, W)
        Returns:
            Reconstructed image, shape (..., 2, H, W)
        """
        x = A_adj(k0, mask)

        for i in range(self.num_iters):
            z = self.denoiser(x)
            lam = torch.abs(self.lams[i])
            b0 = A_adj(k0, mask)
            b = b0 + lam * z
            x = cg_solve(b, x, mask, lam, cg_iters=self.cg_iters)
        return x
