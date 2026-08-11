"""Complex-valued FFT helpers and the single-coil forward/adjoint MRI
operators (A, A^H) used by MoDL and the flow-based mask generator."""
import torch
import torch.fft as fft


def to_complex_tensor(x):
    """Interpret a real tensor with shape (..., 2, H, W) as complex parts:
    x[...,0]+i*x[...,1]."""
    return x[..., 0, :, :].contiguous(), x[..., 1, :, :].contiguous()


def from_complex_tensor(re, im):
    """Stack real and imaginary parts into shape (..., 2, H, W)."""
    return torch.stack([re, im], dim=-3)


def fft2c(x):
    """Centered 2D FFT."""
    return fft.fftshift(fft.fft2(fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(x):
    """Centered 2D inverse FFT."""
    return fft.fftshift(fft.ifft2(fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def rfft2c(re, im):
    """Centered 2D FFT on separate real/imaginary tensors."""
    c = torch.complex(re, im)
    k = fft.fftshift(fft.fft2(fft.ifftshift(c, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    return k.real, k.imag


def rifft2c(k_re, k_im):
    """Centered 2D inverse FFT on separate real/imaginary tensors."""
    c = torch.complex(k_re, k_im)
    im = fft.fftshift(fft.ifft2(fft.ifftshift(c, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    return im.real, im.imag


def normalize_complex_image(img, eps=1e-11):
    """Normalize each complex image in a batch by its max magnitude.

    Args:
        img: (batch, 2, H, W) tensor where the channel dim is [real, imag].
    Returns:
        (img_norm, scale_factors)
    """
    batch_size = img.shape[0]
    img_norm = torch.zeros_like(img)
    scale_factors = torch.zeros(batch_size, device=img.device)

    for i in range(batch_size):
        magnitude = torch.sqrt(img[i, 0, :, :] ** 2 + img[i, 1, :, :] ** 2 + eps)
        max_val = magnitude.max()
        if max_val > eps:
            img_norm[i] = img[i] / max_val
            scale_factors[i] = max_val
        else:
            img_norm[i] = img[i]
            scale_factors[i] = 1.0

    return img_norm, scale_factors


def unnormalize_complex_image(img_norm, scale_factors):
    """Inverse of normalize_complex_image, per-sample in the batch."""
    batch_size = img_norm.shape[0]
    img = torch.zeros_like(img_norm)
    for i in range(batch_size):
        img[i] = img_norm[i] * scale_factors[i]
    return img


def A_forward(x, mask):
    """Single-coil forward operator: image -> masked k-space.

    Args:
        x: (..., 2, H, W) real/imag image channels
        mask: (..., H, W) 0/1 sampling mask (broadcastable)
    Returns:
        k-space (..., 2, H, W)
    """
    re, im = to_complex_tensor(x)
    k_re, k_im = rfft2c(re, im)
    k_re = k_re * mask
    k_im = k_im * mask
    return from_complex_tensor(k_re, k_im)


def A_adj(k, mask):
    """Single-coil adjoint operator: masked k-space -> image.

    Args:
        k: (..., 2, H, W) k-space real/imag channels
        mask: (..., H, W) sampling mask
    Returns:
        image (..., 2, H, W)
    """
    k_re, k_im = to_complex_tensor(k)
    k_re = k_re * mask
    k_im = k_im * mask
    re, im = rifft2c(k_re, k_im)
    return from_complex_tensor(re, im)
