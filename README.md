# Flow-Based Generative Modeling for Optimizing Sampling Policies in Compressed Sensing Applications

Code for the paper *"Flow-Based Generative Modeling for Optimizing Sampling
Policies in Compressed Sensing Applications."* [https://doi.org/10.48550/arXiv.2606.00078]

## Abstract

> Numerous modern applications in signal processing and medical imaging
> necessitate acquiring high-dimensional signals under tight resource
> constraints. Traditional sampling theory suggests that accurate signal
> reconstruction requires a number of measurements proportional to the
> signal's ambient dimension, a requirement often too expensive or
> impractical. Compressed sensing challenges this notion by demonstrating
> that sparse signals can be recovered with fewer measurements, provided the
> measurement operator meets certain conditions. This proof-of-concept study
> presents a task-aware flow-based generative framework — a reformulation of
> the conventional Flow Matching training paradigm with a flow model trained
> to optimize subsampling in compressed sensing applications. We establish
> the fundamental feasibility of the proposed framework of learning
> subsampling masks that improve the performance of compressed sensing for
> image classification, image reconstruction, and MRI acceleration. For the
> image reconstruction task, our method achieved Peak Signal-to-Noise Ratio
> of 25.17 dB (improvement of 33% over the SotA) at the subsampling rate of
> 5% on the CelebA dataset and 29.24 dB of PSNR and 0.7163 of SSIM
> (improvement of 23% over the SotA) when reconstructing 8x accelerated MRI
> measurements (fastMRI dataset) with minimal computational overhead. These
> results highlight the effectiveness of task-conditioning within generative
> flow models and reveal a promising direction for representation learning
> strategies. Overall, the proposed framework offers a unified, flexible
> approach to designing data- and task-driven sensing schemes that can be
> potentially adapted to a broad range of inverse problems.

## Overview

The core idea is a **task-aware flow-matching model** that learns to
generate compressed-sensing subsampling masks conditioned on a frozen,
pre-trained task model (a classifier, an autoencoder, or an MRI
reconstruction network). The repository reproduces three experiments from
the paper:

| # | Experiment | Task model | Dataset |
|---|---|---|---|
| 1 | MNIST classification & reconstruction | `DigitClassifier` / `DigitFeatureEncoder` (U-Net AE) | MNIST (via `sklearn.datasets.fetch_openml`) |
| 2 | Image reconstruction | `CelebAFeatureEncoder` (U-Net AE) | CelebA |
| 3 | Accelerated MRI reconstruction | MoDL (unrolled CNN denoiser + CG data consistency) | fastMRI (single-coil knee) |

Each experiment trains a random-subsampling baseline task model, a greedy
sensor-selection (GSE) mask baseline, and the flow-matching mask generator,
then evaluates reconstruction/classification quality under each sampling
strategy.

## Repository structure

```
.
├── src/flowcs/              # Installable shared library
│   ├── models/               # Encoders, classifier, flow-matching models, MoDL
│   ├── mri/                  # FFT operators, single-coil A/A^H, fastMRI h5 I/O
│   ├── data/                 # CelebA dataset/dataloader
│   └── utils/                # Reproducibility helpers (seeding)
├── experiments/
│   ├── exp1_mnist.py         # Experiment 1: MNIST classification & reconstruction
│   ├── exp2_celeba.py        # Experiment 2: CelebA reconstruction
│   └── exp3_mri.py           # Experiment 3: fastMRI acceleration
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Setup

Requires Python >= 3.9 and a CUDA-capable GPU (recommended; CPU will work
but training will be slow).

```bash
git clone <this-repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
pip install -e .    # installs the `flowcs` package used by the experiment scripts
```

`fastmri` and `torchdiffeq` are required only for Experiments 3 and
1-3 (Experiment 3 uses `fastmri`; all three use `torchdiffeq` for the
flow-matching ODE integration).

### Datasets

None of the datasets are bundled with this repository.

- **MNIST** (Experiment 1) is downloaded automatically via
  `sklearn.datasets.fetch_openml` on first run.
- **CelebA** (Experiment 2): download the aligned & cropped images from
  the [official CelebA page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  and point `--celeba-train-dir` / `--celeba-test-dir` at folders of
  `.jpg`/`.png` images (178x218 px; the loader center-crops to 128x128).
- **fastMRI** (Experiment 3): request access and download the single-coil
  knee dataset from [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu/),
  then point `--fastmri-train-dir` / `--fastmri-val-dir` at the
  `singlecoil_train` / `singlecoil_val` folders of `.h5` files.

## Usage

Each experiment is a single, self-contained script that runs all training
and evaluation stages from the paper end-to-end (pre-training the task
model, training the GSE baseline, training the flow-matching mask
generator, and evaluating every sampling strategy on the test set).
Checkpoints, learned masks/logits, and result figures are written to
`--output-dir`.

**Experiment 1 — MNIST classification & reconstruction**
```bash
python experiments/exp1_mnist.py --output-dir checkpoints/mnist
```

**Experiment 2 — CelebA reconstruction**
```bash
python experiments/exp2_celeba.py \
    --celeba-train-dir /path/to/celeba/images \
    --celeba-test-dir /path/to/celeba/test_set \
    --output-dir checkpoints/celeba
```

**Experiment 3 — fastMRI acceleration**
```bash
python experiments/exp3_mri.py \
    --fastmri-train-dir /path/to/fastMRI/knee/singlecoil_train \
    --fastmri-val-dir /path/to/fastMRI/knee/singlecoil_val \
    --output-dir checkpoints/mri
```

All three scripts accept `--seed` (default `42`) for reproducibility.
Full runs are long (hours on a single GPU) since each script reproduces
every training stage in the paper; comment out later stages in the script
if you only need a subset (e.g. only the flow-matching training, assuming
checkpoints for the pre-trained task model already exist in `--output-dir`).

## Using the shared library directly

The `flowcs` package can also be used outside of the experiment scripts,
e.g. to load a pre-trained model for downstream analysis:

```python
import torch
from flowcs.models import DigitFeatureEncoder, CondFlow

encoder = DigitFeatureEncoder()
encoder.load_state_dict(torch.load("checkpoints/mnist/unet_mnist32by32_ae_random-samp_v9.pth"))
encoder.freeze_parameters()

flow_model = CondFlow(encoder=encoder)
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{flowcs2026,
  title   = {Flow-Based Generative Modeling for Optimizing Sampling Policies in Compressed Sensing Applications},
  author  = {Roman Pavelkin, Luis A. Zavala-Mondragon, Christiaan G. A. Viviers, Fons van der Sommen},
  journal = {arXiv preprint https://doi.org/10.48550/arXiv.2606.00078},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
