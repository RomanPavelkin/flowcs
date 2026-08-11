"""
Experiment 3: Accelerated single-coil MRI reconstruction (fastMRI knee) with
task-aware flow-matching sampling masks.

Reproduces the MRI results from the paper "Flow-Based Generative Modeling
for Optimizing Sampling Policies in Compressed Sensing Applications".

This script:
  1. Loads single-coil fastMRI knee k-space volumes (.h5 files) from local
     train/val directories.
  2. Pre-trains (warm-up) and fine-tunes an unrolled MoDL reconstruction
     network (CNN denoiser + differentiable conjugate-gradient
     data-consistency) under randomly-sampled Cartesian k-space masks.
  3. Trains the task-aware flow-matching mask generator
     (FlowMatchingMaskGenerator), conditioned on the frozen pre-trained
     MoDL denoiser, to learn an accelerated sampling pattern.
  4. Evaluates reconstruction quality (PSNR/SSIM) and per-slice inference
     latency of the learned sampling masks on the held-out validation set.

You must supply local paths to fastMRI singlecoil knee train/val
directories (this dataset requires registration; see
https://fastmri.med.nyu.edu/).

Checkpoints and figures are written to ``--output-dir``
(default: ``checkpoints/mri``).

Usage:
    python experiments/exp3_mri.py \\
        --fastmri-train-dir /path/to/fastMRI/knee/singlecoil_train \\
        --fastmri-val-dir /path/to/fastMRI/knee/singlecoil_val \\
        --output-dir checkpoints/mri
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from torchmetrics.functional.image import structural_similarity_index_measure

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

from flowcs.models import CNN_denoiser, MoDL_SingleCoilMRI_acceleration, FlowMatchingMaskGenerator
from flowcs.mri import A_forward, A_adj, normalize_complex_image, load_volume_kspaces, create_kspace_dataloader
from flowcs.utils import set_seed


def run_experiment(args):
    CKPT_DIR = args.output_dir
    os.makedirs(CKPT_DIR, exist_ok=True)

    set_seed(args.seed)


    # ==================================
    # EXPERIMENT 3: MRI acceleration
    # ==================================



    # ======================
    # Create dataloaders
    # ======================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    batch_size = 4

    crop_size = (128, 128) # Nolan 2024
    acceleration = 4 # Nolan 2024
    center_fraction = 0.08 # Nolan 2024
    # crop_size = (208, 208) # van Gorp 2021
    # acceleration = 8 # van Gorp 2021
    # center_fraction = 0.04 # van Gorp 2021

    num_low_freqs = int(crop_size[0]*center_fraction)
    version = f'9_acc_{acceleration}'

    # Create sets and data loaders
    train_set_path = args.fastmri_train_dir
    val_set_path = args.fastmri_val_dir
    # test_set_path = r'C:\MY_FILES\TUe_job\AI_FORSchung_project\Datasets\MRI\fastMRI\knee\singlecoil_test'

    train_loader = create_kspace_dataloader(
        data_dir=train_set_path,
        batch_size=batch_size,
        shuffle=True,
        transform=None,
        crop_size=crop_size
    )

    val_loader = create_kspace_dataloader(
        data_dir=val_set_path,
        batch_size=batch_size,
        shuffle=False,
        transform=None,
        crop_size=crop_size
    )

    # test_loader = create_kspace_dataloader(
    #     data_dir=val_set_path,
    #     batch_size=1,
    #     shuffle=False,
    #     transform=None
    # )


    # Plot examples from the datasets

    print("\nDataLoader Example:")

    example_train_batch = next(iter(train_loader))
    example_val_batch = next(iter(val_loader))
    # example_test_batch = next(iter(test_loader))

    print("Train Set Example:")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(np.log(fastmri.complex_abs(example_train_batch[0]).cpu().numpy() + 1e-9), cmap='viridis')
    plt.title('Train K-space (log magnitude)')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 2, 2)
    train_image = fastmri.ifft2c(example_train_batch)
    train_image_abs = fastmri.complex_abs(train_image)
    plt.imshow(train_image_abs[0].cpu().numpy(), cmap='gray')
    plt.title('Train Reconstructed Image')
    plt.colorbar()
    plt.axis('off')

    plt.suptitle('Train Set - Example Sample', fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"Train batch shape: {example_train_batch.shape}")

    print("\nValidation Set Example:")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(np.log(fastmri.complex_abs(example_val_batch[0]).cpu().numpy() + 1e-9), cmap='viridis')
    plt.title('Validation K-space (log magnitude)')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 2, 2)
    val_image = fastmri.ifft2c(example_val_batch)
    val_image_abs = fastmri.complex_abs(val_image)
    plt.imshow(val_image_abs[0].cpu().numpy(), cmap='gray')
    plt.title('Validation Reconstructed Image')
    plt.colorbar()
    plt.axis('off')

    plt.suptitle('Validation Set - Example Sample', fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"Validation batch shape: {example_val_batch.shape}")

    # print("\nTest Set Example:")
    # plt.figure(figsize=(10, 4))

    # plt.subplot(1, 2, 1)
    # plt.imshow(np.log(fastmri.complex_abs(example_test_batch[0]).cpu().numpy() + 1e-9), cmap='viridis')
    # plt.title('Test K-space (log magnitude)')
    # plt.colorbar()
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # test_image = fastmri.ifft2c(example_test_batch)
    # test_image_abs = fastmri.complex_abs(test_image)
    # plt.imshow(test_image_abs[0].cpu().numpy(), cmap='gray')
    # plt.title('Test Reconstructed Image')
    # plt.colorbar()
    # plt.axis('off')

    # plt.suptitle('Test Set - Example Sample', fontsize=14)
    # plt.tight_layout()
    # plt.show()

    # print(f"Test batch shape: {example_test_batch.shape}")



    # ============================
    # Pre-train the MoDL model
    # ============================

    # Set the model parameters
    depth = 4 # 5 in the original paper
    num_features = 32 # 64 in the original paper
    K_warmup = 1 # Number of unrolled iterations for the warm-up phase
    cg_iters = 8 # 10 # Number of CG iterations
    lam_init = 0.05 # Initial value for lambda
    sigma = 0.01 # the noise level in the Fourier domain
    n_steps = 5*((21*973)//batch_size) # num. of passes x total. num. of samples (central slice +-20/2 x num of files) // batch_size
    train_losses = []
    val_losses = []
    lam_values = []

    print(f"Number of steps for warm-up MoDL training: {n_steps:,}")

    offset = 0

    # Initialize the denoising model
    denoiser = CNN_denoiser(n_layers=depth,
                                in_ch=2,
                                out_ch=2,
                                features=num_features).to(device)

    # Load pre-trained weights from checkpoint
    # denoiser.load_state_dict(torch.load(f'supplementary\denoiser_warmup_v8{version}_{offset}.pth', map_location=device))

    # Initialize the warm-up MoDL model
    modl_warmup = MoDL_SingleCoilMRI_acceleration(denoiser=denoiser, 
                                                  num_iters=K_warmup,
                                                  cg_iters=cg_iters,
                                                  lam_init=lam_init).to(device)

    # Load pre-trained weights from checkpoint
    # modl_warmup.load_state_dict(torch.load(f'supplementary\modl_mri-acc_warmup_v{version}_{offset}.pth', map_location=device))

    # Print number of trainable parameters in the denoiser and MoDL model
    num_params_denoiser = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    num_params_modl_warmup = sum(p.numel() for p in modl_warmup.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the denoiser: {num_params_denoiser:,}")
    print(f"Number of trainable parameters in the warm-up MoDL model: {num_params_modl_warmup:,}")

    # Instantiate the optimizer for the warm-up phase
    optimizer = torch.optim.Adam(modl_warmup.parameters(), lr=1e-4)

    # Prevent the system from going to sleep

    try:

        # Training loop for warm-up phase
        for step in np.arange(offset, n_steps):

            # Sample a batch from the DataLoaders
            batch_kspaces = next(iter(train_loader))
            batch_kspaces_val = next(iter(val_loader))

            # Convert the full k-space to image domain using inverse FFT
            batch_images = A_adj(batch_kspaces.permute(0,3,1,2).to(device), torch.ones((batch_size, crop_size[0], crop_size[1]), device=device))
            batch_images_val = A_adj(batch_kspaces_val.permute(0,3,1,2).to(device), torch.ones((batch_size, crop_size[0], crop_size[1]), device=device))

            # Normalize the complex image channels
            batch_images, _ = normalize_complex_image(batch_images)
            batch_images_val, _ = normalize_complex_image(batch_images_val)

            # Create a random Gaussian sampling mask and apply it to k-spaces
            mask_func = RandomMaskFunc(center_fractions=[center_fraction], accelerations=[acceleration])  # Create the mask function object
            _, mask, _ = T.apply_mask(batch_kspaces[0], mask_func)   # Apply the mask to k-space
            mask = mask.repeat(1, 1, crop_size[1]).permute(0, 2, 1).to(device)
            _, mask_val, _ = T.apply_mask(batch_kspaces_val[0], mask_func)   # Apply the mask to k-space
            mask_val = mask_val.repeat(1, 1, crop_size[1]).permute(0, 2, 1).to(device)

            batch_kspaces_sampled = torch.zeros((batch_size, crop_size[0], crop_size[1], 2), device=device)
            batch_kspaces_val_sampled = torch.zeros((batch_size, crop_size[0], crop_size[1], 2), device=device)

            for i in range(batch_size):

                # Training batch k-space sampling
                kspace_sampled = A_forward(batch_images[i].unsqueeze(0).to(device), mask.squeeze(0)).squeeze(0).permute(1,2,0)

                # Add noise as % of the magnitude for real and imaginary channels separately
                kspace_real = kspace_sampled[:, :, 0]
                kspace_imag = kspace_sampled[:, :, 1]

                # Calculate % of magnitude for each channel
                noise_std_real = sigma * torch.std(torch.abs(kspace_real))
                noise_std_imag = sigma * torch.std(torch.abs(kspace_imag))

                # Add noise to each channel with its respective standard deviation
                noise_real = noise_std_real * torch.randn_like(kspace_real)
                noise_imag = noise_std_imag * torch.randn_like(kspace_imag)

                kspace_sampled = torch.stack((kspace_real + noise_real, kspace_imag + noise_imag), dim=-1)

                batch_kspaces_sampled[i] = kspace_sampled.to(device)

                # Validation batch k-space sampling
                kspace_sampled_val = A_forward(batch_images_val[i].unsqueeze(0).to(device), mask_val.squeeze(0)).squeeze(0).permute(1,2,0)

                # Add noise as % of the magnitude for real and imaginary channels separately
                kspace_real_val = kspace_sampled_val[:, :, 0]
                kspace_imag_val = kspace_sampled_val[:, :, 1]

                # Calculate % of magnitude for each channel
                noise_std_real_val = sigma * torch.std(torch.abs(kspace_real_val))
                noise_std_imag_val = sigma * torch.std(torch.abs(kspace_imag_val))

                # Add noise to each channel with its respective standard deviation
                noise_real_val = noise_std_real_val * torch.randn_like(kspace_real_val)
                noise_imag_val = noise_std_imag_val * torch.randn_like(kspace_imag_val)

                kspace_sampled_val[:, :, 0] += noise_real_val
                kspace_sampled_val[:, :, 1] += noise_imag_val

                batch_kspaces_val_sampled[i] = kspace_sampled_val.to(device)

            batch_kspaces_sampled = batch_kspaces_sampled.permute(0, 3, 1, 2)
            batch_kspaces_val_sampled = batch_kspaces_val_sampled.permute(0, 3, 1, 2)


            # Forward pass
            batch_images_reconstructed = modl_warmup(batch_kspaces_sampled, mask)

            with torch.no_grad():
                batch_images_val_reconstructed = modl_warmup(batch_kspaces_val_sampled, mask_val)

            # Compute MSE loss
            loss = F.mse_loss(batch_images_reconstructed, batch_images)

            # Compute validation loss without gradients
            with torch.no_grad():
                val_loss = F.mse_loss(batch_images_val_reconstructed, batch_images_val)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())        
            val_losses.append(val_loss.item())

            # Store λ values and loss for visualization
            lam_values_step = [torch.abs(lam).item() for lam in modl_warmup.lams]
            lam_values.append(lam_values_step)

            # Print the progress
            if (step + 1) % (n_steps//20) == 0:
                print(f"[{(step+1)/n_steps:.2%}], Train Loss: {loss.item():.4e}, Val Loss: {val_loss.item():.4e}")

                # Save new checkpoints
                torch.save(modl_warmup.state_dict(), f'{CKPT_DIR}/modl_mri-acc_warmup_v{version}_{step}.pth')
                torch.save(denoiser.state_dict(), f'{CKPT_DIR}/denoiser_warmup_v{version}_{step}.pth')

            # Clear memory to prevent memory leak
            if step < n_steps - 1:
                del batch_kspaces, batch_kspaces_val, batch_kspaces_sampled, batch_kspaces_val_sampled
                del batch_images, batch_images_val, batch_images_reconstructed, batch_images_val_reconstructed

            torch.cuda.empty_cache()


    finally:
        # Allow sleep again
        pass



    # Plot the losses and lambda evolution during warm-up training phase

    # Calculate moving average window size
    window_size = max(1, len(train_losses) // 100)  # 1% of total epochs, minimum 1

    # Calculate moving averages for losses
    warmup_losses_ma = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
    warmup_val_losses_ma = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
    offset = window_size // 2
    x_ma = np.arange(offset, offset + len(warmup_losses_ma))

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.3, linewidth=0.8)
    plt.plot(val_losses, label='Validation Loss', color='orange', alpha=0.3, linewidth=0.8)
    plt.plot(x_ma, warmup_losses_ma, label=f'Training Loss MA (window={window_size})', color='darkblue', linewidth=2)
    plt.plot(x_ma, warmup_val_losses_ma, label=f'Validation Loss MA (window={window_size})', color='darkorange', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Warm-up Loss Evolution")
    plt.legend()
    # plt.xscale('log')
    plt.yscale('log')

    # Display loss reduction for the last 100 epochs or all epochs
    initial_loss_all = train_losses[0]
    final_loss_all = train_losses[-1]
    loss_reduction_all = ((initial_loss_all - final_loss_all) / initial_loss_all) * 100

    initial_val_loss_all = val_losses[0]
    final_val_loss_all = val_losses[-1]
    val_loss_reduction_all = ((initial_val_loss_all - final_val_loss_all) / initial_val_loss_all) * 100

    plt.text(0.02, 0.98, f'Train loss reduction (all {len(train_losses)}): {loss_reduction_all:.1f}%\nVal loss reduction (all {len(val_losses)}): {val_loss_reduction_all:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    print(f"Training loss reduction for all {len(train_losses)} epochs: {loss_reduction_all:.1f}%")
    print(f"Validation loss reduction for all {len(val_losses)} epochs: {val_loss_reduction_all:.1f}%")
    # plt.savefig(f'{CKPT_DIR}/warmup_loss_evolution' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the lambda values evolution
    lam_values = np.array(lam_values)
    plt.figure(figsize=(10, 6))
    for i in range(lam_values.shape[1]):
        plt.plot(lam_values[:, i], label=f'λ_{i+1}')
    plt.xlabel("Epoch")
    plt.ylabel("λ value")
    plt.title("Evolution of λ values during Warm-up Training")
    plt.legend()
    plt.yscale('log')
    # plt.savefig(f'{CKPT_DIR}/warmup_lambda_evolution' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()



    # ====================================================
    # Save the warm-up trained MoDL model and denoiser
    # ====================================================
    torch.save(modl_warmup.state_dict(), f'{CKPT_DIR}/long_modl_mri-acc_warmup_v' + version + '.pth')
    torch.save(denoiser.state_dict(), f'{CKPT_DIR}/long_denoiser_warmup_v' + version + '.pth')


    # ==========================================================
    # Fine-tune the MoDL model with more unrolled iterations
    # ==========================================================

    # Set the model parameters
    depth = 4 # 5 in the original paper
    num_features = 32 # 64 in the original paper
    K = 10 # Number of unrolled iterations for fine-tuning
    cg_iters = 8 # 10 # Number of CG iterations
    lam_init = 0.05 # Initial value for lambda
    sigma = 0.01 # the noise level in the Fourier domain
    n_steps = 200000 # 10*((21*973)//batch_size) # num. of passes x total. num. of samples (central slice +-20/2 x num of files) // batch_size
    train_losses = []
    val_losses = []
    lam_values = []

    print(f"Number of steps for fine-tune MoDL training: {n_steps:,}")

    offset = 0

    # Initialize the denoising model
    denoiser = CNN_denoiser(n_layers=depth,
                                in_ch=2,
                                out_ch=2,
                                features=num_features).to(device)

    denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/long_denoiser_warmup_v' + version + '.pth', map_location=device))

    # Initialize the fine-tuning MoDL model
    modl_finetune = MoDL_SingleCoilMRI_acceleration(denoiser=denoiser, 
                                                   num_iters=K,
                                                   cg_iters=cg_iters,
                                                   lam_init=lam_init).to(device)

    num_params_denoiser = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    num_params_modl = sum(p.numel() for p in modl_finetune.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the denoiser: {num_params_denoiser:,}")
    print(f"Number of trainable parameters in the fine-tuning MoDL model: {num_params_modl:,}")

    # Load the pre-trained weights
    # modl_finetune.load_state_dict(torch.load(r'supplementary\modl_mri-acc_warmup_v' + version + '.pth', map_location=device))

    # Instantiate the optimizer for the fine-tuning phase
    optimizer = torch.optim.Adam(modl_finetune.parameters(), lr=1e-4)

    # Prevent the system from going to sleep

    try:

        # Training loop for fine-tuning phase
        for step in np.arange(offset, n_steps):

            # Sample a batch from the DataLoaders
            batch_kspaces = next(iter(train_loader))
            batch_kspaces_val = next(iter(val_loader))

            # Convert the full k-space to image domain using inverse FFT
            batch_images = A_adj(batch_kspaces.permute(0,3,1,2).to(device), torch.ones((batch_size, crop_size[0], crop_size[1]), device=device))
            batch_images_val = A_adj(batch_kspaces_val.permute(0,3,1,2).to(device), torch.ones((batch_size, crop_size[0], crop_size[1]), device=device))

            # Normalize the complex image channels
            batch_images, _ = normalize_complex_image(batch_images)
            batch_images_val, _ = normalize_complex_image(batch_images_val)

            # Create a random Gaussian sampling mask and apply it to k-spaces
            mask_func = RandomMaskFunc(center_fractions=[center_fraction], accelerations=[acceleration])  # Create the mask function object
            _, mask, _ = T.apply_mask(batch_kspaces[0], mask_func)   # Apply the mask to k-space
            mask = mask.repeat(1, 1, crop_size[1]).permute(0, 2, 1).to(device)
            _, mask_val, _ = T.apply_mask(batch_kspaces_val[0], mask_func)   # Apply the mask to k-space
            mask_val = mask_val.repeat(1, 1, crop_size[1]).permute(0, 2, 1).to(device)

            batch_kspaces_sampled = torch.zeros((batch_size, crop_size[0], crop_size[1], 2), device=device)
            batch_kspaces_val_sampled = torch.zeros((batch_size, crop_size[0], crop_size[1], 2), device=device)

            for i in range(batch_size):

                # Training batch k-space sampling
                kspace_sampled = A_forward(batch_images[i].unsqueeze(0).to(device), mask.squeeze(0)).squeeze(0).permute(1,2,0)

                # Add noise as % of the magnitude for real and imaginary channels separately
                kspace_real = kspace_sampled[:, :, 0]
                kspace_imag = kspace_sampled[:, :, 1]
                noise_std_real = sigma * torch.std(torch.abs(kspace_real))
                noise_std_imag = sigma * torch.std(torch.abs(kspace_imag))
                noise_real = noise_std_real * torch.randn_like(kspace_real)
                noise_imag = noise_std_imag * torch.randn_like(kspace_imag)
                kspace_sampled = torch.stack((kspace_real + noise_real, kspace_imag + noise_imag), dim=-1)

                batch_kspaces_sampled[i] = kspace_sampled.to(device)

                # Validation batch k-space sampling
                kspace_sampled_val = A_forward(batch_images_val[i].unsqueeze(0).to(device), mask_val.squeeze(0)).squeeze(0).permute(1,2,0)

                # Add noise as % of the magnitude for real and imaginary channels separately
                kspace_real_val = kspace_sampled_val[:, :, 0]
                kspace_imag_val = kspace_sampled_val[:, :, 1]
                noise_std_real_val = sigma * torch.std(torch.abs(kspace_real_val))
                noise_std_imag_val = sigma * torch.std(torch.abs(kspace_imag_val))
                noise_real_val = noise_std_real_val * torch.randn_like(kspace_real_val)
                noise_imag_val = noise_std_imag_val * torch.randn_like(kspace_imag_val)
                kspace_sampled_val[:, :, 0] += noise_real_val
                kspace_sampled_val[:, :, 1] += noise_imag_val

                batch_kspaces_val_sampled[i] = kspace_sampled_val.to(device)

            batch_kspaces_sampled = batch_kspaces_sampled.permute(0, 3, 1, 2)
            batch_kspaces_val_sampled = batch_kspaces_val_sampled.permute(0, 3, 1, 2)


            # Forward pass
            batch_images_reconstructed = modl_finetune(batch_kspaces_sampled, mask)

            with torch.no_grad():
                batch_images_val_reconstructed = modl_finetune(batch_kspaces_val_sampled, mask_val)

            # Compute loss
            loss = F.mse_loss(batch_images_reconstructed, batch_images)

            # Compute validation loss without gradients
            with torch.no_grad():
                val_loss = F.mse_loss(batch_images_val_reconstructed, batch_images_val)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())        
            val_losses.append(val_loss.item())

            # Store λ values and loss for visualization
            lam_values_step = [torch.abs(lam).item() for lam in modl_finetune.lams]
            lam_values.append(lam_values_step.copy())

            # Print the progress
            if (step + 1) % (n_steps//100) == 0:
                print(f"[{(step+1)/n_steps:.2%}], Train Loss: {loss.item():.4e}, Val Loss: {val_loss.item():.4e}")

                # Save new checkpoints
                torch.save(modl_warmup.state_dict(), f'{CKPT_DIR}/modl_mri-acc_v{version}_{step}.pth')
                torch.save(denoiser.state_dict(), f'{CKPT_DIR}/denoiser_v{version}_{step}.pth')

            # Every 10% of training steps, plot the train and validation losses with smoothed moving average curves
            if (step + 1) % (n_steps//10) == 0:
                # Calculate moving average window size
                window_size = max(1, len(train_losses) // 100)  # 1% of total epochs, minimum 1

                # Calculate moving averages for losses
                finetune_losses_ma = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
                finetune_val_losses_ma = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
                offset = window_size // 2
                x_ma = np.arange(offset, offset + len(finetune_losses_ma))

                plt.figure(figsize=(12, 6))
                plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.3, linewidth=0.8)
                plt.plot(val_losses, label='Validation Loss', color='orange', alpha=0.3, linewidth=0.8)
                plt.plot(x_ma, finetune_losses_ma, label=f'Training Loss MA (window={window_size})', color='darkblue', linewidth=2)
                plt.plot(x_ma, finetune_val_losses_ma, label=f'Validation Loss MA (window={window_size})', color='darkorange', linewidth=2)
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title(f"Fine-tuning Loss Evolution at Step {step+1}")
                plt.legend()
                # plt.xscale('log')
                plt.yscale('log')

                # Display loss reduction for the last 100 epochs or all epochs
                initial_loss_all = train_losses[0]
                final_loss_all = train_losses[-1]
                loss_reduction_all = ((initial_loss_all - final_loss_all) / initial_loss_all) * 100

                initial_val_loss_all = val_losses[0]
                final_val_loss_all = val_losses[-1]
                val_loss_reduction_all = ((initial_val_loss_all - final_val_loss_all) / initial_val_loss_all) * 100

                plt.text(0.02, 0.98, f'Train loss reduction (all {len(train_losses)}): {loss_reduction_all:.4f}%\nVal loss reduction (all {len(val_losses)}): {val_loss_reduction_all:.4f}%', 
                            transform=plt.gca().transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                print(f"Training loss reduction for all {len(train_losses)} epochs: {loss_reduction_all:.4f}%")
                print(f"Validation loss reduction for all {len(val_losses)} epochs: {val_loss_reduction_all:.4f}%")

            # Clear memory to prevent memory leak
            if step < n_steps - 1:
                del batch_kspaces, batch_kspaces_val, batch_kspaces_sampled, batch_kspaces_val_sampled
                del batch_images, batch_images_val, batch_images_reconstructed, batch_images_val_reconstructed

            torch.cuda.empty_cache()


    finally:
        # Allow sleep again
        pass



    # Plot the losses and lambda evolution during fine-tuning training phase

    # Calculate moving average window size
    window_size = max(1, len(train_losses) // 100)  # 1% of total epochs, minimum 1

    # Calculate moving averages for losses
    finetune_losses_ma = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
    finetune_val_losses_ma = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
    offset = window_size // 2
    x_ma = np.arange(offset, offset + len(finetune_losses_ma))

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.3, linewidth=0.8)
    plt.plot(val_losses, label='Validation Loss', color='orange', alpha=0.3, linewidth=0.8)
    plt.plot(x_ma, finetune_losses_ma, label=f'Training Loss MA (window={window_size})', color='darkblue', linewidth=2)
    plt.plot(x_ma, finetune_val_losses_ma, label=f'Validation Loss MA (window={window_size})', color='darkorange', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Fine-tuning Loss Evolution")
    plt.legend()
    # plt.xscale('log')
    plt.yscale('log')

    # Display loss reduction for the last 100 epochs or all epochs
    initial_loss_all = train_losses[0]
    final_loss_all = train_losses[-1]
    loss_reduction_all = ((initial_loss_all - final_loss_all) / initial_loss_all) * 100

    initial_val_loss_all = val_losses[0]
    final_val_loss_all = val_losses[-1]
    val_loss_reduction_all = ((initial_val_loss_all - final_val_loss_all) / initial_val_loss_all) * 100

    plt.text(0.02, 0.98, f'Train loss reduction (all {len(train_losses)}): {loss_reduction_all:.1f}%\nVal loss reduction (all {len(val_losses)}): {val_loss_reduction_all:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    print(f"Training loss reduction for all {len(train_losses)} epochs: {loss_reduction_all:.1f}%")
    print(f"Validation loss reduction for all {len(val_losses)} epochs: {val_loss_reduction_all:.1f}%")
    plt.show()


    # Plot the lambda values evolution
    lam_values = np.array(lam_values)
    plt.figure(figsize=(10, 6))
    for i in range(lam_values.shape[1]):
        plt.plot(lam_values[:, i], label=f'λ_{i+1}')
    plt.xlabel("Epoch")
    plt.ylabel("λ value")
    plt.title("Evolution of λ values during Fine-tuning Training")
    plt.legend()
    plt.yscale('log')
    # plt.savefig(f'{CKPT_DIR}/finetune_lambda_evolution' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()       


    # ==============================================================
    # Save the fine-tuning trained MoDL model and U-Net denoiser
    # ==============================================================
    torch.save(modl_finetune.state_dict(), f'{CKPT_DIR}/long_modl_mri-acc_v' + version + '.pth')
    torch.save(denoiser.state_dict(), f'{CKPT_DIR}/long_denoiser_v' + version + '.pth')


    # ======================================================================
    # Test the trained and fine-tuned MoDL model for accelerated MRI reconstruction
    # ======================================================================

    # Set the model parameters
    # depth = 4 # 5
    # num_features = 32 # 64
    # K = 10 # Number of unrolled iterations for fine-tuning
    # cg_iters = 8 # 10 # Number of CG iterations
    # lam_init = 0.05 # Initial value for lambda
    sigma = 0 # 0.01 # the noise level in the Fourier domain


    # Initialize and load the trained denoiser and MoDL model
    denoiser = CNN_denoiser(n_layers=depth,
                                in_ch=2,
                                out_ch=2,
                                features=num_features).to(device)

    denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/long_denoiser_v' + version + '.pth', map_location=device))

    modl_model = MoDL_SingleCoilMRI_acceleration(denoiser=denoiser, 
                                                num_iters=K,
                                                cg_iters=cg_iters,
                                                lam_init=lam_init).to(device)
    modl_model.load_state_dict(torch.load(f'{CKPT_DIR}/long_modl_mri-acc_v' + version + '.pth', map_location=device))


    denoiser.eval()
    modl_model.eval()

    # Load the test dataset
    test_data = load_volume_kspaces(val_set_path, N_samples=20)
    random.shuffle(test_data)

    # Initialize lists to store metrics
    mse_list_modl = []
    psnr_list_modl = []
    ssim_list_modl = []
    mse_list_zf = []
    psnr_list_zf = []
    ssim_list_zf = []

    # Store example images for plotting
    example_images = []
    example_count = 0
    max_examples = 3

    print(f"Evaluating MoDL model on {len(test_data)} test samples...")

    # Process test data in batches
    test_batch_size = 8
    num_test_batches = (len(test_data) + test_batch_size - 1) // test_batch_size

    with torch.no_grad():
        for batch_idx in range(num_test_batches):

            # Get batch indices
            start_idx = batch_idx * test_batch_size
            end_idx = min((batch_idx + 1) * test_batch_size, len(test_data))
            current_batch_size = end_idx - start_idx

            # Sample batch
            batch_indices = list(range(start_idx, end_idx))
            batch_kspaces = [test_data[i][0] for i in batch_indices]

            # Convert k-space lists to tensors and apply center crop
            batch_kspaces_cropped = []

            for kspace in batch_kspaces:
                kspace_tensor = T.to_tensor(kspace)
                H_k, W_k, _ = kspace_tensor.shape
                crop_H, crop_W = crop_size

                # Standard center crop for k-space
                start_H = (H_k - crop_H) // 2
                end_H = start_H + crop_H
                start_W = (W_k - crop_W) // 2
                end_W = start_W + crop_W

                kspace_cropped = kspace_tensor[start_H:end_H, start_W:end_W, :]
                batch_kspaces_cropped.append(kspace_cropped)

            # Stack the cropped k-spaces into tensors
            batch_kspaces = torch.stack(batch_kspaces_cropped, dim=0)

            # Convert the full k-space to image domain using inverse FFT
            batch_images = A_adj(batch_kspaces.permute(0,3,1,2).to(device), torch.ones((current_batch_size, crop_size[0], crop_size[1]), device=device))

            # Normalize the complex image channels
            batch_images, batch_scales = normalize_complex_image(batch_images)

            # Create a random Gaussian sampling mask and apply it to k-spaces
            mask_func = RandomMaskFunc(center_fractions=[center_fraction], accelerations=[acceleration])  # Create the mask function object
            _, mask, _ = T.apply_mask(batch_kspaces[0], mask_func)   # Apply the mask to k-space
            mask = mask.repeat(1, 1, crop_size[1]).permute(0, 2, 1).to(device)
            batch_kspaces_sampled = torch.zeros((current_batch_size, crop_size[0], crop_size[1], 2), device=device)

            for i in range(current_batch_size):
                kspace_sampled = A_forward(batch_images[i].unsqueeze(0).to(device), mask.squeeze(0)).squeeze(0).permute(1,2,0)
                kspace_real = kspace_sampled[:, :, 0]
                kspace_imag = kspace_sampled[:, :, 1]
                noise_std_real = sigma * torch.std(torch.abs(kspace_real))
                noise_std_imag = sigma * torch.std(torch.abs(kspace_imag))
                noise_real = noise_std_real * torch.randn_like(kspace_real)
                noise_imag = noise_std_imag * torch.randn_like(kspace_imag)
                kspace_sampled = torch.stack((kspace_real + noise_real, kspace_imag + noise_imag), dim=-1)
                batch_kspaces_sampled[i] = kspace_sampled.to(device)

            batch_kspaces_sampled = batch_kspaces_sampled.permute(0, 3, 1, 2)

            # Forward pass through MoDL
            batch_images_reconstructed = modl_model(batch_kspaces_sampled, mask)

            # Zero-filled reconstruction for comparison
            batch_images_zf = A_adj(batch_kspaces_sampled, mask)

            # Convert to magnitude images for evaluation
            batch_images_gt_abs = fastmri.complex_abs(batch_images.permute(0,2,3,1))
            batch_images_recon_abs = fastmri.complex_abs(batch_images_reconstructed.permute(0,2,3,1))
            batch_images_zf_abs = fastmri.complex_abs(batch_images_zf.permute(0,2,3,1))

            # Calculate metrics for each image in the batch
            for i in range(current_batch_size):
                gt = batch_images_gt_abs[i].cpu().numpy()
                recon = batch_images_recon_abs[i].cpu().numpy()
                zf = batch_images_zf_abs[i].cpu().numpy()

                # # Normalize to [0, 1] range
                gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-11)
                # recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-11)
                # zf = (zf - zf.min()) / (zf.max() - zf.min() + 1e-11)

                # Clamp values to [0, 1] range for fair metric calculation
                # gt = np.clip(gt, 0, 1)
                recon = np.clip(recon, 0, 1)
                zf = np.clip(zf, 0, 1)

                # MoDL metrics
                mse_modl = mean_squared_error(gt, recon)

                # Calculate data range from ground truth image
                data_range = 1.0 # gt.max() - gt.min()

                psnr_modl = peak_signal_noise_ratio(gt, recon, data_range=data_range)
                ssim_modl = structural_similarity(gt, recon, data_range=data_range)

                mse_list_modl.append(mse_modl)
                psnr_list_modl.append(psnr_modl)
                ssim_list_modl.append(ssim_modl)

                # Zero-filled metrics
                mse_zf = mean_squared_error(gt, zf)
                psnr_zf = peak_signal_noise_ratio(gt, zf, data_range=data_range)
                ssim_zf = structural_similarity(gt, zf, data_range=data_range)

                mse_list_zf.append(mse_zf)
                psnr_list_zf.append(psnr_zf)
                ssim_list_zf.append(ssim_zf)

                # Store examples for plotting
                if example_count < max_examples:
                    mask_2d = mask[0].cpu().numpy()
                    example_images.append({
                        'gt': gt,
                        'zf': zf,
                        'recon': recon,
                        'mask': mask_2d,
                        'mse_modl': mse_modl,
                        'psnr_modl': psnr_modl,
                        'ssim_modl': ssim_modl,
                        'mse_zf': mse_zf,
                        'psnr_zf': psnr_zf,
                        'ssim_zf': ssim_zf
                    })
                    example_count += 1

            # Print progress
            if (batch_idx + 1) % (num_test_batches//10) == 0 or batch_idx == num_test_batches - 1:
                print(f"Processed batch {(batch_idx + 1)/num_test_batches:.2%}")

    # Calculate average metrics
    avg_mse_modl = np.mean(mse_list_modl)
    avg_psnr_modl = np.mean(psnr_list_modl)
    avg_ssim_modl = np.mean(ssim_list_modl)

    avg_mse_zf = np.mean(mse_list_zf)
    avg_psnr_zf = np.mean(psnr_list_zf)
    avg_ssim_zf = np.mean(ssim_list_zf)

    # Print metrics comparison
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ON {len(test_data)} TEST SAMPLES")
    print(f"{'='*60}")
    print(f"Zero-filled Reconstruction:")
    print(f"  Average MSE:  {avg_mse_zf:.6f}")
    print(f"  Average PSNR: {avg_psnr_zf:.2f} dB")
    print(f"  Average SSIM: {avg_ssim_zf:.4f}")
    print(f"\nMoDL Reconstruction:")
    print(f"  Average MSE:  {avg_mse_modl:.6f}")
    print(f"  Average PSNR: {avg_psnr_modl:.2f} dB")
    print(f"  Average SSIM: {avg_ssim_modl:.4f}")
    # print(f"\nImprovement (MoDL vs Zero-filled):")
    # print(f"  MSE improvement:  {((avg_mse_zf - avg_mse_modl) / avg_mse_zf * 100):.1f}%")
    # print(f"  PSNR improvement: {(avg_psnr_modl - avg_psnr_zf):.2f} dB")
    # print(f"  SSIM improvement: {((avg_ssim_modl - avg_ssim_zf) / avg_ssim_zf * 100):.1f}%")
    print(f"{'='*60}")

    # Plot 3 example reconstructions
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for i in range(3):
        example = example_images[i]

        # Ground truth
        im0 = axes[i, 0].imshow(example['gt'], cmap='gray')
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

        # Sampling mask
        im1 = axes[i, 1].imshow(example['mask'], cmap='gray')
        axes[i, 1].set_title(f'Sampling Mask\n(R={acceleration})')
        axes[i, 1].axis('off')

        # Zero-filled reconstruction
        im2 = axes[i, 2].imshow(example['zf'], cmap='gray')
        axes[i, 2].set_title(f'Zero-filled\nPSNR: {example["psnr_zf"]:.1f} dB\nSSIM: {example["ssim_zf"]:.3f}')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

        # MoDL reconstruction
        im3 = axes[i, 3].imshow(example['recon'], cmap='gray')
        axes[i, 3].set_title(f'MoDL Recon\nPSNR: {example["psnr_modl"]:.1f} dB\nSSIM: {example["ssim_modl"]:.3f}')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], shrink=0.8)

    plt.tight_layout()
    plt.suptitle(f'MRI Reconstruction Examples (Acceleration R={acceleration})', y=1.02, fontsize=16)
    # plt.savefig(f'{CKPT_DIR}/test_reconstruction_examples' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot metrics distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MSE distribution (log scale)
    axes[0].hist(np.log10(mse_list_zf), bins=30, alpha=0.7, label='Zero-filled', color='orange')
    axes[0].hist(np.log10(mse_list_modl), bins=30, alpha=0.7, label='MoDL', color='blue')
    axes[0].set_xlabel('log10(MSE)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('MSE Distribution (log scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PSNR distribution
    axes[1].hist(psnr_list_zf, bins=30, alpha=0.7, label='Zero-filled', color='orange')
    axes[1].hist(psnr_list_modl, bins=30, alpha=0.7, label='MoDL', color='blue')
    axes[1].set_xlabel('PSNR (dB)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('PSNR Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SSIM distribution
    axes[2].hist(ssim_list_zf, bins=30, alpha=0.7, label='Zero-filled', color='orange')
    axes[2].hist(ssim_list_modl, bins=30, alpha=0.7, label='MoDL', color='blue')
    axes[2].set_xlabel('SSIM')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('SSIM Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig(f'{CKPT_DIR}/test_metrics_distribution' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Free up memory
    # del test_data
    # del batch_kspaces, batch_kspaces_sampled, batch_images, batch_images_reconstructed, batch_images_zf
    torch.cuda.empty_cache()




    # ==========================================
    # Train the flow model on the MoDL model
    # ==========================================

    # Define parameters
    depth = 4 # 5 in the original paper
    num_features = 32 # 64 in the original paper
    K = 10 # Number of unrolled iterations for fine-tuning
    cg_iters = 8 # 10 # Number of CG iterations
    lam_init = 0.05 # Initial value for lambda
    sigma = 0.01 # the noise level in the Fourier domain
    n_steps = 1000 # 10*((21*973)//batch_size)
    hid_dim = 512
    joint_training = False # flag whether to train the flow and task models jointly or not
    train_fm_losses = []
    train_task_losses = []
    train_losses = []
    sig_1_values = []
    sig_2_values = []
    sig_3_values = []

    # Print the number of training steps
    print(f"Number of steps for flow model training: {n_steps}")

    # Sigmoid steepness function parameters
    steepness = 20
    power = 2

    offset = 0

    # Initialize the trainable loss uncertainty parameters
    # log_sig_1 = torch.nn.Parameter(torch.zeros(1, device=device))
    # log_sig_2 = torch.nn.Parameter(torch.zeros(1, device=device))

    # Initialize (and load) the trained denoiser and MoDL model
    denoiser = CNN_denoiser(n_layers=depth,
                                in_ch=2,
                                out_ch=2,
                                features=num_features).to(device)

    if joint_training == False:
        denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/long_denoiser_v' + version + '.pth', map_location=device))
        denoiser.eval()

    else:
        denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/denoiser_warmup_v' + version + '.pth', map_location=device))

    modl_model = MoDL_SingleCoilMRI_acceleration(denoiser=denoiser, 
                                                num_iters=K,
                                                cg_iters=cg_iters,
                                                lam_init=lam_init).to(device)

    if joint_training == False:
        modl_model.load_state_dict(torch.load(f'{CKPT_DIR}/long_modl_mri-acc_v' + version + '.pth', map_location=device))
        modl_model.eval()

    # Initialize the flow model
    flow_model = FlowMatchingMaskGenerator(cnn_denoiser=denoiser,
                                           mask_size=crop_size[0],
                                           hidden_dim=hid_dim).to(device)

    # # Load the flow model checkpoint
    # if joint_training == False:
    #     flow_model.load_state_dict(torch.load(f'{CKPT_DIR}/flow_modl_checkpoint_step_{step}.pth', map_location=device))

    # else:
    #     checkpoint = torch.load(f'{CKPT_DIR}/flow_modl_checkpoint_step_{offset}.pth', map_location=device)
    #     denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
    #     modl_model.load_state_dict(checkpoint['modl_model_state_dict'])
    #     flow_model.load_state_dict(checkpoint['flow_model_state_dict'])

    #     # Initialize the trainable loss uncertainty parameters
    #     log_sig_1 = torch.nn.Parameter(checkpoint['log_sig_1'][0])
    #     log_sig_2 = torch.nn.Parameter(checkpoint['log_sig_2'][0])
    #     log_sig_3 = torch.nn.Parameter(checkpoint['log_sig_3'][0])

    #     # Initialize the optimizer
    #     optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(denoiser.parameters()) + list(modl_model.parameters()) + [log_sig_1, log_sig_2, log_sig_3], lr=1e-4)


    # Print the number of trainable parameters in the flow model
    num_params_fm = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the flow model: {num_params_fm:,}")

    # Create a LF mask for k-space
    lf_mask = torch.zeros(crop_size[0], device=device)
    lf_mask[(crop_size[0]-num_low_freqs)//2:(crop_size[0]+num_low_freqs)//2] = 1.0
    lf_mask_expanded = lf_mask.unsqueeze(1).repeat(1, crop_size[1]).permute(1, 0).unsqueeze(0).to(device)  # Shape: (1, crop_size[0], crop_size[1])

    # # Initialize the optimizer
    # if joint_training == True:
    #     optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(denoiser.parameters()) + list(modl_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)

    # else:
    #     optimizer = torch.optim.Adam(list(flow_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)


    # Prevent the system from going to sleep

    try:

        # Training loop
        for step in np.arange(offset, n_steps):

            # Sample a batch from the DataLoaders
            batch_kspaces = next(iter(train_loader))

            # Convert the full k-space to image domain using inverse FFT
            batch_images = A_adj(batch_kspaces.permute(0,3,1,2).to(device), torch.ones((batch_size, crop_size[0], crop_size[1]), device=device))

            # Normalize the complex image channels
            batch_images, _ = normalize_complex_image(batch_images)

            # Set the number of sensors as the total number of lines to sample (excluding low frequencies)
            n_sensors = crop_size[0] // acceleration

            # Create the LF sampled conditioning batch for the flow model
            batch_kspaces_lf_sampled = torch.zeros((batch_size, crop_size[0], crop_size[1], 2), device=device)

            for i in range(batch_size):
                kspace_lf_sampled = A_forward(batch_images[i].unsqueeze(0).to(device), lf_mask_expanded.squeeze(0)).squeeze(0).permute(1,2,0)
                kspace_real = kspace_lf_sampled[:, :, 0]
                kspace_imag = kspace_lf_sampled[:, :, 1]
                noise_std_real = sigma * torch.std(torch.abs(kspace_real))
                noise_std_imag = sigma * torch.std(torch.abs(kspace_imag))
                noise_real = noise_std_real * torch.randn_like(kspace_real)
                noise_imag = noise_std_imag * torch.randn_like(kspace_imag)
                kspace_lf_sampled = torch.stack((kspace_real + noise_real, kspace_imag + noise_imag), dim=-1)
                batch_kspaces_lf_sampled[i] = kspace_lf_sampled.to(device)

            batch_images_cond = A_adj(batch_kspaces_lf_sampled.permute(0,3,1,2), lf_mask_expanded)


            # # Sample a batch of GT mask estimation from the Gaussian distribution
            # gt_mask_func = RandomMaskFunc(center_fractions=[center_fraction], accelerations=[acceleration])  # Create the mask function object
            # batch_mask_gt = torch.zeros((batch_size, crop_size[0]), device=device)

            # for i in range(batch_size):
            #     _, mask_gt, _ = T.apply_mask(batch_kspaces[i], gt_mask_func)
            #     mask_gt = mask_gt.squeeze().to(device)

            #     # Remove the low frequencies from the GT mask to only learn the high frequency sampling pattern
            #     mask_gt[(crop_size[0]-num_low_freqs)//2:(crop_size[0]+num_low_freqs)//2] = 0.0

            #     batch_mask_gt[i] = mask_gt.to(device)     


            # Sample a batch of GT mask estimation from the k-space energy distribution
            batch_mask_gt = torch.zeros((batch_size, crop_size[0]), device=device)

            for i in range(batch_size):
                gt_logits = torch.mean(torch.sqrt(batch_kspaces[i, :, :, 0]**2 + batch_kspaces[i, :, :, 1]**2 + 1e-11), dim=0)
                gt_logits = gt_logits/torch.max(gt_logits)

                # Set the LF region logits to a very low value to ensure they are always sampled
                gt_logits[(crop_size[0]-num_low_freqs)//2:(crop_size[0]+num_low_freqs)//2] = -1e6

                gt_probs = torch.softmax(gt_logits, dim=0)

                if step == 0 and i == 0:  # Plot the GT sampling distribution for the first image in the first batch
                    plt.figure(figsize=(6, 4))
                    plt.plot(gt_probs.cpu().numpy(), color='blue')
                    lf_start = (crop_size[0] - num_low_freqs) // 2
                    lf_end = (crop_size[0] + num_low_freqs) // 2
                    plt.axvspan(lf_start, lf_end, color='orange', alpha=0.3, label='Low Frequency Region')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.title("GT Sampling Distribution based on k-space Energy")
                    plt.xlabel("k-space line index")
                    plt.ylabel("Sampling Probability")
                    plt.show()

                sampling_indices_gt = torch.multinomial(gt_probs, n_sensors, replacement=False).long()
                mask_gt = torch.zeros(crop_size[0], device=device)
                mask_gt[sampling_indices_gt] = 1.0
                batch_mask_gt[i] = torch.maximum(mask_gt, lf_mask)  # Combine with LF mask to ensure LF sampling


            # Sample noise x0 and time t
            x0 = torch.randn(batch_size, crop_size[0]).to(device)
            t = torch.rand(batch_size, 1).to(device)

            # Linear coupling between x0 and x1 to create a continuous path for the flow model to learn
            xt = x0 * (1 - t) + batch_mask_gt * t

            # GT velocity for the flow model (derivative of the linear coupling)
            gt_velocity = batch_mask_gt - x0

            # Forward pass through the flow model
            pred_velocity = flow_model(x0, t, batch_images_cond)
            pred_x = x0 + pred_velocity * t

            # Apply the sigmoid gate to create soft masks
            mask_train = torch.zeros_like(pred_x)

            for i in range(batch_size):
                threshold = torch.quantile(pred_x[i], 1 - 1/acceleration + center_fraction)
                soft_mask = torch.sigmoid(steepness * (pred_x[i] - threshold) * (t[i].item() ** power))

                # Merge the soft mask with the LF mask to create the final sampling mask for the task model
                mask_train[i] = torch.maximum(soft_mask, lf_mask)

            # Create the soft-sampled k-space batch for the task model
            mask_train_expanded = torch.zeros((batch_size, crop_size[0], crop_size[1]), device=device)
            batch_kspaces_sampled = torch.zeros((batch_size, crop_size[0], crop_size[1], 2), device=device)

            for i in range(batch_size):

                # Training batch k-space sampling
                mask_train_expanded[i] = mask_train[i].unsqueeze(1).repeat(1, crop_size[1]).permute(1, 0)
                kspace_sampled = A_forward(batch_images[i].unsqueeze(0).to(device), mask_train_expanded[i]).squeeze(0).permute(1,2,0)

                # Add noise as % of the magnitude for real and imaginary channels separately
                kspace_real = kspace_sampled[:, :, 0]
                kspace_imag = kspace_sampled[:, :, 1]
                noise_std_real = sigma * torch.std(torch.abs(kspace_real))
                noise_std_imag = sigma * torch.std(torch.abs(kspace_imag))
                noise_real = noise_std_real * torch.randn_like(kspace_real)
                noise_imag = noise_std_imag * torch.randn_like(kspace_imag)
                kspace_sampled = torch.stack((kspace_real + noise_real, kspace_imag + noise_imag), dim=-1)

                batch_kspaces_sampled[i] = kspace_sampled.to(device)

            batch_kspaces_sampled = batch_kspaces_sampled.permute(0, 3, 1, 2)

            # Forward pass through the task model
            batch_images_reconstructed = modl_model(batch_kspaces_sampled, mask_train_expanded)

            # Calculate the losses
            fm_loss = F.mse_loss(pred_velocity, gt_velocity) # Compute FM loss (MSE between predicted and GT velocities)
            task_loss_1 = F.mse_loss(batch_images_reconstructed, batch_images) # Compute task loss (MSE for regression)
            task_loss_2 = 1 - structural_similarity_index_measure(batch_images_reconstructed, batch_images, data_range=None) # Compute task loss (SSIM for image quality)
            train_fm_losses.append(fm_loss.item())
            # train_task_losses.append(task_loss.item())
            train_task_losses.append(task_loss_1.item() + task_loss_2.item())

            # At the first iteration, initialize the log_sig parameters and the optimizer after seeing the scale of the losses
            if step == 0:

                # Initialize the trainable loss uncertainty parameters
                # sig_1_init = np.sqrt(np.abs(fm_loss.item()))
                # sig_2_init = np.sqrt(np.abs(task_loss.item()))
                sig_1_init = np.sqrt(3/2*np.abs(fm_loss.item()))
                sig_2_init = np.sqrt(3/2*np.abs(task_loss_1.item()))
                sig_3_init = np.sqrt(3/2*np.abs(task_loss_2.item()))
                log_sig_1 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(sig_1_init))
                log_sig_2 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(sig_2_init))
                log_sig_3 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(sig_3_init))

                # Initialize the optimizer
                if joint_training == True:
                    optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(denoiser.parameters()) + list(modl_model.parameters()) + [log_sig_1, log_sig_2, log_sig_3], lr=1e-4)

                else:
                    optimizer = torch.optim.Adam(list(flow_model.parameters()) + [log_sig_1, log_sig_2, log_sig_3], lr=1e-4)
                    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=n_steps)


            sig_1 = torch.exp(log_sig_1)
            sig_2 = torch.exp(log_sig_2)
            sig_3 = torch.exp(log_sig_3)
            total_loss = 1/(2*torch.pow(sig_1, 2)) * fm_loss + 1/(2*torch.pow(sig_2, 2)) * task_loss_1 + 1/(2*torch.pow(sig_3, 2)) * task_loss_2 + log_sig_1 + log_sig_2 + log_sig_3

            # Track uncertainty parameters and the total training loss
            sig_1_values.append(sig_1.item())
            sig_2_values.append(sig_2.item())
            sig_3_values.append(sig_3.item())
            train_losses.append(total_loss.item())

            # Backpropagation and optimization step
            optimizer.zero_grad()
            total_loss.backward()

            if joint_training == True:
                torch.nn.utils.clip_grad_norm_(list(flow_model.parameters()) + list(denoiser.parameters()) + list(modl_model.parameters()) + [log_sig_1, log_sig_2, log_sig_3], max_norm=1.0)

            else:
                torch.nn.utils.clip_grad_norm_(list(flow_model.parameters()) + [log_sig_1, log_sig_2, log_sig_3], max_norm=1.0)

            optimizer.step()
            # scheduler.step()  # Update the learning rate
            optimizer.zero_grad()

            # Print the losses over epochs
            if (step+1) % (n_steps//20) == 0:
                print(f"[{(step+1)/n_steps:.2%}] Total Loss: {total_loss.item():.6e}, FM Loss: {fm_loss.item():.6e}, Task Loss: {task_loss_1.item():.6e}, Sig_1: {sig_1_values[-1]:.4f}, Sig_2: {sig_2_values[-1]:.4f}, Sig_3: {sig_3_values[-1]:.4f}")

                # # Save the model checkpoint
                # torch.save({
                #     'flow_model_state_dict': flow_model.state_dict(),
                #     'denoiser_state_dict': denoiser.state_dict(),
                #     'modl_model_state_dict': modl_model.state_dict(),
                #     'log_sig_1': log_sig_1,
                #     'log_sig_2': log_sig_2,
                #     'log_sig_3': log_sig_3
                # }, f'{CKPT_DIR}/flow_modl_checkpoint_step_{step+1}.pth')

    finally:
        # Allow sleep again
        pass



    # Plot the training losses evolution
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_fm_losses, color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("FM Loss")
    plt.title("Flow Matching Loss Evolution")
    plt.grid(True, alpha=0.3)


    plt.subplot(1, 3, 2)
    plt.plot(train_task_losses, color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("Task Loss")
    plt.title("Task Loss Evolution")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(1, 3, 3)
    plt.plot(train_losses, color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Total Training Loss Evolution")
    plt.grid(True, alpha=0.3)

    # Display average loss for the last 100 epochs
    if len(train_losses) >= 100:
        avg_loss_last_100 = sum(train_losses[-100:]) / 100
        plt.text(0.02, 0.98, f'Avg train loss (last 100): {avg_loss_last_100:.6f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"Average training loss for the last 100 epochs: {avg_loss_last_100:.6f}")

    else:
        avg_loss_all = sum(train_losses) / len(train_losses)
        plt.text(0.02, 0.98, f'Avg train loss (all {len(train_losses)}): {avg_loss_all:.6f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"Average training loss for all {len(train_losses)} epochs: {avg_loss_all:.6f}")

    plt.show()

    # Plot the uncertainty parameters over training
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(sig_1_values)/sig_1_values[0], label='Sigma 1 (FM)', alpha=0.7)
    plt.plot(np.array(sig_2_values)/sig_2_values[0], label='Sigma 2 (Task (MSE))', alpha=0.7)
    plt.plot(np.array(sig_3_values)/sig_3_values[0], label='Sigma 3 (Task (SSIM))', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('Learned Sigma Values')
    plt.title('Learned Uncertainty Parameters (normalized) During Training')
    plt.legend()
    plt.grid(True)
    plt.show()


    # ===============================
    # Save the trained flow model
    # ===============================

    if joint_training == True:
        torch.save({
            'flow_model_state_dict': flow_model.state_dict(),
            'denoiser_state_dict': denoiser.state_dict(),
            'modl_model_state_dict': modl_model.state_dict(),
            'log_sig_1': log_sig_1,
            'log_sig_2': log_sig_2,
            'log_sig_3': log_sig_3
        }, f'{CKPT_DIR}/joint_flow_modl_v' + version + '.pth')

    else:
        torch.save({
            'flow_model_state_dict': flow_model.state_dict(),
            'log_sig_1': log_sig_1,
            'log_sig_2': log_sig_2,
            'log_sig_3': log_sig_3
        }, f'{CKPT_DIR}/flow_modl_v' + version + '.pth')



    # ======================================================================
    # Evaluate the performance of MoDL with the flow generated masks on the test set
    # ======================================================================

    # Set the model parameters
    depth = 4 # 5
    num_features = 32 # 64
    K = 10 # Number of unrolled iterations for fine-tuning
    cg_iters = 8 # 10 # Number of CG iterations
    lam_init = 0.05 # Initial value for lambda
    hid_dim = 512
    joint_training = False
    steepness = 20
    power = 2
    sigma = 0 # 0.01 # the noise level in the Fourier domain
    time_points = torch.linspace(0, 1, 20).to(device)



    # Initialize the trained denoiser, MoDL model, and the flow model
    denoiser = CNN_denoiser(n_layers=depth,
                                in_ch=2,
                                out_ch=2,
                                features=num_features).to(device)

    modl_model = MoDL_SingleCoilMRI_acceleration(denoiser=denoiser, 
                                                num_iters=K,
                                                cg_iters=cg_iters,
                                                lam_init=lam_init).to(device)

    if joint_training == True:
        checkpoint = torch.load(f'{CKPT_DIR}/joint_flow_modl_v' + version + '.pth', map_location=device)
        denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
        modl_model.load_state_dict(checkpoint['modl_model_state_dict'])
        denoiser.eval()
        modl_model.eval()

        flow_model = FlowMatchingMaskGenerator(cnn_denoiser=denoiser,
                                           mask_size=crop_size[0],
                                           hidden_dim=hid_dim).to(device)

        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])

    else:
        # checkpoint = torch.load(f'{CKPT_DIR}/flow_modl_checkpoint_step_1050.pth', map_location=device)
        checkpoint = torch.load(f'{CKPT_DIR}/flow_modl_v' + version + '.pth', map_location=device)
        denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/denoiser_v' + version + '.pth', map_location=device))
        modl_model.load_state_dict(torch.load(f'{CKPT_DIR}/modl_mri-acc_v' + version + '.pth', map_location=device))
        denoiser.eval()
        modl_model.eval()

        flow_model = FlowMatchingMaskGenerator(cnn_denoiser=denoiser,
                                           mask_size=crop_size[0],
                                           hidden_dim=hid_dim).to(device)

        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])


    denoiser.eval()
    modl_model.eval()
    flow_model.eval()


    # Create a LF mask for k-space
    lf_mask = torch.zeros(crop_size[0], device=device)
    lf_mask[(crop_size[0]-num_low_freqs)//2:(crop_size[0]+num_low_freqs)//2] = 1.0
    lf_mask_expanded = lf_mask.unsqueeze(1).repeat(1, crop_size[1]).permute(1, 0).unsqueeze(0).to(device)  # Shape: (1, crop_size[0], crop_size[1])

    # Load the test dataset
    test_data = load_volume_kspaces(val_set_path, N_samples=20)
    random.shuffle(test_data)

    # Initialize lists to store metrics
    mse_list_modl = []
    psnr_list_modl = []
    ssim_list_modl = []
    mse_list_zf = []
    psnr_list_zf = []
    ssim_list_zf = []

    # Initialize accumulated mask for averaging
    accumulated_mask = torch.zeros((crop_size[0], crop_size[1]), device=device)
    mask_count = 0

    # Store example images for plotting
    example_images = []
    example_count = 0
    max_examples = 3

    print(f"Evaluating MoDL model on {len(test_data)} test samples...")

    # Process test data in batches
    test_batch_size = 8
    num_test_batches = (len(test_data) + test_batch_size - 1) // test_batch_size

    with torch.no_grad():
        for batch_idx in range(num_test_batches):

            # Get batch indices
            start_idx = batch_idx * test_batch_size
            end_idx = min((batch_idx + 1) * test_batch_size, len(test_data))
            current_batch_size = end_idx - start_idx

            # Sample batch
            batch_indices = list(range(start_idx, end_idx))
            batch_kspaces = [test_data[i][0] for i in batch_indices]

            # Convert k-space lists to tensors and apply center crop
            batch_kspaces_cropped = []

            for kspace in batch_kspaces:
                kspace_tensor = T.to_tensor(kspace)
                H_k, W_k, _ = kspace_tensor.shape
                crop_H, crop_W = crop_size

                # Standard center crop for k-space
                start_H = (H_k - crop_H) // 2
                end_H = start_H + crop_H
                start_W = (W_k - crop_W) // 2
                end_W = start_W + crop_W

                kspace_cropped = kspace_tensor[start_H:end_H, start_W:end_W, :]
                batch_kspaces_cropped.append(kspace_cropped)

            # Stack the cropped k-spaces into tensors
            batch_kspaces = torch.stack(batch_kspaces_cropped, dim=0)

            # Convert the full k-space to image domain using inverse FFT
            batch_images = A_adj(batch_kspaces.permute(0,3,1,2).to(device), torch.ones((current_batch_size, crop_size[0], crop_size[1]), device=device))

            # Normalize the complex image channels
            batch_images, batch_scales = normalize_complex_image(batch_images)

            # Set the number of sensors as the total number of lines to sample
            n_sensors = 1 + crop_size[0] // acceleration
            # n_sensors = crop_size[0] - 10

            # Create the LF sampled conditioning batch for the flow model
            batch_kspaces_lf_sampled = torch.zeros((current_batch_size, crop_size[0], crop_size[1], 2), device=device)

            for i in range(current_batch_size):
                kspace_lf_sampled = A_forward(batch_images[i].unsqueeze(0).to(device), lf_mask_expanded.squeeze(0)).squeeze(0).permute(1,2,0)
                kspace_real = kspace_lf_sampled[:, :, 0]
                kspace_imag = kspace_lf_sampled[:, :, 1]
                noise_std_real = sigma * torch.std(torch.abs(kspace_real))
                noise_std_imag = sigma * torch.std(torch.abs(kspace_imag))
                noise_real = noise_std_real * torch.randn_like(kspace_real)
                noise_imag = noise_std_imag * torch.randn_like(kspace_imag)
                kspace_lf_sampled = torch.stack((kspace_real + noise_real, kspace_imag + noise_imag), dim=-1)
                batch_kspaces_lf_sampled[i] = kspace_lf_sampled.to(device)

            batch_images_cond = A_adj(batch_kspaces_lf_sampled.permute(0,3,1,2), lf_mask_expanded)

            # Generate a batch of sampling masks with flow and apply them to k-spaces
            batch_masks = torch.zeros((current_batch_size, crop_size[0], crop_size[1]), device=device)
            batch_kspaces_sampled = torch.zeros((current_batch_size, crop_size[0], crop_size[1], 2), device=device)

            # Sample noise x0
            x0 = torch.randn(current_batch_size, crop_size[0]).to(device)

            # Make the predictions of the masks using the flow model
            pred_x = odeint(lambda t_step, x: flow_model(x, t_step.expand(x.shape[0], 1), batch_images_cond), x0, time_points, method='rk4')[-1]

            # Apply the sigmoid gate to create soft masks
            mask_test = torch.zeros_like(pred_x)

            for i in range(current_batch_size):
                # threshold = torch.quantile(pred_x[i], 1 - 1/acceleration + center_fraction)
                threshold = torch.quantile(pred_x[i], 1 - 1/acceleration)
                # threshold = torch.quantile(pred_x[i], 1 - n_sensors/crop_size[0])
                soft_mask = torch.sigmoid(steepness * (pred_x[i] - threshold) * (1 ** power))

                # Merge the soft mask with the LF mask to create the final sampling mask for the task model
                mask_test_combined = torch.maximum(soft_mask, lf_mask)
                _, topk_indices = torch.topk(mask_test_combined, n_sensors, largest=True)
                hard_mask = torch.zeros_like(mask_test_combined)
                hard_mask[topk_indices] = 1.0
                mask_test[i] = hard_mask.to(device)

                # Accumulate the masks for averaging
                accumulated_mask += mask_test[i]
                mask_count += 1

            # Create the soft-sampled k-space batch for the task model
            mask_test_expanded = torch.zeros((current_batch_size, crop_size[0], crop_size[1]), device=device)
            batch_kspaces_sampled = torch.zeros((current_batch_size, crop_size[0], crop_size[1], 2), device=device)

            for i in range(current_batch_size):
                mask_test_expanded[i] = mask_test[i].unsqueeze(1).repeat(1, crop_size[1]).permute(1, 0)
                kspace_sampled = A_forward(batch_images[i].unsqueeze(0).to(device), mask_test_expanded[i]).squeeze(0).permute(1,2,0)
                kspace_real = kspace_sampled[:, :, 0]
                kspace_imag = kspace_sampled[:, :, 1]
                noise_std_real = sigma * torch.std(torch.abs(kspace_real))
                noise_std_imag = sigma * torch.std(torch.abs(kspace_imag))
                noise_real = noise_std_real * torch.randn_like(kspace_real)
                noise_imag = noise_std_imag * torch.randn_like(kspace_imag)
                kspace_sampled = torch.stack((kspace_real + noise_real, kspace_imag + noise_imag), dim=-1)
                batch_kspaces_sampled[i] = kspace_sampled.to(device)

            batch_kspaces_sampled = batch_kspaces_sampled.permute(0, 3, 1, 2)

            # Forward pass through the task model
            batch_images_reconstructed = modl_model(batch_kspaces_sampled, mask_test_expanded)

            # Zero-filled reconstruction for comparison
            batch_images_zf = A_adj(batch_kspaces_sampled, mask_test_expanded)

            # Convert to magnitude images for evaluation
            batch_images_gt_abs = fastmri.complex_abs(batch_images.permute(0,2,3,1))
            batch_images_recon_abs = fastmri.complex_abs(batch_images_reconstructed.permute(0,2,3,1))
            batch_images_zf_abs = fastmri.complex_abs(batch_images_zf.permute(0,2,3,1))

            # Calculate metrics for each image in the batch
            for i in range(current_batch_size):
                gt = batch_images_gt_abs[i].cpu().numpy()
                recon = batch_images_recon_abs[i].cpu().numpy()
                zf = batch_images_zf_abs[i].cpu().numpy()

                # Normalize to [0, 1] range
                gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-11)
                # recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-11)
                # zf = (zf - zf.min()) / (zf.max() - zf.min() + 1e-11)

                # Clamp the values to [0, 1] range for fair metric calculation
                # gt = np.clip(gt, 0, 1)
                recon = np.clip(recon, 0, 1)
                zf = np.clip(zf, 0, 1)

                # MoDL metrics
                mse_modl = mean_squared_error(gt, recon)

                # Calculate data range from ground truth image
                # data_range = gt.max() - gt.min()
                data_range = 1.0

                psnr_modl = peak_signal_noise_ratio(gt, recon, data_range=data_range)
                ssim_modl = structural_similarity(gt, recon, data_range=data_range)

                mse_list_modl.append(mse_modl)
                psnr_list_modl.append(psnr_modl)
                ssim_list_modl.append(ssim_modl)

                # Zero-filled metrics
                mse_zf = mean_squared_error(gt, zf)
                psnr_zf = peak_signal_noise_ratio(gt, zf, data_range=data_range)
                ssim_zf = structural_similarity(gt, zf, data_range=data_range)

                mse_list_zf.append(mse_zf)
                psnr_list_zf.append(psnr_zf)
                ssim_list_zf.append(ssim_zf)

                # Store examples for plotting
                if example_count < max_examples:
                    mask_2d = mask_test_expanded[i].cpu().numpy()
                    example_images.append({
                        'gt': gt,
                        'zf': zf,
                        'recon': recon,
                        'mask': mask_2d,
                        'mse_modl': mse_modl,
                        'psnr_modl': psnr_modl,
                        'ssim_modl': ssim_modl,
                        'mse_zf': mse_zf,
                        'psnr_zf': psnr_zf,
                        'ssim_zf': ssim_zf
                    })
                    example_count += 1

            # Print progress
            if (batch_idx + 1) % (num_test_batches//10) == 0 or batch_idx == num_test_batches - 1:
                print(f"Processed batch {(batch_idx + 1)/num_test_batches:.2%}")

    # Calculate average metrics
    avg_mse_modl = np.mean(mse_list_modl)
    avg_psnr_modl = np.mean(psnr_list_modl)
    avg_ssim_modl = np.mean(ssim_list_modl)

    avg_mse_zf = np.mean(mse_list_zf)
    avg_psnr_zf = np.mean(psnr_list_zf)
    avg_ssim_zf = np.mean(ssim_list_zf)

    # Print metrics comparison
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ON {len(test_data)} TEST SAMPLES")
    print(f"{'='*60}")
    print(f"Zero-filled Reconstruction:")
    print(f"  Average MSE:  {avg_mse_zf:.6f}")
    print(f"  Average PSNR: {avg_psnr_zf:.2f} dB")
    print(f"  Average SSIM: {avg_ssim_zf:.4f}")
    print(f"\nMoDL Reconstruction (flow mask):")
    print(f"  Average MSE:  {avg_mse_modl:.6f}")
    print(f"  Average PSNR: {avg_psnr_modl:.2f} dB")
    print(f"  Average SSIM: {avg_ssim_modl:.4f}")
    print(f"{'='*60}")

    # Plot 3 example reconstructions
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for i in range(3):
        example = example_images[i]

        # Ground truth
        im0 = axes[i, 0].imshow(example['gt'], cmap='gray')
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

        # Sampling mask
        im1 = axes[i, 1].imshow(example['mask'], cmap='gray')
        axes[i, 1].set_title(f'Sampling Mask\n(R={acceleration})')
        axes[i, 1].axis('off')

        # Zero-filled reconstruction
        im2 = axes[i, 2].imshow(example['zf'], cmap='gray')
        axes[i, 2].set_title(f'Zero-filled\nPSNR: {example["psnr_zf"]:.1f} dB\nSSIM: {example["ssim_zf"]:.3f}')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

        # MoDL reconstruction
        im3 = axes[i, 3].imshow(example['recon'], cmap='gray')
        axes[i, 3].set_title(f'MoDL Recon\nPSNR: {example["psnr_modl"]:.1f} dB\nSSIM: {example["ssim_modl"]:.3f}')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], shrink=0.8)

    plt.tight_layout()
    plt.suptitle(f'MRI Reconstruction Examples (Acceleration R={acceleration})', y=1.02, fontsize=16)
    # plt.savefig(f'{CKPT_DIR}/test_reconstruction_examples' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot metrics distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    # MSE distribution (log scale)
    axes[0].hist(np.log10(mse_list_zf), bins=30, alpha=0.7, label='Zero-filled', color='orange')
    axes[0].hist(np.log10(mse_list_modl), bins=30, alpha=0.7, label='MoDL', color='blue')
    axes[0].set_xlabel('log10(MSE)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('MSE Distribution (log scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PSNR distribution
    axes[1].hist(psnr_list_zf, bins=30, alpha=0.7, label='Zero-filled', color='orange')
    axes[1].hist(psnr_list_modl, bins=30, alpha=0.7, label='MoDL', color='blue')
    axes[1].set_xlabel('PSNR (dB)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('PSNR Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SSIM distribution
    axes[2].hist(ssim_list_zf, bins=30, alpha=0.7, label='Zero-filled', color='orange')
    axes[2].hist(ssim_list_modl, bins=30, alpha=0.7, label='MoDL', color='blue')
    axes[2].set_xlabel('SSIM')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('SSIM Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig(f'{CKPT_DIR}/test_metrics_distribution' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Compute the averaged mask (normalized by number of samples)
    averaged_mask = (accumulated_mask / mask_count).cpu().numpy()

    # Plot the averaged mask and its profile
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Averaged mask (2D heatmap)
    im0 = axes[0].imshow(averaged_mask, cmap='hot', aspect='auto')
    axes[0].set_title(f'Averaged Binary Mask\n({mask_count} samples, R={acceleration})', fontsize=12)
    axes[0].set_xlabel('k-space columns')
    axes[0].set_ylabel('k-space rows')
    plt.colorbar(im0, ax=axes[0], label='Probability')

    # 2. Column-wise profile (averaged across rows)
    col_profile = averaged_mask.mean(axis=0)
    axes[1].plot(col_profile, linewidth=2, color='green')
    axes[1].axhline(y=1/acceleration, color='red', linestyle='--', linewidth=1.5, label=f'Expected rate (1/{acceleration})')
    axes[1].fill_between(range(len(col_profile)), col_profile, alpha=0.3, color='green')
    axes[1].set_xlabel('k-space column index', fontsize=11)
    axes[1].set_ylabel('Average sampling probability', fontsize=11)
    axes[1].set_title('Column-wise Sampling Profile', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([0, max(col_profile.max() * 1.1, 1/acceleration * 1.2)])

    plt.tight_layout()
    # plt.savefig(f'{CKPT_DIR}/averaged_mask_profile' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Free up memory
    # del test_data
    # del batch_kspaces, batch_kspaces_sampled, batch_images, batch_images_reconstructed, batch_images_zf
    torch.cuda.empty_cache()





    # ========================================
    # MRI acceleration latency assessement
    # ========================================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    batch_size = 1

    # crop_size = (128, 128) # Nolan 2024
    # acceleration = 4 # Nolan 2024
    # center_fraction = 0.08 # Nolan 2024
    crop_size = (208, 208) # van Gorp 2021
    acceleration = 8 # van Gorp 2021
    center_fraction = 0.04 # van Gorp 2021

    num_low_freqs = int(crop_size[0]*center_fraction) # Calculate the number of low-frequency lines to sample based on the center fraction and crop size
    n_sensors = 1 + crop_size[0] // acceleration # Set the number of sensors as the total number of lines to sample
    version = f'9_acc_{acceleration}'

    # Create sets and data loaders
    val_set_path = args.fastmri_val_dir


    # Set the model parameters
    depth = 4 # 5
    num_features = 32 # 64
    K = 10 # Number of unrolled iterations for fine-tuning
    cg_iters = 8 # 10 # Number of CG iterations
    lam_init = 0.05 # Initial value for lambda
    hid_dim = 512
    joint_training = False
    steepness = 20
    power = 2
    sigma = 0 # the noise level in the Fourier domain
    time_points = torch.linspace(0, 1, 20).to(device)


    # Initialize the trained denoiser, MoDL model, and the flow model
    denoiser = CNN_denoiser(n_layers=depth,
                                in_ch=2,
                                out_ch=2,
                                features=num_features).to(device)

    modl_model = MoDL_SingleCoilMRI_acceleration(denoiser=denoiser, 
                                                num_iters=K,
                                                cg_iters=cg_iters,
                                                lam_init=lam_init).to(device)

    if joint_training == True:
        checkpoint = torch.load(f'{CKPT_DIR}/joint_flow_modl_v' + version + '.pth', map_location=device)
        denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
        modl_model.load_state_dict(checkpoint['modl_model_state_dict'])
        denoiser.eval()
        modl_model.eval()

        flow_model = FlowMatchingMaskGenerator(cnn_denoiser=denoiser,
                                           mask_size=crop_size[0],
                                           hidden_dim=hid_dim).to(device)

        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])

    else:
        # checkpoint = torch.load(f'{CKPT_DIR}/flow_modl_checkpoint_step_1050.pth', map_location=device)
        checkpoint = torch.load(f'{CKPT_DIR}/flow_modl_v' + version + '.pth', map_location=device)
        denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/denoiser_v' + version + '.pth', map_location=device))
        modl_model.load_state_dict(torch.load(f'{CKPT_DIR}/modl_mri-acc_v' + version + '.pth', map_location=device))
        denoiser.eval()
        modl_model.eval()

        flow_model = FlowMatchingMaskGenerator(cnn_denoiser=denoiser,
                                           mask_size=crop_size[0],
                                           hidden_dim=hid_dim).to(device)

        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])

    denoiser.eval()
    modl_model.eval()
    flow_model.eval()


    # Create a LF mask for k-space
    lf_mask = torch.zeros(crop_size[0], device=device)
    lf_mask[(crop_size[0]-num_low_freqs)//2:(crop_size[0]+num_low_freqs)//2] = 1.0
    lf_mask_expanded = lf_mask.unsqueeze(1).repeat(1, crop_size[1]).permute(1, 0).unsqueeze(0).to(device)  # Shape: (1, crop_size[0], crop_size[1])

    # Load the test dataset
    test_data = load_volume_kspaces(val_set_path, N_samples=20)


    # Initialize lists to store latencies
    latencies = []
    slice_count = 0

    print(f"Evaluating MoDL model on {len(test_data)} test samples...")

    # Process test data sample by sample for latency measurement

    with torch.no_grad():
        for mri_slice in test_data:

            batch_kspaces = [mri_slice[0]]

            # Convert k-space lists to tensors and apply center crop
            batch_kspaces_cropped = []

            for kspace in batch_kspaces:
                kspace_tensor = T.to_tensor(kspace)
                H_k, W_k, _ = kspace_tensor.shape
                crop_H, crop_W = crop_size

                # Standard center crop for k-space
                start_H = (H_k - crop_H) // 2
                end_H = start_H + crop_H
                start_W = (W_k - crop_W) // 2
                end_W = start_W + crop_W

                kspace_cropped = kspace_tensor[start_H:end_H, start_W:end_W, :]
                batch_kspaces_cropped.append(kspace_cropped)

            # Stack the cropped k-spaces into tensors
            batch_kspaces = torch.stack(batch_kspaces_cropped, dim=0)

            # Convert the full k-space to image domain using inverse FFT
            batch_images = A_adj(batch_kspaces.permute(0,3,1,2).to(device), torch.ones((batch_size, crop_size[0], crop_size[1]), device=device))

            # Normalize the complex image channels
            batch_images, batch_scales = normalize_complex_image(batch_images)

            # Create the LF sampled conditioning batch for the flow model
            batch_kspaces_lf_sampled = torch.zeros((batch_size, crop_size[0], crop_size[1], 2), device=device)

            for i in range(batch_size):
                kspace_lf_sampled = A_forward(batch_images[i].unsqueeze(0).to(device), lf_mask_expanded.squeeze(0)).squeeze(0).permute(1,2,0)
                kspace_real = kspace_lf_sampled[:, :, 0]
                kspace_imag = kspace_lf_sampled[:, :, 1]
                noise_std_real = sigma * torch.std(torch.abs(kspace_real))
                noise_std_imag = sigma * torch.std(torch.abs(kspace_imag))
                noise_real = noise_std_real * torch.randn_like(kspace_real)
                noise_imag = noise_std_imag * torch.randn_like(kspace_imag)
                kspace_lf_sampled = torch.stack((kspace_real + noise_real, kspace_imag + noise_imag), dim=-1)
                batch_kspaces_lf_sampled[i] = kspace_lf_sampled.to(device)

            batch_images_cond = A_adj(batch_kspaces_lf_sampled.permute(0,3,1,2), lf_mask_expanded)

            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()

            # Generate a batch of sampling masks with flow and apply them to k-spaces
            batch_masks = torch.zeros((batch_size, crop_size[0], crop_size[1]), device=device)
            batch_kspaces_sampled = torch.zeros((batch_size, crop_size[0], crop_size[1], 2), device=device)

            # Sample noise x0
            x0 = torch.randn(batch_size, crop_size[0]).to(device)

            # Make the predictions of the masks using the flow model
            pred_x = odeint(lambda t_step, x: flow_model(x, t_step.expand(x.shape[0], 1), batch_images_cond), x0, time_points, method='rk4')[-1]

            # Apply the sigmoid gate to create soft masks
            mask_test = torch.zeros_like(pred_x)

            for i in range(batch_size):
                threshold = torch.quantile(pred_x[i], 1 - 1/acceleration)
                soft_mask = torch.sigmoid(steepness * (pred_x[i] - threshold) * (1 ** power))

                # Merge the soft mask with the LF mask to create the final sampling mask for the task model
                mask_test_combined = torch.maximum(soft_mask, lf_mask)
                _, topk_indices = torch.topk(mask_test_combined, n_sensors, largest=True)
                hard_mask = torch.zeros_like(mask_test_combined)
                hard_mask[topk_indices] = 1.0
                mask_test[i] = hard_mask.to(device)

            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))  # ms

            slice_count += 1

            # Print progress
            if (slice_count) % (len(test_data)//10) == 0 or slice_count == len(test_data) - 1:
                print(f"Processed slice {slice_count/len(test_data):.2%}")

    latencies = torch.tensor(latencies)

    # Clean the cache
    torch.cuda.empty_cache()

    # Print the statistics of the latencies
    print(f"\n{'='*60}")
    print(f"LATENCY ASSESSMENT FOR FLOW-BASED MASK GENERATION, RESOLUTION: {crop_size}, ACCELERATION: {acceleration}x")
    print(f"{'='*60}")
    print(f"Average latency per slice: {latencies.mean().item():.2f} ms")
    print(f"Median latency per slice: {latencies.median().item():.2f} ms")
    print(f"Standard deviation of latency: {latencies.std().item():.2f} ms")
    print(f"0.95 quantile latency: {latencies.quantile(0.95).item():.2f} ms")
    print(f"0.99 quantile latency: {latencies.quantile(0.99).item():.2f} ms")

    # Plot the latency distribution
    plt.figure(figsize=(10, 6))
    plt.hist(latencies.cpu().numpy(), bins=30, alpha=0.7, color='purple')
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Latency Distribution for Flow-based Mask Generation\n(Resolution: {crop_size}, Acceleration: {acceleration}x)', fontsize=14)
    plt.grid(True, alpha=0.3)
    # plt.savefig(f'{CKPT_DIR}/latency_distribution' + '_v' + version + '.png', dpi=300, bbox_inches='tight')
    plt.show()






def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce Experiment 3 (fastMRI acceleration)."
    )
    parser.add_argument(
        "--fastmri-train-dir", type=str, required=True,
        help="Path to the fastMRI singlecoil knee training .h5 directory."
    )
    parser.add_argument(
        "--fastmri-val-dir", type=str, required=True,
        help="Path to the fastMRI singlecoil knee validation .h5 directory."
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/mri",
        help="Directory to save model checkpoints and figures."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_experiment(args)
