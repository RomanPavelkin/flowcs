"""
Experiment 2: CelebA image reconstruction under learned subsampling.

Reproduces the CelebA results from the paper "Flow-Based Generative Modeling
for Optimizing Sampling Policies in Compressed Sensing Applications".

This script:
  1. Loads CelebA images from a local folder of RGB images (178x218,
     center-cropped to 128x128).
  2. Pre-trains a U-Net autoencoder (AE) on randomly-subsampled measurements.
  3. Trains the task-aware flow-matching mask generator (CondFlow),
     conditioned on the frozen pre-trained AE.
  4. Evaluates reconstruction quality (PSNR/SSIM) under the learned
     sampling mask on a held-out CelebA test folder.

You must supply local paths to your CelebA image folders (this dataset is
not downloaded automatically -- see https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Checkpoints and figures are written to ``--output-dir``
(default: ``checkpoints/celeba``).

Usage:
    python experiments/exp2_celeba.py \\
        --celeba-train-dir /path/to/celeba/images \\
        --celeba-test-dir /path/to/celeba/test_set \\
        --output-dir checkpoints/celeba
"""
import argparse
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from flowcs.data import create_rgb_dataloader
from flowcs.models import CelebAFeatureEncoder, CondFlow
from flowcs.utils import set_seed


def run_experiment(args):
    CKPT_DIR = args.output_dir
    os.makedirs(CKPT_DIR, exist_ok=True)

    set_seed(args.seed)


    # =======================================
    # EXPERIMENT 2: CelebA reconstruction
    # =======================================



    # ==========================
    # Create the dataloaders
    # ==========================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    image_size = 128
    n_pixels = image_size * image_size
    batch_size = 32
    train_val_path = args.celeba_train_dir

    # Create train and validation dataloaders
    train_dataloader = create_rgb_dataloader(
        folder_path=train_val_path,
        batch_size=batch_size
        # shuffle=True,
        # num_workers=0  # Increase if you have multiple CPU cores
    )

    val_dataloader = create_rgb_dataloader(
        folder_path=train_val_path,
        batch_size=batch_size
        # shuffle=False,  # Don't shuffle validation data
        # num_workers=0
    )

    # Plot sample images from train set

    # Get 5 random images from train dataset
    num_samples = 5
    random_indices = random.sample(range(len(train_dataloader.dataset)), num_samples)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i, idx in enumerate(random_indices):
        img = train_dataloader.dataset[idx]  # Shape: [3, 128, 128]

        # Convert from tensor [C, H, W] to numpy [H, W, C] for plotting
        img_np = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')

    plt.suptitle('Random Training Images (128x128 RGB)')
    plt.tight_layout()
    plt.show()


    # ======================================================================
    # Train the U-Net AE to reconstruct the CelebA images from randomly sampled measurements
    # ======================================================================

    # Define parameters
    n_steps = 10*202499//batch_size
    train_losses = []
    val_losses = []

    # Print the number of training steps
    print(f"Number of training steps for reconstructing U-Net: {n_steps}")

    # Instantiate the AE model
    image_encoder = CelebAFeatureEncoder().to(device)

    # Print the number of trainable parameters
    n_params = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in CelebA U-Net AE: {n_params:,}")

    optimizer = torch.optim.Adam(image_encoder.parameters(), 1e-4)

    # Prevent the system from going to sleep

    try:

        for step in range(n_steps):

            # Randomly sample batch_size images from the training set
            x_original = next(iter(train_dataloader)).to(device)  # [B, 3, img_size, img_size]

            # Add noise
            x_noisy = x_original + torch.randn_like(x_original) * 0.02  # Add small Gaussian noise to the original images

            # Randomly occlude some portion of pixels to simulate subsampling

            # Pick a random integer between 50*16 and 500*16 of the pixels to keep (same for all images in the batch)
            n_sensors = torch.randint(50*16, 500*16, (1,), device=device).item()

            # Create a sampling mask
            mask = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            sampling_indices = torch.randperm(n_pixels, device=device)[:n_sensors].long()  # Randomly select n_sensors unique pixel indices to keep
            mask.view(batch_size, -1)[:, sampling_indices] = 1.0  # Set the selected pixel indices in the mask to 1 for all images in the batch

            # Apply the mask to the noisy image
            x_sampled = x_noisy * mask  # Element-wise multiplication to occlude pixels

            # Forward pass through the encoder
            x_recon = image_encoder(x_sampled)  # [B, 3, img_size, img_size]

            # Loss between clean and reconstructed
            # loss = F.mse_loss(x_recon, x_original)
            loss = F.l1_loss(x_recon, x_original)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Calculate validation loss
            with torch.no_grad():
                x_val_original = next(iter(val_dataloader)).to(device)
                x_val_noisy = x_val_original + torch.randn_like(x_val_original) * 0.02
                n_sensors_val = torch.randint(50*16, 500*16, (1,), device=device).item()
                mask_val = torch.zeros((batch_size, 1, image_size, image_size), device=device)
                sampling_indices_val = torch.randperm(n_pixels, device=device)[:n_sensors_val].long()
                mask_val.view(batch_size, -1)[:, sampling_indices_val] = 1.0
                x_val_sampled = x_val_noisy * mask_val
                x_val_recon = image_encoder(x_val_sampled)
                # val_loss = F.mse_loss(x_val_recon, x_val_original)
                val_loss = F.l1_loss(x_val_recon, x_val_original)
                val_losses.append(val_loss.item())

            if (step+1) % (n_steps//20) == 0:
                print(f"[{step+1}] CelebA U-Net AE Train Loss: {loss.item():.4e}, Val Loss: {val_loss.item():.4e}")

                # Save the model checkpoint
                torch.save(image_encoder.state_dict(), f'{CKPT_DIR}/unet_celeba_encoder_checkpoint_step_{step}.pth')

    finally:
        # Allow sleep again
        pass



    # Plot loss evolution for digit encoder
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("CelebA U-Net AE Loss Evolution")
    plt.legend()
    # plt.xscale('log')
    plt.yscale('log')

    # Display average loss for the last 100 epochs
    if len(train_losses) >= 100:
        avg_loss_last_100 = sum(train_losses[-100:]) / 100
        avg_val_loss_last_100 = sum(val_losses[-100:]) / 100
        plt.text(0.02, 0.98, f'Avg train loss (last 100): {avg_loss_last_100:.6f}\nAvg val loss (last 100): {avg_val_loss_last_100:.6f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"Average training loss for the last 100 epochs: {avg_loss_last_100:.6f}")
        print(f"Average validation loss for the last 100 epochs: {avg_val_loss_last_100:.6f}")

    else:
        avg_loss_all = sum(train_losses) / len(train_losses)
        avg_val_loss_all = sum(val_losses) / len(val_losses)
        plt.text(0.02, 0.98, f'Avg train loss (all {len(train_losses)}): {avg_loss_all:.6f}\nAvg val loss (all {len(val_losses)}): {avg_val_loss_all:.6f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"Average training loss for all {len(train_losses)} epochs: {avg_loss_all:.6f}")
        print(f"Average validation loss for all {len(val_losses)} epochs: {avg_val_loss_all:.6f}")

    plt.show()


    # ========================
    # Save the autoencoder
    # ========================
    torch.save(image_encoder.state_dict(), f'{CKPT_DIR}/unet_celeba_ae_v9.pth')
    print("The trained CelebA U-Net AE saved")


    # ==================================================================
    # Evaluate the performace of the U-Net AE on the test CelebA set
    # ==================================================================

    # Load all test images from test folder
    test_folder = args.celeba_test_dir

    # Get all image paths
    test_image_paths = []
    for ext in ['*.jpg']: #, '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
        test_image_paths.extend(glob.glob(os.path.join(test_folder, ext)))

    print(f"Found {len(test_image_paths)} test images")

    # Pre-process all test images (center crop and convert to tensor)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
    ])

    test_set = []
    for img_path in test_image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        test_set.append(img_tensor)

    test_set = torch.stack(test_set).to(device)  # [N, 3, 128, 128]
    print(f"Test images shape: {test_set.shape}")


    rates = 16 * np.array([50, 100, 200, 300, 500])


    # Instantiate and load teh models
    image_encoder = CelebAFeatureEncoder().to(device)
    image_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_celeba_ae_v9.pth', map_location=device))


    # Set the model to evaluation mode
    image_encoder.eval()

    for rate in rates:

        test_psnr_values = []
        test_ssim_values = []

        with torch.no_grad():

            for i in range(0, test_set.shape[0], batch_size):
                end_idx = min(i + batch_size, test_set.shape[0])
                current_batch_size = end_idx - i
                x_original = test_set[i:end_idx]
                x_noisy = x_original.clone() # + torch.randn_like(x_original) * 0.02
                n_sensors = rate
                mask = torch.zeros((current_batch_size, 1, image_size, image_size), device=device)

                for i in range(current_batch_size):
                    sampling_indices = torch.randperm(n_pixels, device=device)[:n_sensors].long()
                    mask.view(current_batch_size, -1)[i, sampling_indices] = 1.0

                x_sampled = x_noisy * mask
                x_recon = image_encoder(x_sampled)

                for i in range(current_batch_size):
                    original_np = x_original[i].detach().cpu().permute(1, 2, 0).numpy()
                    recon_np = x_recon[i].detach().cpu().permute(1, 2, 0).numpy()
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1, channel_axis=2)
                    test_psnr_values.append(psnr)
                    test_ssim_values.append(ssim)

            if rate == 16*100:

                # Plot 3 examples of original and reconstructed test images
                fig, axs = plt.subplots(3, 3, figsize=(12, 12))

                for i in range(3):
                    axs[0, i].imshow(x_original[i].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
                    axs[0, i].set_title("Original")
                    axs[0, i].axis('off')

                    # Display the sampling mask
                    axs[1, i].imshow(mask[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[1, i].set_title(f"Sampling Mask\n({n_sensors} sensors)")
                    axs[1, i].axis('off')

                    # Calculate PSNR for reconstruction
                    original_np = x_original[i].detach().cpu().permute(1, 2, 0).numpy()
                    recon_np = x_recon[i].detach().cpu().permute(1, 2, 0).numpy()
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1, channel_axis=2)

                    axs[2, i].imshow(x_recon[i].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
                    axs[2, i].set_title(f"Reconstructed (16x100 px)\nPSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
                    axs[2, i].axis('off')

                plt.tight_layout()
                plt.show()

                # Plot histograms of PSNR and SSIM values
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.hist(test_psnr_values, bins=30, color='blue', alpha=0.7)
                plt.xlabel('PSNR (dB)')
                plt.ylabel('Frequency')
                plt.title(f'PSNR Distribution (16x{rate//16} sensors)')
                plt.grid(True)

                plt.subplot(1, 2, 2)
                plt.hist(test_ssim_values, bins=30, color='orange', alpha=0.7)
                plt.xlabel('SSIM')
                plt.ylabel('Frequency')
                plt.title(f'SSIM Distribution (16x{rate//16} sensors)')
                plt.grid(True)

                plt.tight_layout()
                plt.show()


            avg_psnr = np.mean(test_psnr_values)
            avg_ssim = np.mean(test_ssim_values)
            print(f"=== Performance at {rate} sensors ({np.round(rate/n_pixels*100, 2)}% of pixels) ===")
            print(f"Average PSNR on test set: {avg_psnr:.4f} dB")
            print(f"Average SSIM on test set: {avg_ssim:.4f}")
            print(" ")



    # ======================================================================
    # Train the flow model for the subsampled CelebA reconstruction through the pre-trained AE
    # ======================================================================

    # Define parameters
    n_steps = 10*202499//batch_size
    joint_training = False # flag whether to train the flow and task models jointly or not
    train_losses = []
    sig_1_values = []
    sig_2_values = []

    # Print number of steps
    print(f"Number of steps for flow model training: {n_steps}")

    # Sigmoid steepness function parameters
    steepness = 20
    power = 2

    offset = 0

    # # Initialize the trainable loss uncertainty parameters
    # log_sig_1 = torch.nn.Parameter(torch.zeros(1, device=device))  # log of the standard deviation for the flow model loss
    # # log_sig_1 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(4))
    # log_sig_2 = torch.nn.Parameter(torch.zeros(1, device=device))  # log of the standard deviation for the task model loss

    # Instantiate the AE model
    task_model = CelebAFeatureEncoder().to(device)

    # if joint_training == False:

    # Load the pre-trained weights
    task_model.load_state_dict(torch.load(f'{CKPT_DIR}/unet_celeba_ae_v9.pth', map_location=device))

    if joint_training == False:

        # Freeze the AE weights
        for param in task_model.parameters():
            param.requires_grad = False

        task_model.eval()

    # Initialize the U-Net-based feature extractor and load the pre-trained weights
    image_encoder = CelebAFeatureEncoder().to(device)
    image_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_celeba_ae_v9.pth', map_location=device))
    image_encoder.eval()

    # Initialize the flow model
    flow_model = CondFlow(encoder=image_encoder, norm_type='bn').to(device)

    # Print the number of trainable parameters in the flow model
    num_params_fm = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the flow model: {num_params_fm:,}")

    if joint_training == True:
        # Print the number of trainable parameters in the task model
        num_params_task = sum(p.numel() for p in task_model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the task model: {num_params_task:,}")

    # # Initialize the optimizer
    # if joint_training == True:
    #     optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(task_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)

    # else:
    #     optimizer = torch.optim.Adam(list(flow_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)


    # Prevent the system from going to sleep

    try:

        # Training loop
        for step in np.arange(offset, n_steps):

            # Randomly sample batch_size images from the training set
            x_batch_train = next(iter(train_dataloader)).to(device)  # [B, 3, img_size, img_size]

            # Add noise to the images
            x_batch_train_noisy = x_batch_train + torch.randn_like(x_batch_train) * 0.02

            # Pick a random integer between 50*16 and 500*16 of the pixels to keep (same for all images in the batch)
            n_sensors = torch.randint(50*16, 500*16, (1,), device=device).item()

            # Create a random sampling mask for conditioning the flow model
            mask_random = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            sampling_indices_random = torch.randperm(n_pixels, device=device)[:n_sensors].long()
            mask_random.view(batch_size, -1)[:, sampling_indices_random] = 1.0

            # Create a conditioning batch
            x_batch_train_conditioning = x_batch_train_noisy * mask_random

            # Sample the GT mask estimation from the mean of the training batch
            mask_gt = torch.zeros((batch_size, 1, image_size, image_size), device=device)

            for i in range(batch_size):
                gt_probs = torch.softmax(torch.mean(x_batch_train, dim=(0,1)).view(-1) + 1e-2, dim=0)
                sampling_indices_gt = torch.multinomial(gt_probs, n_sensors, replacement=False).long()
                mask_gt.view(batch_size, -1)[i, sampling_indices_gt] = 1.0

            # Sample noise x0 and time t
            x0 = torch.randn(batch_size, 1, image_size, image_size).to(device)  # Reshape to match image dimensions
            t = torch.rand(batch_size, 1).to(device)

            # Linear coupling between x0 and x1 to create a continuous path for the flow model to learn
            xt = x0 * (1 - t.view(batch_size, 1, 1, 1)) + mask_gt * t.view(batch_size, 1, 1, 1)

            # GT velocity for the flow model (derivative of the linear coupling)
            gt_velocity = mask_gt - x0

            # Forward pass through the flow model
            pred_velocity = flow_model(xt.view(batch_size, -1), t, x_batch_train_conditioning).view(batch_size, 1, image_size, image_size)
            pred_x = x0 + pred_velocity * t.view(batch_size, 1, 1, 1)
            # pred_x = xt + pred_velocity * t.view(batch_size, 1, 1, 1)

            # Apply the sigmoid gate to create soft masks
            soft_mask = torch.zeros_like(pred_x)

            for i in range(batch_size):
                threshold = torch.quantile(pred_x[i].view(-1), 1 - n_sensors/n_pixels)
                soft_mask[i] = torch.sigmoid(steepness * (pred_x[i] - threshold) * (t[i].item() ** power))

            # Pass the entire batch through the task model
            x_batch_train_soft_sampled = x_batch_train_noisy * soft_mask
            x_recon = task_model(x_batch_train_soft_sampled)

            # Calculate the losses
            fm_loss = F.mse_loss(pred_velocity, gt_velocity) # Compute FM loss (MSE between predicted and GT velocities)
            task_loss = F.l1_loss(x_recon, x_batch_train) # Compute task loss (L1 loss for regression)

            if step == 0:

                # Initialize the trainable loss uncertainty parameters
                sig_1_init = np.sqrt(2/2*np.abs(fm_loss.item()))
                sig_2_init = np.sqrt(2/2*np.abs(task_loss.item()))
                log_sig_1 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(sig_1_init))
                log_sig_2 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(sig_2_init))

                # Initialize the optimizer
                if joint_training == True:
                    optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(task_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)

                else:
                    optimizer = torch.optim.Adam(list(flow_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)

            sig_1 = torch.exp(log_sig_1)
            sig_2 = torch.exp(log_sig_2)                   
            total_loss = 1/(2*torch.pow(sig_1, 2)) * fm_loss + 1/(2*torch.pow(sig_2, 2)) * task_loss + log_sig_1 + log_sig_2

            # Track uncertainty parameters and the total training loss
            sig_1_values.append(sig_1.item())
            sig_2_values.append(sig_2.item())
            train_losses.append(total_loss.item())

            # Backpropagation and optimization step
            optimizer.zero_grad()
            total_loss.backward()

            if joint_training == True:
                torch.nn.utils.clip_grad_norm_(list(flow_model.parameters()) + list(task_model.parameters()) + [log_sig_1, log_sig_2], max_norm=1.0)

            else:
                torch.nn.utils.clip_grad_norm_(list(flow_model.parameters()) + [log_sig_1, log_sig_2], max_norm=1.0)

            optimizer.step()

            # Print the progress
            if (step+1) % (n_steps//50) == 0:
                print(f"[{(step+1)/n_steps:.2%}] Total Loss: {total_loss.item():.6e}, FM Loss: {fm_loss.item():.6e}, Task Loss: {task_loss.item():.6e}, Sig_1: {sig_1_values[-1]:.4f}, Sig_2: {sig_2_values[-1]:.4f}")

                # Save the model checkpoint
                torch.save({
                    'flow_model_state_dict': flow_model.state_dict(),
                    'task_model_state_dict': task_model.state_dict(),
                    'log_sig_1': log_sig_1,
                    'log_sig_2': log_sig_2
                }, f'{CKPT_DIR}/flow_celeba_ae_checkpoint_step_{step+1}.pth')

    finally:
        # Allow sleep again
        pass


    # Plot the training losses evolution
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Flow (MNIST Reconstruction) Loss Evolution")
    plt.legend()
    plt.grid(True)
    # plt.xscale('log')
    # plt.yscale('log')

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
    plt.plot(np.array(sig_2_values)/sig_2_values[0], label='Sigma 2 (Task)', alpha=0.7)
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
            'task_model_state_dict': task_model.state_dict(),
            'log_sig_1': log_sig_1,
            'log_sig_2': log_sig_2
        }, f'{CKPT_DIR}/joint_flow_celeba_ae_v9.pth')
        print("The flow model jointly trained with the classifier saved")

    else:
        torch.save({
            'flow_model_state_dict': flow_model.state_dict(),
            'log_sig_1': log_sig_1,
            'log_sig_2': log_sig_2
        }, f'{CKPT_DIR}/flow_celeba_ae_v9.pth')
        print("The flow model saved")


    # ======================================================================
    # Evaluate the performace of the U-Net AE + flow generated mask on the test CelebA set
    # ======================================================================

    # Load all test images from test folder
    test_folder = args.celeba_test_dir

    # Get all image paths
    test_image_paths = []
    for ext in ['*.jpg']: #, '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
        test_image_paths.extend(glob.glob(os.path.join(test_folder, ext)))

    print(f"Found {len(test_image_paths)} test images")

    # Pre-process all test images (center crop and convert to tensor)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
    ])

    test_set = []
    for img_path in test_image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        test_set.append(img_tensor)

    test_set = torch.stack(test_set).to(device)  # [N, 3, 128, 128]
    print(f"Test images shape: {test_set.shape}")


    # joint_training = False
    # steepness = 20
    # power = 2
    time_points = torch.linspace(0, 1, 20).to(device) # ODE integration time points for the flow model inference
    rates = 16 * np.array([50, 100, 200, 300, 500])
    # rates = 16 * np.array([100])


    # Initialize the AE model and load the pre-trained weights
    task_model =  CelebAFeatureEncoder().to(device)

    if joint_training == False:
        task_model.load_state_dict(torch.load(f'{CKPT_DIR}/unet_celeba_ae_v9.pth', map_location=device))

    else:
        checkpoint = torch.load(f'{CKPT_DIR}/joint_flow_mnist32by32_ae_v9.pth', map_location=device)
        task_model.load_state_dict(checkpoint['task_model_state_dict'])

    # Initialize the U-Net-based feature extractor and load the pre-trained weights
    image_encoder =  CelebAFeatureEncoder().to(device)
    image_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_celeba_ae_v9.pth', map_location=device))
    image_encoder.eval()

    # Initialize the flow model and load the trained weights
    flow_model = CondFlow(encoder=image_encoder, norm_type='bn').to(device)

    if joint_training == False:
        # flow_checkpoint = torch.load(f'{CKPT_DIR}/flow_celeba_ae_v9.pth', map_location=device)
        flow_checkpoint = torch.load(f'{CKPT_DIR}/flow_celeba_ae_checkpoint_step_1265.pth', map_location=device)
        flow_model.load_state_dict(flow_checkpoint['flow_model_state_dict'])

    else:
        flow_checkpoint = torch.load(f'{CKPT_DIR}/joint_flow_celeba_ae_v9.pth', map_location=device)
        flow_model.load_state_dict(flow_checkpoint['flow_model_state_dict'])


    # Set the models to evaluation mode
    task_model.eval()
    flow_model.eval()

    for rate in rates:

        test_psnr_values = []
        test_ssim_values = []
        flow_masks = []

        with torch.no_grad():

            for i in range(0, test_set.shape[0], batch_size):
                end_idx = min(i + batch_size, test_set.shape[0])
                current_batch_size = end_idx - i
                x_original = test_set[i:end_idx]
                x_noisy = x_original.clone() # + torch.randn_like(x_original) * 0.02
                n_sensors = rate
                mask_random = torch.zeros((current_batch_size, 1, image_size, image_size), device=device)

                for b_i in range(current_batch_size):
                    sampling_indices = torch.randperm(n_pixels, device=device)[:n_sensors].long()
                    mask_random.view(current_batch_size, -1)[b_i, sampling_indices] = 1.0

                x_conditioning = x_noisy * mask_random # Conditioning the flow model on a randomly sampled measurements
                x0 = torch.randn(current_batch_size, 1, image_size, image_size).to(device)
                pred_x = odeint(lambda t, x: flow_model(x.view(current_batch_size, -1), t.expand(current_batch_size, 1), x_conditioning), x0, time_points, method='rk4')[-1].view(current_batch_size, 1, image_size, image_size)
                flow_mask = torch.zeros_like(pred_x)

                for b_i in range(current_batch_size):
                    threshold = torch.quantile(pred_x[b_i].view(-1), 1 - n_sensors/n_pixels)
                    flow_mask_soft = torch.sigmoid(steepness * (pred_x[b_i] - threshold) * (1 ** power))
                    _, sampling_indices_flow = torch.topk(flow_mask_soft.view(-1), n_sensors)
                    flow_mask.view(current_batch_size, -1)[b_i, sampling_indices_flow] = 1.0

                flow_masks.append(flow_mask.cpu())
                x_sampled = x_noisy * flow_mask
                x_recon = task_model(x_sampled)

                for i in range(current_batch_size):
                    original_np = x_original[i].detach().cpu().permute(1, 2, 0).numpy()
                    recon_np = x_recon[i].detach().cpu().permute(1, 2, 0).numpy()
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1, channel_axis=2)
                    test_psnr_values.append(psnr)
                    test_ssim_values.append(ssim) 

            if rate == 16*100:

                # Plot 3 examples of original and reconstructed test images
                fig, axs = plt.subplots(3, 3, figsize=(12, 12))

                for i in range(3):
                    axs[0, i].imshow(x_original[i].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
                    axs[0, i].set_title("Original")
                    axs[0, i].axis('off')

                    # Display the sampling mask
                    axs[1, i].imshow(flow_mask[i].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
                    axs[1, i].set_title(f"Sampling Mask (GSE)\n({n_sensors} sensors)")
                    axs[1, i].axis('off')

                    # Calculate PSNR for reconstruction
                    original_np = x_original[i].detach().cpu().permute(1, 2, 0).numpy()
                    recon_np = x_recon[i].detach().cpu().permute(1, 2, 0).numpy()
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1, channel_axis=2)

                    axs[2, i].imshow(x_recon[i].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
                    axs[2, i].set_title(f"Reconstructed (16x100 px)\nPSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
                    axs[2, i].axis('off')

                plt.tight_layout()
                plt.show()

                # Concatenate the flow masks for the entire test set
                flow_masks = torch.cat(flow_masks, dim=0)

                # Plot average mask
                avg_mask = flow_masks.mean(dim=0).cpu().numpy().reshape(image_size, image_size)
                plt.figure(figsize=(8, 8))
                plt.imshow(avg_mask, cmap='gray')
                plt.title(f'Average Flow Mask ({rate} sensors)', fontsize=14)
                plt.colorbar()
                plt.show()  

                # Plot histograms of PSNR and SSIM values
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.hist(test_psnr_values, bins=30, color='blue', alpha=0.7)
                plt.xlabel('PSNR (dB)')
                plt.ylabel('Frequency')
                plt.title(f'PSNR Distribution (16x{rate//16} sensors)')
                plt.grid(True)

                plt.subplot(1, 2, 2)
                plt.hist(test_ssim_values, bins=30, color='orange', alpha=0.7)
                plt.xlabel('SSIM')
                plt.ylabel('Frequency')
                plt.title(f'SSIM Distribution (16x{rate//16} sensors)')
                plt.grid(True)

                plt.tight_layout()
                plt.show()


            avg_psnr = np.mean(test_psnr_values)
            avg_ssim = np.mean(test_ssim_values)
            print(f"=== Performance at {rate} sensors ({np.round(rate/n_pixels*100, 2)}% of pixels for flow mask) ===")
            print(f"Average PSNR on test set: {avg_psnr:.4f} dB")
            print(f"Average SSIM on test set: {avg_ssim:.4f}")
            print(" ")





def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce Experiment 2 (CelebA reconstruction)."
    )
    parser.add_argument(
        "--celeba-train-dir", type=str, required=True,
        help="Path to a folder of CelebA training/validation images."
    )
    parser.add_argument(
        "--celeba-test-dir", type=str, required=True,
        help="Path to a folder of held-out CelebA test images."
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/celeba",
        help="Directory to save model checkpoints and figures."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_experiment(args)
