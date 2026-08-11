"""
Experiment 1: MNIST classification & reconstruction under learned subsampling.

Reproduces the MNIST results from the paper "Flow-Based Generative Modeling
for Optimizing Sampling Policies in Compressed Sensing Applications".

This script runs, end to end:
  1. Data loading (MNIST via scikit-learn's fetch_openml).
  2. Pre-training a U-Net autoencoder (AE) and an MNIST classifier on
     randomly-subsampled measurements.
  3. Training a greedy sensor-selection (GSE) baseline mask for both the
     classifier and the AE.
  4. Training the task-aware flow-matching mask generator (CondFlow) for
     both classification and reconstruction, conditioned on the frozen
     pre-trained task models.
  5. Evaluating all sampling strategies (random / GSE / flow-generated) on
     the held-out MNIST test set.

Checkpoints, logits, and CSV logs are written to ``--output-dir``
(default: ``checkpoints/mnist``).

Usage:
    python experiments/exp1_mnist.py --output-dir checkpoints/mnist
"""
import argparse
import os
from math import comb

import matplotlib
matplotlib.use("Agg")  # headless-safe; figures are still saved/shown via plt.show() no-ops
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

from flowcs.models import DigitFeatureEncoder, DigitClassifier, CondFlow
from flowcs.utils import set_seed


def run_experiment(args):
    CKPT_DIR = args.output_dir
    os.makedirs(CKPT_DIR, exist_ok=True)

    set_seed(args.seed)


    # =======================================================
    # EXPERIMENT 1: MNIST classification & reconstruction
    # =======================================================



    # ======================================
    # Load and prepare the MNIST dataset
    # ======================================

    # Define the shape of the images
    image_size = 28
    n_pixels = image_size**2

    mnist = fetch_openml('mnist_784')
    # Reshape MNIST (originally 28x28) then resize to desired image_size (upscale or downscale)

    # Original MNIST images are 28x28
    X_orig = mnist.data.values.reshape(-1, 28, 28).astype(np.float32)

    if image_size == 28:
        X = X_orig
    else:
        # resize each image with anti-aliasing, preserve original 0-255 range
        X = np.stack([
            resize(img, (image_size, image_size), preserve_range=True, anti_aliasing=True).astype(np.float32)
            for img in X_orig
        ], axis=0)
    y = mnist.target.astype(np.int64).values  # Load labels and convert to numpy array

    # Create datasets for the images and labels
    train_set = X[:50000]
    train_labels = y[:50000]
    validation_set = X[50000:60000]
    validation_labels = y[50000:60000]
    test_set = X[60000:]
    test_labels = y[60000:]

    # Convert datasets to PyTorch tensors and move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_set = torch.tensor(train_set, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    validation_set = torch.tensor(validation_set, dtype=torch.float32).unsqueeze(1).to(device)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long).to(device)
    test_set = torch.tensor(test_set, dtype=torch.float32).unsqueeze(1).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Check for NaN values in all sets
    train_has_nan = torch.isnan(train_set).any().item()
    val_has_nan = torch.isnan(validation_set).any().item()
    test_has_nan = torch.isnan(test_set).any().item()

    print("=== NaN Check ===")
    print(f"Train set contains NaN: {train_has_nan} (count: {torch.isnan(train_set).sum().item()})")
    print(f"Validation set contains NaN: {val_has_nan} (count: {torch.isnan(validation_set).sum().item()})")
    print(f"Test set contains NaN: {test_has_nan} (count: {torch.isnan(test_set).sum().item()})")

    if train_has_nan or val_has_nan or test_has_nan:
        print("⚠️ WARNING: NaN values detected in the datasets!")
    else:
        print("✓ All datasets are NaN-free")
    print()

    # Normalize the sets to [0, 1]
    train_set /= 255.0
    validation_set /= 255.0
    test_set /= 255.0

    # Plot a couple of images with their labels
    plt.figure(figsize=(12, 6))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_set[i].squeeze().cpu(), cmap='gray')
        plt.axis('off')
        plt.title(f'Label: {train_labels[i].cpu()}')

    plt.suptitle('MNIST Images from the train set')
    plt.show()

    # Plot histograms and display data ranges for all sets
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(train_set.cpu().numpy().flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Train Set Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    train_min, train_max = train_set.min().item(), train_set.max().item()
    plt.text(0.02, 0.98, f'Range: [{train_min:.4f}, {train_max:.4f}]\nMean: {train_set.mean().item():.4f}\nStd: {train_set.std().item():.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.subplot(1, 3, 2)
    plt.hist(validation_set.cpu().numpy().flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Validation Set Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    val_min, val_max = validation_set.min().item(), validation_set.max().item()
    plt.text(0.02, 0.98, f'Range: [{val_min:.4f}, {val_max:.4f}]\nMean: {validation_set.mean().item():.4f}\nStd: {validation_set.std().item():.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.subplot(1, 3, 3)
    plt.hist(test_set.cpu().numpy().flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Test Set Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    test_min, test_max = test_set.min().item(), test_set.max().item()
    plt.text(0.02, 0.98, f'Range: [{test_min:.4f}, {test_max:.4f}]\nMean: {test_set.mean().item():.4f}\nStd: {test_set.std().item():.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'Data Distribution for {image_size}x{image_size} Images')
    plt.tight_layout()
    plt.show()

    print(f"Train set - Min: {train_min:.2f}, Max: {train_max:.2f}, Mean: {train_set.mean().item():.2f}, Std: {train_set.std().item():.2f}")
    print(f"Validation set - Min: {val_min:.2f}, Max: {val_max:.2f}, Mean: {validation_set.mean().item():.2f}, Std: {validation_set.std().item():.2f}")
    print(f"Test set - Min: {test_min:.2f}, Max: {test_max:.2f}, Mean: {test_set.mean().item():.2f}, Std: {test_set.std().item():.2f}")
    print()

    del X_orig, X, y # Free memory


    # Evaluate the combinatorial possibilities for the subsampling matrix
    n_pixels = image_size**2

    # percentage2sample = 1 - 87.5/100
    # percentage2sample = 1 - 96.8/100
    percentage2sample = 0.08

    n_sensors = int(np.round(percentage2sample*n_pixels, 0))

    n_comb = comb(n_pixels, n_sensors)

    print(f"Number of possible combinations: {n_comb:.2e} (image size: {image_size} x {image_size}, n_sensors: {n_sensors})")
    print(" ")


    # ======================================================================
    # Train the U-Net AE to reconstruct the MNIST images from randomly sampled measurements
    # ======================================================================

    batch_size = 64
    n_steps = 200*train_set.shape[0]//batch_size
    train_losses = []
    val_losses = []

    # Print the number of training steps
    print(f"Number of training steps for MNIST U-Net AE: {n_steps}")

    # Instantiate the AE model
    digit_encoder = DigitFeatureEncoder().to(device)

    # Print the number of trainable parameters
    n_params = sum(p.numel() for p in digit_encoder.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in MNIST U-Net AE: {n_params:,}")

    optimizer = torch.optim.Adam(digit_encoder.parameters(), 1e-4)

    # Prevent the system from going to sleep

    try:

        for step in range(n_steps):

            # Randomly sample batch_size images from the training set
            batch_indices = np.random.choice(train_set.shape[0], size=batch_size, replace=False)
            x_original = train_set[batch_indices]

            # Add noise
            x_noisy = x_original + torch.randn_like(x_original) * 0.02  # Add small Gaussian noise to the original images

            # Randomly occlude some portion of pixels to simulate subsampling

            # # Pick a random integer between 1% and 8% of the pixels to keep (same for all images in the batch)
            # n_sensors = torch.randint(int(np.round(0.01*n_pixels)), int(np.round(0.08*n_pixels)), (1,), device=device).item()

            # Pick a random integer between 10 and 500 of the pixels to keep (same for all images in the batch)
            n_sensors = torch.randint(10, 500, (1,), device=device).item()

            # Create a sampling mask
            mask = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            sampling_indices = torch.randperm(n_pixels, device=device)[:n_sensors].long()  # Randomly select n_sensors unique pixel indices to keep
            mask.view(batch_size, -1)[:, sampling_indices] = 1.0  # Set the selected pixel indices in the mask to 1 for all images in the batch

            # Apply the mask to the noisy image
            x_sampled = x_noisy * mask  # Element-wise multiplication to occlude pixels

            # Forward pass through the encoder
            x_recon = digit_encoder(x_sampled)  # [B, 1, img_size, img_size]

            # Loss between clean and reconstructed
            # loss = F.mse_loss(x_recon, x_original)
            loss = F.l1_loss(x_recon, x_original)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Calculate validation loss
            with torch.no_grad():
                val_batch_indices = np.random.choice(validation_set.shape[0], size=batch_size, replace=False)
                x_val_original = validation_set[val_batch_indices]
                x_val_noisy = x_val_original + torch.randn_like(x_val_original) * 0.02
                # n_sensors_val = torch.randint(int(np.round(0.01*n_pixels)), int(np.round(0.08*n_pixels)), (1,), device=device).item()
                n_sensors_val = torch.randint(10, 500, (1,), device=device).item()
                mask_val = torch.zeros((batch_size, 1, image_size, image_size), device=device)
                sampling_indices_val = torch.randperm(n_pixels, device=device)[:n_sensors_val].long()
                mask_val.view(batch_size, -1)[:, sampling_indices_val] = 1.0
                x_val_sampled = x_val_noisy * mask_val
                x_val_recon = digit_encoder(x_val_sampled)
                # val_loss = F.mse_loss(x_val_recon, x_val_original)
                val_loss = F.l1_loss(x_val_recon, x_val_original)
                val_losses.append(val_loss.item())

            if (step+1) % (n_steps//20) == 0:
                print(f"[{step+1}] MNIST U-Net AE Train Loss: {loss.item():.4e}, Val Loss: {val_loss.item():.4e}")

                # Save the model checkpoint
                # torch.save(digit_encoder.state_dict(), f'{CKPT_DIR}/unet_mnist28by28_ae_checkpoint_step_{step}.pth')
                torch.save(digit_encoder.state_dict(), f'{CKPT_DIR}/unet_mnist32by32_ae_checkpoint_step_{step}.pth')

    finally:
        # Allow sleep again
        pass



    # Plot loss evolution for digit encoder
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MNIST U-Net AE Loss Evolution")
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


    # # Plot 3 examples of original and reconstructed test images
    # with torch.no_grad():
    #     x_original = test_set[:3]  # Take only the first 3 images for visualization
    #     x_noisy = x_original + torch.randn_like(x_original) * 0.02
    #     n_sensors = torch.randint(int(np.round(0.01*n_pixels)), int(np.round(0.08*n_pixels)), (3,), device=device)
    #     mask = torch.zeros((3, 1, image_size, image_size), device=device)

    #     for i in range(3):
    #         sampling_indices = torch.randperm(n_pixels, device=device)[:n_sensors[i].item()].long()
    #         mask.view(3, -1)[i, sampling_indices] = 1.0

    #     x_sampled = x_noisy * mask

    #     # Forward pass through the encoder
    #     x_recon = digit_encoder(x_sampled)  # [B, 1, img_size, img_size]

    #     fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    #     for i in range(3):
    #         axs[0, i].imshow(x_original[i, 0].detach().cpu().numpy(), cmap='gray')
    #         axs[0, i].set_title("Original")
    #         axs[0, i].axis('off')

    #         # Display the sampling mask
    #         axs[1, i].imshow(mask[i, 0].detach().cpu().numpy(), cmap='gray')
    #         axs[1, i].set_title(f"Sampling Mask\n({n_sensors[i].item()} sensors)")
    #         axs[1, i].axis('off')

    #         # Calculate PSNR for reconstruction
    #         original_np = x_original[i, 0].detach().cpu().numpy()
    #         recon_np = x_recon[i, 0].detach().cpu().numpy()
    #         mse = mean_squared_error(original_np, recon_np)
    #         psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
    #         ssim = structural_similarity(original_np, recon_np, data_range=1)

    #         axs[2, i].imshow(x_recon[i, 0].detach().cpu().numpy(), cmap='gray')
    #         axs[2, i].set_title(f"Reconstructed ({np.round(n_sensors[i].item()/n_pixels*100, 2)}% of px)\nPSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
    #         axs[2, i].axis('off')

    #     plt.tight_layout()
    #     plt.show()



    # ========================
    # Save the autoencoder
    # ========================
    # torch.save(digit_encoder.state_dict(), f'{CKPT_DIR}/unet_mnist28by28_ae_v9.pth')
    torch.save(digit_encoder.state_dict(), f'{CKPT_DIR}/unet_mnist32by32_ae_random-samp_v9.pth')
    print("The trained MNIST U-Net AE saved")


    # =================================================================
    # Evaluate the performace of the U-Net AE on the test MNIST set
    # =================================================================

    batch_size = 64
    # rates = [10, 25, 50, 100, 250, 500]
    rates = [100]
    test_mae_values = []
    test_psnr_values = []
    test_ssim_values = []

    # Initialize the model and load the trained weights
    digit_encoder = DigitFeatureEncoder().to(device)

    # Load the saved model checkpoint
    # digit_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist28by28_ae_v9.pth', map_location=device))
    digit_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist32by32_ae_random-samp_v9.pth', map_location=device))

    # Set the model to evaluation mode
    digit_encoder.eval()

    for rate in rates:

        test_mae_values = []
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
                x_recon = digit_encoder(x_sampled)

                for i in range(current_batch_size):
                    original_np = x_original[i, 0].detach().cpu().numpy()
                    recon_np = x_recon[i, 0].detach().cpu().numpy()
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1)
                    mae = np.mean(np.abs(original_np - recon_np))
                    test_mae_values.append(mae)
                    test_psnr_values.append(psnr)
                    test_ssim_values.append(ssim)

            if rate == 100:

                # Plot 3 examples of original and reconstructed test images
                fig, axs = plt.subplots(3, 3, figsize=(12, 12))

                for i in range(3):
                    axs[0, i].imshow(x_original[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[0, i].set_title("Original")
                    axs[0, i].axis('off')

                    # Display the sampling mask
                    axs[1, i].imshow(mask[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[1, i].set_title(f"Sampling Mask\n({n_sensors} sensors)")
                    axs[1, i].axis('off')

                    # Calculate PSNR for reconstruction
                    original_np = x_original[i, 0].detach().cpu().numpy()
                    recon_np = x_recon[i, 0].detach().cpu().numpy()
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1)

                    axs[2, i].imshow(x_recon[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[2, i].set_title(f"Reconstructed (100 px)\nPSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
                    axs[2, i].axis('off')

                plt.tight_layout()
                plt.show()

                # Plot histograms of MAE, PSNR, and SSIM values
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.hist(test_mae_values, bins=30, color='blue', alpha=0.7)
                plt.xlabel('MAE')
                plt.ylabel('Frequency')
                plt.title(f'MAE Distribution ({rate} sensors)')
                plt.grid(True)

                plt.subplot(1, 3, 2)
                plt.hist(test_psnr_values, bins=30, color='orange', alpha=0.7)
                plt.xlabel('PSNR (dB)')
                plt.ylabel('Frequency')
                plt.title(f'PSNR Distribution ({rate} sensors)')
                plt.grid(True)

                plt.subplot(1, 3, 3)
                plt.hist(test_ssim_values, bins=30, color='red', alpha=0.7)
                plt.xlabel('SSIM')
                plt.ylabel('Frequency')
                plt.title(f'SSIM Distribution ({rate} sensors)')
                plt.grid(True)

                plt.tight_layout()
                plt.show()


            avg_mae = np.mean(test_mae_values)
            avg_psnr = np.mean(test_psnr_values)
            avg_ssim = np.mean(test_ssim_values)
            print(f"=== Performance at {rate} sensors ({np.round(rate/n_pixels*100, 2)}% of pixels) ===")
            print(f"Average MAE on test set: {avg_mae:.6f}")
            print(f"Average PSNR on test set: {avg_psnr:.4f} dB")
            print(f"Average SSIM on test set: {avg_ssim:.4f}")
            print(" ")



    # ===============================================================
    # Train the MNIST classifier on randomly sampled measurements
    # ===============================================================

    # Define parameters
    batch_size = 64
    n_steps = 200*train_set.shape[0]//batch_size  # 10 full passes through the training set
    train_losses = []
    val_losses = []

    # Print the number of training steps
    print(f"Number of training steps for MNIST classifier: {n_steps}")

    # Initialize the classifier
    classifier = DigitClassifier(input_dim=n_pixels).to(device)
    print(f"Number of trainable parameters in the classifying model: {sum(p.numel() for p in classifier.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    # Prevent the system from going to sleep

    try:

        # Training loop
        for step in range(n_steps):

            # Draw a random batch of images from the training and validation sets
            batch_indices_train = np.random.choice(train_set.shape[0], size=batch_size, replace=False)
            x_batch_train = train_set[batch_indices_train]  # Remove the channel dimension
            train_labels_batch = train_labels[batch_indices_train]

            batch_indices_val = np.random.choice(validation_set.shape[0], size=batch_size, replace=False)
            x_batch_val = validation_set[batch_indices_val]
            val_labels_batch = validation_labels[batch_indices_val]

            # Add noise to the images
            x_batch_train = x_batch_train + torch.randn_like(x_batch_train) * 0.02
            x_batch_val = x_batch_val + torch.randn_like(x_batch_val) * 0.02

            # Randomly occlude some portion of pixels to simulate subsampling

            # Pick a random integer between 1% and 8% of the pixels to keep (same for all images in the batch)
            n_sensors = torch.randint(int(np.round(0.01*n_pixels)), int(np.round(0.08*n_pixels)), (1,), device=device).item()

            # Create a sampling mask
            mask_train = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            sampling_indices = torch.randperm(n_pixels, device=device)[:n_sensors].long()  # Randomly select n_sensors unique pixel indices to keep
            mask_train.view(batch_size, -1)[:, sampling_indices] = 1.0  # Set the selected pixel indices in the mask to 1 for all images in the batch

            mask_val = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            sampling_indices_val = torch.randperm(n_pixels, device=device)[:n_sensors].long()
            mask_val.view(batch_size, -1)[:, sampling_indices_val] = 1.0

            # Apply the mask to the images
            x_batch_train_subsampled = x_batch_train.clone()# * mask_train
            x_batch_val_subsampled = x_batch_val.clone()# * mask_val

            # Forward pass through the task model
            result_train = classifier(x_batch_train_subsampled)

            with torch.no_grad():
                result_val = classifier(x_batch_val_subsampled)

            # Calculate the losses
            loss_train = F.cross_entropy(result_train, train_labels_batch)
            train_losses.append(loss_train.item())
            loss_val = F.cross_entropy(result_val, val_labels_batch)
            val_losses.append(loss_val.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # Print the losses over epochs
            if (step+1) % (n_steps//10) == 0:
                print(f"[{step+1}] MNIST classifier Train Loss: {loss_train.item():.6e}, Val Loss: {loss_val.item():.6e}")

                # Save the model checkpoint
                # torch.save(classifier.state_dict(), f'{CKPT_DIR}/mnist28by28_classifier_checkpoint_step_{step+1}.pth')

    finally:
        # Allow sleep again
        pass



    # Plot the training and validation losses evolution
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MNIST Classifier Loss Evolution")
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



    # =======================
    # Save the classifier
    # =======================
    torch.save(classifier.state_dict(), f'{CKPT_DIR}/mnist28by28_classifier_random-samp_v9.pth')
    print("The trained MNIST classifier on randomly sampled measurements saved")


    # ======================================================================
    # Train GSE on the frozen MNIST classifier pre-trained on randomly sampled measurements
    # ======================================================================

    # Define parameters
    batch_size = 64
    n_steps = 200*train_set.shape[0]//batch_size  # 10 full passes through the training set
    eps = 1e-11 # parameter in Gumber noise sampling to prevent log(0)
    train_losses = []

    # Print the number of training steps
    print(f"Number of training steps for GSE (MNIST classification): {n_steps}")

    # Initialize the classifier
    classifier = DigitClassifier(input_dim=n_pixels).to(device)

    # Load the pre-trained weights
    classifier.load_state_dict(torch.load(f'{CKPT_DIR}/mnist28by28_classifier_random-samp_v9.pth', map_location=device))

    # Freeze the classifier weights
    for param in classifier.parameters():
        param.requires_grad = False

    classifier.eval()

    # Temperature anneaing parameters
    temp_init = 10.0  # Initial temperature for Gumbel-Softmax
    temp_end = 0.5  # Final temperature
    temp_step = (temp_init - temp_end) / n_steps  # Annealing rate

    # Initialize the subsampling logits (trainable matrix)
    logits = torch.nn.Parameter(torch.randn(n_pixels, device=device))  # [n_pixels]
    print(f"Number of trainable parameters in the GSE: {logits.numel()}")

    optimizer = torch.optim.Adam([logits], lr=1e-3)

    # Prevent the system from going to sleep

    try:

        # Training loop
        for step in range(n_steps):

            # Set the temperature for this step
            if step == 0:
                temp = temp_init

            # Draw a random batch of images from the training and validation sets
            batch_indices_train = np.random.choice(train_set.shape[0], size=batch_size, replace=False)
            x_batch_train = train_set[batch_indices_train]  # Remove the channel dimension
            train_labels_batch = train_labels[batch_indices_train]

            # Add noise to the images
            x_batch_train = x_batch_train + torch.randn_like(x_batch_train) * 0.02

            # Apply Gumbel noise to the logits
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, device=device) + eps) + eps)
            noisy_logits = logits + gumbel_noise

            # Pick a random integer between 1% and 8% of the pixels to keep (same for all images in the batch)
            n_sensors = torch.randint(int(np.round(0.01*n_pixels)), int(np.round(0.08*n_pixels)), (1,), device=device).item()

            # Hard mask selection for the forward pass (discrete sampling) with zero filling
            _, top_indices = torch.topk(noisy_logits, n_sensors)

            # Create a hard sampling mask
            hard_mask = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            hard_mask.view(batch_size, -1)[:, top_indices] = 1.0

            # Create a soft sampling mask
            soft_mask = torch.softmax(noisy_logits / temp, dim=0).view(1, 1, image_size, image_size)
            soft_mask = soft_mask.expand(batch_size, -1, -1, -1)  # Expand to match batch size
            mask_train = hard_mask - soft_mask.detach() + soft_mask  # Straight-through estimator

            # Apply the mask to the images
            x_batch_train_subsampled = x_batch_train * mask_train

            # Forward pass through the task model
            result_train = classifier(x_batch_train_subsampled)

            # Calculate the losses
            loss_train = F.cross_entropy(result_train, train_labels_batch)
            train_losses.append(loss_train.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_([logits], max_norm=1.0)  # Gradient clipping for stability
            optimizer.step()

            # Update the temperature
            temp = max(temp_end, temp - temp_step)

            # Print the losses over epochs
            if (step+1) % (n_steps//20) == 0:
                print(f"[{step+1}] GSE (classifier) Train Loss: {loss_train.item():.6e}")

                # Save the model checkpoint
                torch.save(logits.detach().cpu(), f'{CKPT_DIR}/gse_mnist28by28_classifier_logits_checkpoint_step_{step+1}.pth')

    finally:
        # Allow sleep again
        pass



    # Plot the training and validation losses evolution
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("GSE (MNIST Classification) Loss Evolution")
    plt.legend()
    # plt.xscale('log')
    plt.yscale('log')

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

    # Plot the learned sampling distribution
    with torch.no_grad():
        learned_probs = torch.softmax(logits, dim=0).cpu().numpy().reshape(image_size, image_size)
        plt.figure(figsize=(6, 6))
        plt.imshow(learned_probs, cmap='gray')
        plt.colorbar(label='Learned Sampling Probability')
        plt.title("Learned Sampling Distribution (GSE for MNIST Classification)")
        plt.axis('off')
        plt.show()


    # ===========================
    # Save the learned logits
    # ===========================
    torch.save(logits.detach().cpu(), f'{CKPT_DIR}/gse_mnist28by28_classifier_logits_v9.pth')
    print("The learned GSE logits for MNIST classification saved")



    # =============================================================
    # Evaluate the MNIST classifier performance on the test set
    # =============================================================

    batch_size = 64
    rates = np.arange(8, 0, -1)/100 # Different sampling rates to evaluate the performance at (from 8% to 1% of pixels)

    # # Initialize the classifier and load the pre-trained weights
    # classifier = DigitClassifier(input_dim=n_pixels).to(device)
    # classifier.load_state_dict(torch.load(f'{CKPT_DIR}/mnist28by28_classifier_random-samp_v9.pth', map_location=device))

    # Set the model to evaluation mode
    classifier.eval()

    sampling_rate_vs_label_accuracy = []

    for rate in rates:
        test_accuracies = []
        test_accuracies_per_label = {i: [] for i in range(10)} 

        with torch.no_grad():
            for i in range(0, test_set.shape[0], batch_size):
                end_idx = min(i + batch_size, test_set.shape[0])
                current_batch_size = end_idx - i
                x_original = test_set[i:end_idx]
                x_noisy = x_original.clone() # + torch.randn_like(x_original) * 0.02
                n_sensors = int(np.round(rate*n_pixels))
                mask = torch.zeros((current_batch_size, 1, image_size, image_size), device=device)

                for b_i in range(current_batch_size):
                    sampling_indices = torch.randperm(n_pixels, device=device)[:n_sensors].long()
                    mask.view(current_batch_size, -1)[b_i, sampling_indices] = 1.0

                x_sampled = x_noisy.clone()# * mask
                test_results = classifier(x_sampled)

                # Calculate the accuracy for this batch
                _, predicted_labels = torch.max(test_results, 1)
                correct_predictions = (predicted_labels == test_labels[i:end_idx]).sum().item()
                accuracy = correct_predictions / current_batch_size * 100

                # Track accuracy per digit label
                for label in range(10):
                    label_mask = (test_labels[i:end_idx] == label)
                    if label_mask.sum() > 0:
                        label_accuracy = (predicted_labels[label_mask] == test_labels[i:end_idx][label_mask]).sum().item() / label_mask.sum().item() * 100
                        test_accuracies_per_label[label].append(label_accuracy)

                test_accuracies.append(accuracy)

        # Print the accuracy
        print(f"Average Test Accuracy of the Classifier alone, rate {100*rate:.2f}%: {sum(test_accuracies) / len(test_accuracies):.2f}%")

        # Print average accuracy per digit label
        print(f"Per Digit Label:")
        rate_row = [100 * rate]
        for label in range(10):
            if len(test_accuracies_per_label[label]) > 0:
                avg_label_accuracy = sum(test_accuracies_per_label[label]) / len(test_accuracies_per_label[label])
                print(f"  Digit {label}: {avg_label_accuracy:.2f}%")
                rate_row.append(np.round(avg_label_accuracy, 0))

            else:
                print(f"  Digit {label}: No samples in test set")
                rate_row.append(np.nan)

        sampling_rate_vs_label_accuracy.append(rate_row)
        print('\n')

    # csv_header = 'sampling_rate_percent,' + ','.join([f'digit_{label}' for label in range(10)])
    # np.savetxt(
    #     f'{CKPT_DIR}/mnist_classifier_sampling_rate_vs_label_accuracy_v9.csv',
    #     np.array(sampling_rate_vs_label_accuracy, dtype=float),
    #     delimiter=',',
    #     header=csv_header,
    #     comments=''
    # )
    # print('Saved sampling-rate-vs-label-accuracy matrix to supplementary/mnist_classifier_sampling_rate_vs_label_accuracy_v9.csv')



    # ======================================================================
    # Evaluate the MNIST classifier + GSE mask performance on the test set
    # ======================================================================

    batch_size = 64
    rates = np.arange(8, 0, -1)/100 

    # Initialize the classifier and load the pre-trained weights
    classifier = DigitClassifier(input_dim=n_pixels).to(device)
    classifier.load_state_dict(torch.load(f'{CKPT_DIR}/mnist28by28_classifier_random-samp_v9.pth', map_location=device))

    # Initialize the GSE logits and create the sampling mask based on the learned distribution
    logits = torch.load(f'{CKPT_DIR}/gse_mnist28by28_classifier_logits_v9.pth', map_location=device)

    # Set the models to evaluation mode
    classifier.eval()
    logits.requires_grad_(False)

    sampling_rate_vs_label_accuracy = []

    for rate in rates:
        test_accuracies = []
        test_accuracies_per_label = {i: [] for i in range(10)} 

        with torch.no_grad():
            for i in range(0, test_set.shape[0], batch_size):
                end_idx = min(i + batch_size, test_set.shape[0])
                current_batch_size = end_idx - i        
                x_original = test_set[i:end_idx]
                x_noisy = x_original.clone() # + torch.randn_like(x_original) * 0.02
                n_sensors = int(np.round(rate*n_pixels))
                mask = torch.zeros((current_batch_size, 1, image_size, image_size), device=device)

                for b_i in range(current_batch_size):
                    # Sample pixels based on the learned GSE distribution
                    # probs = torch.softmax(logits, dim=0)
                    # sampling_indices = torch.multinomial(probs, n_sensors[b_i].item(), replacement=False).long()
                    _, sampling_indices = torch.topk(logits, n_sensors)
                    mask.view(current_batch_size, -1)[b_i, sampling_indices] = 1.0

                x_sampled = x_noisy * mask
                test_results = classifier(x_sampled)

                # Calculate the accuracy for this batch
                _, predicted_labels = torch.max(test_results, 1)
                correct_predictions = (predicted_labels == test_labels[i:end_idx]).sum().item()
                accuracy = correct_predictions / current_batch_size * 100

                # Track accuracy per digit label
                for label in range(10):
                    label_mask = (test_labels[i:end_idx] == label)
                    if label_mask.sum() > 0:
                        label_accuracy = (predicted_labels[label_mask] == test_labels[i:end_idx][label_mask]).sum().item() / label_mask.sum().item() * 100
                        test_accuracies_per_label[label].append(label_accuracy)

                test_accuracies.append(accuracy)

        # Print the accuracy
        print(f"Average Test Accuracy on the GSE Mask, rate {100*rate:.2f}%: {sum(test_accuracies) / len(test_accuracies):.2f}%")

        # Print average accuracy per digit label
        print(f"Per Digit Label:")
        rate_row = [100 * rate]
        for label in range(10):
            if len(test_accuracies_per_label[label]) > 0:
                avg_label_accuracy = sum(test_accuracies_per_label[label]) / len(test_accuracies_per_label[label])
                print(f"  Digit {label}: {avg_label_accuracy:.2f}%")
                rate_row.append(np.round(avg_label_accuracy, 0))

            else:
                print(f"  Digit {label}: No samples in test set")
                rate_row.append(np.nan)

        sampling_rate_vs_label_accuracy.append(rate_row)
        print('\n')

    csv_header = 'sampling_rate_percent,' + ','.join([f'digit_{label}' for label in range(10)])
    np.savetxt(
        f'{CKPT_DIR}/mnist_classifier_sampling_rate_vs_label_accuracy_v9.csv',
        np.array(sampling_rate_vs_label_accuracy, dtype=float),
        delimiter=',',
        header=csv_header,
        comments=''
    )
    print('Saved sampling-rate-vs-label-accuracy matrix to supplementary/mnist_classifier_sampling_rate_vs_label_accuracy_v9.csv')



    # ======================================================================
    # Train the flow model for the subsampled MNIST classification through the pre-trained classifier
    # ======================================================================

    batch_size = 64
    n_steps = 10*train_set.shape[0]//batch_size
    train_fm_losses = []
    train_task_losses = []
    train_losses = []
    joint_training = True # flag whether to train the flow and task models jointly or not
    sig_1_values = []
    sig_2_values = []
    # sig_3_values = []

    # Print number of steps
    print(f"Number of steps for flow model training: {n_steps}")

    # Initialize the trainable loss uncertainty parameters
    # log_sig_1 = torch.nn.Parameter(torch.zeros(1, device=device))  # log of the standard deviation for the flow model loss
    # log_sig_2 = torch.nn.Parameter(torch.zeros(1, device=device))  # log of the standard deviation for the task model loss
    # log_sig_3 = torch.nn.Parameter(torch.zeros(1, device=device))  # log of the standard deviation for the contrastive loss
    # log_sig_1 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(4))

    # Sigmoid steepness function parameters
    steepness = 30
    power = 1

    # Initialize the classifier
    classifier = DigitClassifier(input_dim=n_pixels).to(device)

    if joint_training == False:

        # Load the pre-trained weights
        classifier.load_state_dict(torch.load(f'{CKPT_DIR}/mnist28by28_classifier_random-samp_v9.pth', map_location=device))
        classifier.freeze_parameters()
        classifier.eval()

    # Initialize the U-Net-based feature extractor and load the pre-trained weights
    digit_encoder = DigitFeatureEncoder().to(device)
    digit_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist28by28_ae_v9.pth', map_location=device))
    digit_encoder.eval()

    # Load the GT estimator - GSE learned sampling distribution
    gse_logits = torch.load(f'{CKPT_DIR}/gse_mnist28by28_classifier_logits_v9.pth', map_location=device)
    gse_probs = torch.softmax(gse_logits, dim=0)

    # Initialize the flow model
    flow_model = CondFlow(encoder=digit_encoder, norm_type='bn').to(device)

    # Print the number of trainable parameters in the flow model
    num_params_fm = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the flow model: {num_params_fm}")

    if joint_training == True:
        # Print the number of trainable parameters in the task model
        num_params_task = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the classifying model: {num_params_task}")

    # # Initialize the optimizer
    # if joint_training == True:
    #     optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(classifier.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)

    # else:
    #     optimizer = torch.optim.Adam(list(flow_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)


    # Prevent the system from going to sleep

    try:

        # Training loop
        for step in range(n_steps):

            # Draw a random batch of images from the training and validation sets
            batch_indices_train = np.random.choice(train_set.shape[0], size=batch_size, replace=False)
            x_batch_train = train_set[batch_indices_train]
            train_labels_batch = train_labels[batch_indices_train]

            # # Draw a random batch of images from the contrastive loss
            # batch_indices_contrastive = np.random.choice(train_set.shape[0], size=batch_size, replace=False)
            # x_batch_contrastive = train_set[batch_indices_contrastive]
            # contrastive_labels_batch = train_labels[batch_indices_contrastive]

            # Add noise to the images
            x_batch_train = x_batch_train + torch.randn_like(x_batch_train) * 0.02
            # x_batch_contrastive = x_batch_contrastive + torch.randn_like(x_batch_contrastive) * 0.02

            # Pick a random integer between 1% and 8% of the pixels to keep (same for all images in the batch)
            n_sensors = torch.randint(int(np.round(0.01*n_pixels)), int(np.round(0.08*n_pixels)), (1,), device=device).item()

            # # OR only pick 1% every iteration
            # n_sensors = int(np.round(0.01*n_pixels))

            # Create a random sampling mask for conditioning the flow model
            mask_random = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            sampling_indices_random = torch.randperm(n_pixels, device=device)[:n_sensors].long()
            mask_random.view(batch_size, -1)[:, sampling_indices_random] = 1.0

            # Create a conditioning batch
            x_batch_train_conditioning = x_batch_train * mask_random
            # x_batch_contrastive_conditioning = x_batch_contrastive * mask_random

            # Sample the GT mask estimation from the GSE learned distribution
            mask_gse = torch.zeros((batch_size, 1, image_size, image_size), device=device)

            for i in range(batch_size):
                sampling_indices_gse = torch.multinomial(gse_probs, n_sensors, replacement=False).long()
                mask_gse.view(batch_size, -1)[i, sampling_indices_gse] = 1.0

            # Sample noise x0 and time t
            x0 = torch.randn(batch_size, 1, image_size, image_size).to(device)  # Reshape to match image dimensions
            t = torch.rand(batch_size, 1).to(device)

            # Linear coupling between x0 and x1 (GSE mask) to create a continuous path for the flow model to learn
            xt = x0 * (1 - t.view(batch_size, 1, 1, 1)) + mask_gse * t.view(batch_size, 1, 1, 1)

            # GT velocity for the flow model (derivative of the linear coupling)
            gt_velocity = mask_gse - x0

            # Forward pass through the flow model
            pred_velocity = flow_model(xt.view(batch_size, -1), t, x_batch_train_conditioning).view(batch_size, 1, image_size, image_size)
            # pred_velocity_contrastive = flow_model(xt.view(batch_size, -1), t, x_batch_contrastive_conditioning).view(batch_size, 1, image_size, image_size)
            pred_x = x0 + pred_velocity * t.view(batch_size, 1, 1, 1)
            # pred_x = xt + pred_velocity * t.view(batch_size, 1, 1, 1)

            # Apply the sigmoid gate to create soft masks
            soft_mask = torch.zeros_like(pred_x)

            for i in range(batch_size):
                threshold = torch.quantile(pred_x[i].view(-1), 1 - n_sensors/n_pixels)
                soft_mask[i] = torch.sigmoid(steepness * (pred_x[i] - threshold) * (t[i].item() ** power))

            # Pass the entire batch through the task model
            x_batch_train_soft_sampled = x_batch_train * soft_mask
            result_soft = classifier(x_batch_train_soft_sampled)

            # Calculate the losses
            fm_loss = F.mse_loss(pred_velocity, gt_velocity) # Compute FM loss (MSE between predicted and GT velocities)
            task_loss = F.cross_entropy(result_soft, train_labels_batch) # Compute task loss (cross-entropy for classification)
            # contrastive_loss = F.mse_loss(pred_velocity, pred_velocity_contrastive) # Compute contrastive loss
            train_fm_losses.append(fm_loss.item())
            train_task_losses.append(task_loss.item())

            # At the first iteration, initialize the log_sig parameters and the optimizer after seeing the scale of the losses
            if step == 0:

                # Initialize the trainable loss uncertainty parameters
                sig_1_init = np.sqrt(np.abs(fm_loss.item()))
                sig_2_init = np.sqrt(np.abs(task_loss.item()))
                log_sig_1 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(sig_1_init))
                log_sig_2 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(sig_2_init))

                # Initialize the optimizer
                if joint_training == True:
                    # optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(classifier.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)
                    optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(classifier.parameters()), lr=1e-4)

                else:
                    optimizer = torch.optim.Adam(list(flow_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)

            sig_1 = torch.exp(log_sig_1)
            sig_2 = torch.exp(log_sig_2)
            # sig_3 = torch.exp(log_sig_3)
            # total_loss = 1/(2*torch.pow(sig_1, 2)) * fm_loss + 1/(2*torch.pow(sig_2, 2)) * task_loss + log_sig_1 + log_sig_2  # Total loss with learned uncertainty weighting
            # total_loss = 1/(2*torch.pow(sig_1, 2)) * fm_loss + 1/(2*torch.pow(sig_2, 2)) * task_loss + 1/(2*torch.pow(sig_3, 2)) * contrastive_loss + log_sig_1 + log_sig_2 + log_sig_3  # Total loss with learned uncertainty weighting
            total_loss = F.cross_entropy(result_soft, train_labels_batch)

            # Track uncertainty parameters and the total training loss
            sig_1_values.append(sig_1.item())
            sig_2_values.append(sig_2.item())
            # sig_3_values.append(sig_3.item())
            train_losses.append(total_loss.item())

            # Backpropagation and optimization step
            optimizer.zero_grad()
            total_loss.backward()

            if joint_training == True:
                # torch.nn.utils.clip_grad_norm_(list(flow_model.parameters()) + list(classifier.parameters()) + [log_sig_1, log_sig_2], max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(list(flow_model.parameters()) + list(classifier.parameters()), max_norm=1.0)

            else:
                torch.nn.utils.clip_grad_norm_(list(flow_model.parameters()) + [log_sig_1, log_sig_2], max_norm=1.0)

            optimizer.step()

            # Print the progress
            if (step+1) % (n_steps//20) == 0:
                print(f"[{(step+1)/n_steps:.2%}] Total Loss: {total_loss.item():.6e}, FM Loss: {fm_loss.item():.6e}, Task Loss: {task_loss.item():.6e}, Sig_1: {sig_1_values[-1]:.4f}, Sig_2: {sig_2_values[-1]:.4f}")


                # # Save the model checkpoint
                # torch.save({
                #     'flow_model_state_dict': flow_model.state_dict(),
                #     'classifier_state_dict': classifier.state_dict(),
                #     'log_sig_1': log_sig_1,
                #     'log_sig_2': log_sig_2
                # }, f'{CKPT_DIR}/flow_mnist28by28_classifier_checkpoint_step_{step+1}.pth')


    finally:
        # Allow sleep again
        pass


    # Plot the training FM, task, and total losses evolution
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_fm_losses, color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("FM Loss")
    plt.title("Flow Model Loss Evolution")
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')

    plt.subplot(1, 3, 2)
    plt.plot(train_task_losses, color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("Task Loss")
    plt.title("Task Loss Evolution")
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')

    plt.subplot(1, 3, 3)
    plt.plot(train_losses, color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Total Loss Evolution")
    plt.legend()
    plt.grid(True)
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
    # plt.plot(sig_3_values, label='Sigma 3 (Contrastive)', alpha=0.7)
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
            'classifier_state_dict': classifier.state_dict(),
            'log_sig_1': log_sig_1,
            'log_sig_2': log_sig_2
        }, f'{CKPT_DIR}/joint_flow_mnist28by28_classifier_v9.pth')
        print("The flow model jointly trained with the classifier saved")

    else:
        torch.save({
            'flow_model_state_dict': flow_model.state_dict(),
            'log_sig_1': log_sig_1,
            'log_sig_2': log_sig_2
        }, f'{CKPT_DIR}/test_flow_mnist28by28_classifier_v9.pth')
        print("The flow model saved")



    # ======================================================================
    # Evaluate the MNIST classifier + flow generated mask performance on the test set
    # ======================================================================

    # joint_training = True
    # steepness = 20
    # power = 2
    time_points = torch.linspace(0, 1, 20).to(device) # ODE integration time points for the flow model inference
    # rates = np.arange(8, 0, -1)/100
    # rates = [0.08, 0.04, 0.01]
    rates = [0.08]


    # # Initialize the classifier and load the pre-trained weights
    # classifier = DigitClassifier(input_dim=n_pixels).to(device)

    # if joint_training == False:
    #     classifier.load_state_dict(torch.load(f'{CKPT_DIR}/mnist28by28_classifier_random-samp_v9.pth', map_location=device))

    # else:
    #     checkpoint = torch.load(f'{CKPT_DIR}/joint_flow_mnist28by28_classifier_v9.pth', map_location=device)
    #     classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # # Initialize the U-Net-based feature extractor and load the pre-trained weights
    # digit_encoder = DigitFeatureEncoder().to(device)
    # digit_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist28by28_ae_v9.pth', map_location=device))
    # digit_encoder.eval()

    # # Initialize the flow model and load the trained weights
    # flow_model = CondFlow(encoder=digit_encoder, norm_type='bn').to(device)

    # if joint_training == False:
    #     flow_checkpoint = torch.load(f'{CKPT_DIR}/flow_mnist28by28_classifier_v9.pth', map_location=device)
    #     flow_model.load_state_dict(flow_checkpoint['flow_model_state_dict'])

    # else:
    #     flow_checkpoint = torch.load(f'{CKPT_DIR}/joint_flow_mnist28by28_classifier_v9.pth', map_location=device)
    #     flow_model.load_state_dict(flow_checkpoint['flow_model_state_dict'])


    # Set the models to evaluation mode
    classifier.eval()
    flow_model.eval()

    sampling_rate_vs_label_accuracy = []

    for rate in rates:

        test_accuracies = []
        test_accuracies_per_label = {i: [] for i in range(10)}  # To track accuracies per digit label
        flow_masks = []

        with torch.no_grad():
            for i in range(0, test_set.shape[0], batch_size):
                end_idx = min(i + batch_size, test_set.shape[0])
                current_batch_size = end_idx - i
                x_original = test_set[i:end_idx]
                x_noisy = x_original.clone() # + torch.randn_like(x_original) * 0.02
                n_sensors = int(np.round(rate*n_pixels))
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
                test_results = classifier(x_sampled)

                # Calculate the accuracy for this batch
                _, predicted_labels = torch.max(test_results, 1)
                correct_predictions = (predicted_labels == test_labels[i:end_idx]).sum().item()
                accuracy = correct_predictions / current_batch_size * 100
                test_accuracies.append(accuracy)

                # # Track accuracy per digit label
                # for label in range(10):
                #     label_mask = (test_labels[i:end_idx] == label)
                #     if label_mask.sum() > 0:
                #         label_accuracy = (predicted_labels[label_mask] == test_labels[i:end_idx][label_mask]).sum().item() / label_mask.sum().item() * 100
                #         test_accuracies_per_label[label].append(label_accuracy)


            # Concatenate the flow masks for the entire test set
            flow_masks = torch.cat(flow_masks, dim=0)

            if rate == rates[0]:

                # Plot a sample of learned masks
                plt.figure(figsize=(15, 10))
                for i in range(6):
                    plt.subplot(2, 3, i + 1)
                    sample_mask = flow_masks[i].cpu().numpy().reshape(image_size, image_size)
                    plt.imshow(sample_mask, cmap='gray')
                    plt.title(f'Flow Mask {i+1}')
                    plt.axis('off')

                plt.suptitle(f'Examples of the Flow Learne Masks (r={100*rate:.2f}% subsampling)')
                plt.tight_layout()
                plt.show()

                # Plot average mask
                avg_mask = flow_masks.mean(dim=0).cpu().numpy().reshape(image_size, image_size)
                plt.figure(figsize=(8, 8))
                plt.imshow(avg_mask, cmap='gray')
                plt.title(f'Average Flow Mask ({100*rate:.1f}% subsampling)')
                plt.colorbar()
                plt.show()        

                # t-SNE visualization for the generated masks at each subsampling rate

                # Standardize the mask features before t-SNE
                scaler = StandardScaler()
                flow_masks_scaled = scaler.fit_transform(flow_masks.squeeze().view(test_set.shape[0], -1).cpu().numpy())

                # Apply t-SNE
                tsne = TSNE(n_components=2, perplexity=30)
                flow_masks_tsne = tsne.fit_transform(flow_masks_scaled)

                # Create scatter plot with digit labels as colors
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(flow_masks_tsne[:, 0], flow_masks_tsne[:, 1], 
                                    c=test_labels.cpu().numpy(), cmap='tab10', 
                                    s=20, alpha=0.6, edgecolors='k', linewidth=0.5)

                # Add colorbar with digit labels
                cbar = plt.colorbar(scatter, ticks=range(10))
                cbar.set_label('Digit Label', fontsize=12)

                plt.xlabel('t-SNE Component 1', fontsize=12)
                plt.ylabel('t-SNE Component 2', fontsize=12)
                plt.title(f't-SNE Visualization of Generated Masks\n' + 
                        f'{100*rate:.2f}% subsampling), Colored by Digit Label',
                        fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()


        # Print the accuracy
        print(f"Average Test Accuracy on the Flow Mask, rate {100*rate:.2f}%: {sum(test_accuracies) / len(test_accuracies):.2f}%")

        # # Print average accuracy per digit label
        # print(f"Per Digit Label:")
        # rate_row = [100 * rate]
        # for label in range(10):
        #     if len(test_accuracies_per_label[label]) > 0:
        #         avg_label_accuracy = sum(test_accuracies_per_label[label]) / len(test_accuracies_per_label[label])
        #         print(f"  Digit {label}: {avg_label_accuracy:.2f}%")
        #         rate_row.append(np.round(avg_label_accuracy, 0))

        #     else:
        #         print(f"  Digit {label}: No samples in test set")
        #         rate_row.append(np.nan)

        # sampling_rate_vs_label_accuracy.append(rate_row)
        # print('\n')

    # csv_header = 'sampling_rate_percent,' + ','.join([f'digit_{label}' for label in range(10)])
    # np.savetxt(
    #     f'{CKPT_DIR}/mnist_classifier_sampling_rate_vs_label_accuracy_v9.csv',
    #     np.array(sampling_rate_vs_label_accuracy, dtype=float),
    #     delimiter=',',
    #     header=csv_header,
    #     comments=''
    # )
    # print('Saved sampling-rate-vs-label-accuracy matrix to supplementary/mnist_classifier_sampling_rate_vs_label_accuracy_v9.csv')






    # ======================================================================
    # Train GSE on the frozen MNIST AE pre-trained on randomly sampled measurements
    # ======================================================================

    # Define parameters
    batch_size = 64
    n_steps = 200*train_set.shape[0]//batch_size  # 10 full passes through the training set
    eps = 1e-11 # parameter in Gumber noise sampling to prevent log(0)
    train_losses = []

    # Print the number of training steps
    print(f"Number of training steps for GSE (MNIST reconstruction): {n_steps}")

    # Instantiate the AE model
    digit_encoder = DigitFeatureEncoder().to(device)

    # Load the pre-trained weights
    digit_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist32by32_ae_random-samp_v9.pth', map_location=device))

    # Freeze the AE weights
    for param in digit_encoder.parameters():
        param.requires_grad = False

    digit_encoder.eval()

    # Temperature anneaing parameters
    temp_init = 10.0  # Initial temperature for Gumbel-Softmax
    temp_end = 0.5  # Final temperature
    temp_step = (temp_init - temp_end) / n_steps  # Annealing rate

    # Initialize the subsampling logits (trainable matrix)
    logits = torch.nn.Parameter(torch.randn(n_pixels, device=device))  # [n_pixels]

    # Initialize the subsampling logits with soft start (from the training data density)
    # logits = torch.nn.Parameter(torch.mean(train_set, dim=0).view(-1) + 1e-2)  # Adding a small constant to avoid zero probabilities

    print(f"Number of trainable parameters in the GSE: {logits.numel()}")

    optimizer = torch.optim.Adam([logits], lr=1e-3, maximize=True)
    # optimizer = torch.optim.Adam(list(digit_encoder.parameters()) + [logits], lr=1e-4)

    # Prevent the system from going to sleep

    try:

        # Training loop
        for step in range(n_steps):

            # Set the temperature for this step
            if step == 0:
                temp = temp_init

            # Draw a random batch of images from the training and validation sets
            batch_indices_train = np.random.choice(train_set.shape[0], size=batch_size, replace=False)
            x_batch_train = train_set[batch_indices_train]  # Remove the channel dimension

            # Add noise to the images
            x_batch_train_noisy = x_batch_train + torch.randn_like(x_batch_train) * 0.02

            # Apply Gumbel noise to the logits
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, device=device) + eps) + eps)
            noisy_logits = logits + gumbel_noise

            # Pick a random integer between 10 and 500 of the pixels to keep (same for all images in the batch)
            n_sensors = torch.randint(10, 500, (1,), device=device).item()

            # Hard mask selection for the forward pass (discrete sampling) with zero filling
            _, top_indices = torch.topk(noisy_logits, n_sensors)

            # Create a hard sampling mask
            hard_mask = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            hard_mask.view(batch_size, -1)[:, top_indices] = 1.0

            # Create a soft sampling mask
            soft_mask = torch.softmax(noisy_logits / temp, dim=0).view(1, 1, image_size, image_size)
            soft_mask = soft_mask.expand(batch_size, -1, -1, -1)  # Expand to match batch size
            mask_train = hard_mask - soft_mask.detach() + soft_mask  # Straight-through estimator

            # Apply the mask to the images
            x_batch_train_subsampled = x_batch_train_noisy * mask_train

            # Forward pass through the task model
            x_recon = digit_encoder(x_batch_train_subsampled)

            # Calculate the losses
            # loss_train = F.mse_loss(x_recon, x_batch_train)
            loss_train = F.l1_loss(x_recon, x_batch_train)
            train_losses.append(loss_train.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_([logits], max_norm=1.0)  # Gradient clipping for stability
            optimizer.step()

            # Update the temperature
            temp = max(temp_end, temp - temp_step)

            # Print the losses over epochs
            if (step+1) % (n_steps//20) == 0:
                print(f"[{step+1}] GSE (AE) Train Loss: {loss_train.item():.6e}")

                # Save the model checkpoint
                # torch.save(logits.detach().cpu(), f'{CKPT_DIR}/gse_mnist32by32_ae_logits_checkpoint_step_{step+1}.pth')

    finally:
        # Allow sleep again
        pass



    # Plot the training and validation losses evolution
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("GSE (MNIST Reconstruction) Loss Evolution")
    plt.legend()
    # plt.xscale('log')
    plt.yscale('log')

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

    # Plot the learned sampling distribution
    with torch.no_grad():
        # learned_probs = torch.softmax(logits, dim=0).cpu().numpy().reshape(image_size, image_size)
        plt.figure(figsize=(6, 6))
        # plt.imshow(learned_probs, cmap='gray')
        plt.imshow(logits.view(image_size, image_size).cpu().numpy(), cmap='gray')
        plt.colorbar(label='Learned Sampling Probability')
        plt.title("Learned Sampling Distribution (GSE for MNIST Reconstruction)")
        plt.axis('off')
        plt.show()


    # ===========================
    # Save the learned logits
    # ===========================
    torch.save(logits.detach().cpu(), f'{CKPT_DIR}/gse_mnist32by32_ae_logits_v9.pth')
    print("The learned GSE logits for MNIST reconstruction saved")


    # ======================================================================
    # Evaluate the performace of the U-Net AE + GSE mask on the test MNIST set
    # ======================================================================

    batch_size = 64
    rates = [10, 25, 50, 100, 250, 500]
    # rates = [100]

    # Instantiate the AE model
    digit_encoder = DigitFeatureEncoder().to(device)

    # Load the pre-trained weights
    digit_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist32by32_ae_random-samp_v9.pth', map_location=device))

    # Freeze the AE weights
    for param in digit_encoder.parameters():
        param.requires_grad = False

    # Initialize the GSE logits and create the sampling mask based on the learned distribution
    logits = torch.load(f'{CKPT_DIR}/gse_mnist32by32_ae_logits_v9.pth', map_location=device)

    # Set the models to evaluation mode
    digit_encoder.eval()
    logits.requires_grad_(False)

    for rate in rates:

        test_mae_values = []
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
                    _, sampling_indices = torch.topk(logits, n_sensors)
                    mask.view(current_batch_size, -1)[i, sampling_indices] = 1.0

                x_sampled = x_noisy * mask
                x_recon = digit_encoder(x_sampled)

                for i in range(current_batch_size):
                    original_np = x_original[i, 0].detach().cpu().numpy()
                    recon_np = x_recon[i, 0].detach().cpu().numpy()
                    mae = np.mean(np.abs(original_np - recon_np))
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1)
                    test_mae_values.append(mae)
                    test_psnr_values.append(psnr)
                    test_ssim_values.append(ssim)

            if rate == 100:

                # Plot 3 examples of original and reconstructed test images
                fig, axs = plt.subplots(3, 3, figsize=(12, 12))

                for i in range(3):
                    axs[0, i].imshow(x_original[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[0, i].set_title("Original")
                    axs[0, i].axis('off')

                    # Display the sampling mask
                    axs[1, i].imshow(mask[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[1, i].set_title(f"Sampling Mask (GSE)\n({n_sensors} sensors)")
                    axs[1, i].axis('off')

                    # Calculate PSNR for reconstruction
                    original_np = x_original[i, 0].detach().cpu().numpy()
                    recon_np = x_recon[i, 0].detach().cpu().numpy()
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1)

                    axs[2, i].imshow(x_recon[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[2, i].set_title(f"Reconstructed (100 px)\nPSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
                    axs[2, i].axis('off')

                plt.tight_layout()
                plt.show()

                # Plot histograms of MAE, PSNR, and SSIM values
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.hist(test_mae_values, bins=30, color='blue', alpha=0.7)
                plt.xlabel('MAE')
                plt.ylabel('Frequency')
                plt.title(f'MAE Distribution ({rate} sensors)')
                plt.grid(True)

                plt.subplot(1, 3, 2)
                plt.hist(test_psnr_values, bins=30, color='orange', alpha=0.7)
                plt.xlabel('PSNR (dB)')
                plt.ylabel('Frequency')
                plt.title(f'PSNR Distribution ({rate} sensors)')
                plt.grid(True)

                plt.subplot(1, 3, 3)
                plt.hist(test_ssim_values, bins=30, color='red', alpha=0.7)
                plt.xlabel('SSIM')
                plt.ylabel('Frequency')
                plt.title(f'SSIM Distribution ({rate} sensors)')
                plt.grid(True)

                plt.tight_layout()
                plt.show()


            avg_mae = np.mean(test_mae_values)
            avg_psnr = np.mean(test_psnr_values)
            avg_ssim = np.mean(test_ssim_values)
            print(f"=== Performance at {rate} sensors ({np.round(rate/n_pixels*100, 2)}% of pixels for GSE mask) ===")
            print(f"Average MAE on test set: {avg_mae:.6f}")
            print(f"Average PSNR on test set: {avg_psnr:.4f} dB")
            print(f"Average SSIM on test set: {avg_ssim:.4f}")
            print(" ")


    # ======================================================================
    # Train the flow model for the subsampled MNIST reconstruction through the pre-trained AE
    # ======================================================================
    batch_size = 64
    n_steps = 20*train_set.shape[0]//batch_size
    train_losses = []
    joint_training = True # flag whether to train the flow and task models jointly or not
    sig_1_values = []
    sig_2_values = []

    # Print number of steps
    print(f"Number of steps for flow model training: {n_steps}")

    # Initialize the trainable loss uncertainty parameters
    # log_sig_1 = torch.nn.Parameter(torch.zeros(1, device=device))  # log of the standard deviation for the flow model loss
    log_sig_1 = torch.nn.Parameter(torch.ones(1, device=device) * np.log(4))
    log_sig_2 = torch.nn.Parameter(torch.zeros(1, device=device))  # log of the standard deviation for the task model loss

    # Sigmoid steepness function parameters
    steepness = 20
    power = 2

    # Instantiate the AE model
    task_model = DigitFeatureEncoder().to(device)

    # if joint_training == False:

    # Load the pre-trained weights
    task_model.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist32by32_ae_random-samp_v9.pth', map_location=device))

    if joint_training == False:

        # Freeze the AE weights
        for param in task_model.parameters():
            param.requires_grad = False

        task_model.eval()

    # Initialize the U-Net-based feature extractor and load the pre-trained weights
    digit_encoder = DigitFeatureEncoder().to(device)
    digit_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist32by32_ae_random-samp_v9.pth', map_location=device))
    digit_encoder.eval()

    # Load the GT estimator - GSE learned sampling distribution
    gse_logits = torch.load(f'{CKPT_DIR}/gse_mnist32by32_ae_logits_v9.pth', map_location=device)
    gse_probs = torch.softmax(gse_logits, dim=0)

    # Initialize the flow model
    flow_model = CondFlow(encoder=digit_encoder, norm_type='bn').to(device)

    # Print the number of trainable parameters in the flow model
    num_params_fm = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the flow model: {num_params_fm}")

    if joint_training == True:
        # Print the number of trainable parameters in the task model
        num_params_task = sum(p.numel() for p in task_model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the task model: {num_params_task}")

    # Initialize the optimizer
    if joint_training == True:
        optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(task_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)

    else:
        optimizer = torch.optim.Adam(list(flow_model.parameters()) + [log_sig_1, log_sig_2], lr=1e-4)


    # Prevent the system from going to sleep

    try:

        # Training loop
        for step in range(n_steps):

            # Draw a random batch of images from the training and validation sets
            batch_indices_train = np.random.choice(train_set.shape[0], size=batch_size, replace=False)
            x_batch_train = train_set[batch_indices_train]

            # Add noise to the images
            x_batch_train_noisy = x_batch_train + torch.randn_like(x_batch_train) * 0.02

            # Pick a random integer between 10 and 500 of the pixels to keep (same for all images in the batch)
            n_sensors = torch.randint(10, 500, (1,), device=device).item()

            # Create a random sampling mask for conditioning the flow model
            mask_random = torch.zeros((batch_size, 1, image_size, image_size), device=device)
            sampling_indices_random = torch.randperm(n_pixels, device=device)[:n_sensors].long()
            mask_random.view(batch_size, -1)[:, sampling_indices_random] = 1.0

            # Create a conditioning batch
            x_batch_train_conditioning = x_batch_train_noisy * mask_random

            # Sample the GT mask estimation from the GSE learned distribution
            mask_gse = torch.zeros((batch_size, 1, image_size, image_size), device=device)

            for i in range(batch_size):
                sampling_indices_gse = torch.multinomial(gse_probs, n_sensors, replacement=False).long()
                mask_gse.view(batch_size, -1)[i, sampling_indices_gse] = 1.0

            # Sample noise x0 and time t
            x0 = torch.randn(batch_size, 1, image_size, image_size).to(device)  # Reshape to match image dimensions
            t = torch.rand(batch_size, 1).to(device)

            # Linear coupling between x0 and x1 (GSE mask) to create a continuous path for the flow model to learn
            xt = x0 * (1 - t.view(batch_size, 1, 1, 1)) + mask_gse * t.view(batch_size, 1, 1, 1)

            # GT velocity for the flow model (derivative of the linear coupling)
            gt_velocity = mask_gse - x0

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
            if (step+1) % (n_steps//20) == 0:
                print(f"[{step+1}] Total Loss: {total_loss.item():.6e}, FM Loss: {fm_loss.item():.6e}, Task Loss: {task_loss.item():.6e}, Sig_1: {sig_1_values[-1]:.4f}, Sig_2: {sig_2_values[-1]:.4f}")

                # Save the model checkpoint
                torch.save({
                    'flow_model_state_dict': flow_model.state_dict(),
                    'task_model_state_dict': task_model.state_dict(),
                    'log_sig_1': log_sig_1,
                    'log_sig_2': log_sig_2
                }, f'{CKPT_DIR}/flow_mnist32by32_ae_checkpoint_step_{step+1}.pth')

    finally:
        # Allow sleep again
        pass


    # Plot the training and validation losses evolution
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
    plt.plot(sig_1_values, label='Sigma 1 (FM)', alpha=0.7)
    plt.plot(sig_2_values, label='Sigma 2 (Task)', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('Learned Sigma Values')
    plt.title('Learned Uncertainty Parameters During Training')
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
        }, f'{CKPT_DIR}/joint_flow_mnist32by32_ae_v9.pth')
        print("The flow model jointly trained with the classifier saved")

    else:
        torch.save({
            'flow_model_state_dict': flow_model.state_dict(),
            'log_sig_1': log_sig_1,
            'log_sig_2': log_sig_2
        }, f'{CKPT_DIR}/flow_mnist32by32_ae_v9.pth')
        print("The flow model saved")


    # ======================================================================
    # Evaluate the performace of the U-Net AE + flow generated mask on the test MNIST set
    # ======================================================================

    # joint_training = False
    steepness = 20
    power = 2
    batch_size = 64
    time_points = torch.linspace(0, 1, 20).to(device) # ODE integration time points for the flow model inference
    # rates = [10, 25, 50, 100, 250, 500]
    rates = [100]


    # Initialize the classifier and load the pre-trained weights
    task_model = DigitFeatureEncoder().to(device)

    if joint_training == False:
        task_model.load_state_dict(torch.load(f'{CKPT_DIR}/unet_mnist32by32_ae_random-samp_v9.pth', map_location=device))

    else:
        checkpoint = torch.load(f'{CKPT_DIR}/joint_flow_mnist32by32_ae_v9.pth', map_location=device)
        task_model.load_state_dict(checkpoint['task_model_state_dict'])

    # Initialize the U-Net-based feature extractor and load the pre-trained weights
    digit_encoder = DigitFeatureEncoder().to(device)
    # digit_encoder.load_state_dict(torch.load(f'{CKPT_DIR}/joint_flow_mnist32by32_ae_v9.pth', map_location=device))
    digit_encoder.eval()

    # Initialize the flow model and load the trained weights
    flow_model = CondFlow(encoder=digit_encoder, norm_type='bn').to(device)

    if joint_training == False:
        flow_checkpoint = torch.load(f'{CKPT_DIR}/flow_mnist32by32_ae_v9.pth', map_location=device)
        flow_model.load_state_dict(flow_checkpoint['flow_model_state_dict'])

    else:
        flow_checkpoint = torch.load(f'{CKPT_DIR}/joint_flow_mnist32by32_ae_v9.pth', map_location=device)
        flow_model.load_state_dict(flow_checkpoint['flow_model_state_dict'])


    # Set the models to evaluation mode
    task_model.eval()
    flow_model.eval()

    for rate in rates:

        test_mae_values = []
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
                    original_np = x_original[i, 0].detach().cpu().numpy()
                    recon_np = x_recon[i, 0].detach().cpu().numpy()
                    mae = np.mean(np.abs(original_np - recon_np))
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1)
                    test_mae_values.append(mae)
                    test_psnr_values.append(psnr)
                    test_ssim_values.append(ssim) 

            if rate == 100:

                # Plot 3 examples of original and reconstructed test images
                fig, axs = plt.subplots(3, 3, figsize=(12, 12))

                for i in range(3):
                    axs[0, i].imshow(x_original[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[0, i].set_title("Original")
                    axs[0, i].axis('off')

                    # Display the sampling mask
                    axs[1, i].imshow(flow_mask[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[1, i].set_title(f"Sampling Mask (GSE)\n({n_sensors} sensors)")
                    axs[1, i].axis('off')

                    # Calculate PSNR for reconstruction
                    original_np = x_original[i, 0].detach().cpu().numpy()
                    recon_np = x_recon[i, 0].detach().cpu().numpy()
                    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1)
                    ssim = structural_similarity(original_np, recon_np, data_range=1)

                    axs[2, i].imshow(x_recon[i, 0].detach().cpu().numpy(), cmap='gray')
                    axs[2, i].set_title(f"Reconstructed (100 px)\nPSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
                    axs[2, i].axis('off')

                plt.tight_layout()
                plt.show()

                # Plot histograms of MAE, PSNR, and SSIM values
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.hist(test_mae_values, bins=30, color='blue', alpha=0.7)
                plt.xlabel('MAE')
                plt.ylabel('Frequency')
                plt.title(f'MAE Distribution ({rate} sensors)')
                plt.grid(True)

                plt.subplot(1, 3, 2)
                plt.hist(test_psnr_values, bins=30, color='orange', alpha=0.7)
                plt.xlabel('PSNR (dB)')
                plt.ylabel('Frequency')
                plt.title(f'PSNR Distribution ({rate} sensors)')
                plt.grid(True)

                plt.subplot(1, 3, 3)
                plt.hist(test_ssim_values, bins=30, color='red', alpha=0.7)
                plt.xlabel('SSIM')
                plt.ylabel('Frequency')
                plt.title(f'SSIM Distribution ({rate} sensors)')
                plt.grid(True)

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


            avg_mae = np.mean(test_mae_values)
            avg_psnr = np.mean(test_psnr_values)
            avg_ssim = np.mean(test_ssim_values)
            print(f"=== Performance at {rate} sensors ({np.round(rate/n_pixels*100, 2)}% of pixels for flow mask) ===")
            print(f"Average MAE on test set: {avg_mae:.6f}")
            print(f"Average PSNR on test set: {avg_psnr:.4f} dB")
            print(f"Average SSIM on test set: {avg_ssim:.4f}")
            print(" ")





def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce Experiment 1 (MNIST classification & reconstruction)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/mnist",
        help="Directory to save model checkpoints, learned logits, and CSV logs."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_experiment(args)
