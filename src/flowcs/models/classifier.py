"""MNIST classifier used as the frozen task model for the classification
sub-experiment of Experiment 1."""
from torch import nn
import torch.nn.functional as F


class DigitClassifier(nn.Module):
    """
    MNIST digit classifier.

    Designed to work with complex-valued Fourier coefficients that have been
    flattened and concatenated (real and imaginary parts), or with real-valued
    image-domain data. Based on the architecture from Huijben et al. (2020).

    Architecture:
    - Input: Flattened real + imaginary Fourier coefficients [B, 2*28*28]
      or image [B, 28*28]
    - FC1: input_dim -> input_dim (dimension preserving layer)
    - FC2: input_dim -> 256
    - FC3: 256 -> 128
    - FC4: 128 -> 128
    - FC5: 128 -> 10 (classification layer)
    - Output: Class probabilities [B, 10]
    """

    def __init__(self, input_dim=2 * 28 * 28, num_classes=10, dropout_rate=0.3):
        super(DigitClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, num_classes)

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        self.drop3 = nn.Dropout(dropout_rate)

        for m in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def freeze_parameters(self):
        """Freeze all parameters of the classifier (requires_grad=False)."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_parameters(self):
        """Unfreeze all parameters of the classifier (requires_grad=True)."""
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, input_dim] (or an image tensor that
               will be flattened) containing concatenated real/imaginary
               Fourier coefficients.
        Returns:
            Class probabilities of shape [B, num_classes]
        """
        x = x.squeeze(1).view(x.size(0), -1)

        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.drop3(x)

        x = self.fc4(x)
        x = self.leaky_relu(x)

        x = self.fc5(x)

        out = F.softmax(x, dim=1)
        return out
