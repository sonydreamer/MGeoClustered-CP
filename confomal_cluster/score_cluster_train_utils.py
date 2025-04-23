import torch
import numpy as np
from torch import nn
from tqdm import tqdm


def calculate_class_weights(dataloader, num_classes, c=1.02):
    """
    Computes class weights for a classification problem.

    Keyword arguments:
    - dataloader (``data.DataLoader``): A data loader to iterate over the dataset.
    - num_classes (``int``): The number of classes.
    - c (``float``, optional): A hyper-parameter that controls the range of weights. Default: 1.02.

    Returns:
    - class_weights (``np.array``): An array of weights for each class.
    """
    class_count = np.zeros(num_classes)
    total_samples = 0

    # Iterate over the dataset
    for _, labels, _ in tqdm(dataloader):
        labels = labels.cpu().numpy()
        total_samples += labels.shape[0]

        # Count the number of samples for each class
        class_count += np.bincount(labels, minlength=num_classes)

    # Compute the propensity score and then the weights for each class
    propensity_score = class_count / total_samples
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights

class WeightedMSE(nn.Module):
    def __init__(self, weight=None):
        super(WeightedMSE, self).__init__()
        if weight is not None:
            self.weight = weight / weight.sum()
        else:
            self.weight = None

    def forward(self, inputs, targets):
        # Compute the MSE loss
        mse = (inputs - targets) ** 2
        
        if self.weight is not None:
            mse = mse.mean(dim=0)
            mse = (mse * self.weight).sum()
        else:
            mse = mse.mean()
        return mse


class GeodesicDistanceLoss(nn.Module):
    def __init__(self, model, criterion, step=10, num_copies=16, noise_std=0.1, inv_lr=0.01):
        """
        Geodesic Distance Loss, which measures the average distance during the optimization process in latent space.
        Args:
            model: The neural network model to be used for evaluation.
            criterion: The criterion to use for calculating prediction loss.
            step: Number of optimization steps.
            num_copies: Number of noisy copies to generate for geodesic distance calculation.
            noise_std: Standard deviation of noise to add to the latent representation.
            inv_lr: Learning rate for optimizing the latent vectors.
        """
        super(GeodesicDistanceLoss, self).__init__()
        self.model = model
        self.Encoder = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten(), *list(self.model.fc.children())[:-1])
        self.Decoder = list(model.fc.children())[-1]
        self.criterion = criterion  # Loss function for predictions (e.g., CrossEntropyLoss)
        self.step = step
        self.num_copies = num_copies
        self.noise_std = noise_std
        self.inv_lr = inv_lr

    def forward(self, x, label):
        """
        Forward pass to calculate Geodesic Distance Loss.
        Args:
            x: Input data sample.
            label: True label for the input data sample.
            layer: The encoder layer to obtain latent representation.
        Returns:
            Geodesic distance as a loss value.
        """
        # Prepare input data and obtain initial latent representation
        self.model.eval() 
        x = torch.unsqueeze(x, dim=0)  # Add batch dimension
        z0 = self.Encoder(x)  # Get latent representation
        z = z0.detach().clone().squeeze()  # Detach to avoid affecting original gradients
        
        # Generate noisy versions of latent representation
        z_argument_batch = [z]
        for i in range(self.num_copies - 1):
            noise = torch.randn_like(z) * self.noise_std
            z_argument_batch.append(z + noise)
        z_argument_batch = torch.stack(z_argument_batch)
        z_argument_batch.requires_grad_()  # Enable gradient computation

        # Initialize optimizer for the latent vectors
        optimizer = torch.optim.SGD([z_argument_batch], lr=self.inv_lr)
        # optimizer = torch.optim.Adam([z_argument_batch], lr=self.inv_lr)

        # self.model.eval()  # Set model to evaluation mode
        each_step_distance = []
        for _ in range(self.step):
            # Deep copy of z_argument_batch before update
            z_clone = z_argument_batch.detach().clone()

            # Forward pass through the model's decoder or relevant function
            pred = self.Decoder(z_argument_batch)
            loss = self.criterion(pred.squeeze(), label)  # Compute prediction loss

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate L2 distance between the updated and original latent representations
            distance = torch.mean(torch.abs(z_clone - z_argument_batch))  # L1范数
            # distance = torch.mean(torch.norm(z_clone - z_argument_batch, p=2, dim=1))  # L2 norm across copies
            each_step_distance.append(distance)

        # Calculate the average geodesic distance across all steps
        geodesic_distance = torch.mean(torch.stack(each_step_distance))

        return geodesic_distance

# Example usage within a training loop
# Assuming `model` is your neural network model and `criterion` is the primary loss function (e.g., CrossEntropyLoss)
# geodesic_loss_fn = GeodesicDistanceLoss(model=model, criterion=nn.CrossEntropyLoss())
# 
# for data, target in dataloader:
#     optimizer.zero_grad()
#     output = model(data)
#     primary_loss = criterion(output, target)
#     geodesic_loss = geodesic_loss_fn(data, target, model.model.encoder)
#     total_loss = primary_loss + geodesic_loss
#     total_loss.backward()
#     optimizer.step()


if __name__ == "__main__":
    import sys
    weights = torch.tensor([0.2, 0.3, 0.5])
    inputs = torch.tensor([[0.1, 0.3, 0.1], [0.2, 0.1, 0.5]] )
    targets = torch.tensor([[0, 1, 0], [0, 0, 1]])
    labeles = torch.tensor([1, 2])
    certaion = WeightedMSE(weights)
    certaion(inputs, targets)
    