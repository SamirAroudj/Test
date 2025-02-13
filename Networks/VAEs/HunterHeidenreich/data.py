from typing import Tuple
import os
import torch
from torchvision import datasets
from torchvision.transforms import v2


batch_size = 128

def get_mnist_loaders(
    batch_size: int,
    dataset_dir: str ="~/DevData/Test/Datasets/",
    verbose: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    mnist_dir = os.path.expanduser(dataset_dir)
    if verbose:
        m = f"Saving MNIST dataset in:\n{mnist_dir}"
        print(m, flush=True)


    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.view(-1) - 0.5),
    ])

    # Download and load the training data
    train_data = datasets.MNIST(
        mnist_dir, 
        download=True, 
        train=True, 
        transform=transform,
    )
    # Download and load the test data
    test_data = datasets.MNIST(
        mnist_dir, 
        download=True, 
        train=False, 
        transform=transform,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False,
    )

    return train_loader, test_loader