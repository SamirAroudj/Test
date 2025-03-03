import os
import torch
import matplotlib.pyplot as plt


def interpolate(model, device, run_dir, n: int = 16) -> None:
    print("Interpolating example numbers.")

    z1 = torch.linspace(-0, 1, n)
    z2 = torch.zeros_like(z1) + 2
    z = torch.stack([z1, z2], dim=-1).to(device)
    samples = model.decode(z)
    samples = torch.sigmoid(samples)

    # Plot the generated images
    fig, ax = plt.subplots(1, n, figsize=(n, 1))
    for i in range(n):
        ax[i].imshow(samples[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
        ax[i].axis('off')
        
    local_name = 'vae_mnist_interp.webp'
    file_name = os.path.join(run_dir, local_name)
    plt.savefig(file_name)