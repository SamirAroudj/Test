
import argparse
import os
import torch
from data import get_mnist_loaders
from datetime import datetime
from interpolate import interpolate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Dict
from vae import VAE

SimpleConfig = Dict[str, Any]

def get_config() -> SimpleConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", "-e", type=int, required=False, default=50)
    parser.add_argument("--model", "-m", type=str, required=False, default="")

    args = parser.parse_args()
    cmd_args = vars(args)
    return cmd_args


def train(model, device: torch.device, data_loader, batch_size: int, optimizer, prev_updates: int = 0, writer: SummaryWriter = None) -> int:
    """
    Trains the model on the given data.
    
    Args:
        model (nn.Module): The model to train.
        data_loader (torch.utils.data.data_loader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    model.train()  # Set the model to training mode
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        n_upd = prev_updates + batch_idx
        
        data = data.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        
        output = model(data)  # Forward pass
        loss = output.loss
        
        loss.backward()
        
        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        
            print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)
            
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters
        
    return prev_updates + len(data_loader)


def test(model, latent_dim: int, device, data_loader, cur_step, writer: SummaryWriter = None):
    """
    Tests the model on the given data.
    
    Args:
        model (nn.Module): The model to test.
        data_loader (torch.utils.data.data_loader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Testing'):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data
            
            output = model(data, compute_loss=True)  # Forward pass
            
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
            
    test_loss /= len(data_loader)
    test_recon_loss /= len(data_loader)
    test_kl_loss /= len(data_loader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    
    if writer is not None:
        print("")
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)
        
        # Log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)
        
        # Log random samples from the latent space
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)


def run(output_dir: str, config: SimpleConfig, verbose: bool = False) -> None:
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_epochs = config["epochs"]
    latent_dim = 2
    hidden_dim = 512
    batch_size = 128

    # create run directory
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_dir, run_name)

    # use cuda?
    cuda_support = torch.cuda.is_available()
    if cuda_support:
        print("Using CUDA for training.")
    else:
        print("Running training on the CPU.")
    device = torch.device('cuda' if cuda_support else 'cpu')

    # model & optimizer
    print("Creating VAE model and optimizer.")
    model_file_name = config["model"]
    if len(model_file_name) > 0:
        model = torch.load(model_file_name, weights_only=False)
    else:
        model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # logger
    print("Creating logger.")
    log_dir = os.path.join(run_dir, "logging")
    writer = SummaryWriter(log_dir)

    # data sources
    print("Getting data loaders.")
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, verbose=verbose)

    print("Running training and testing loop.")
    prev_updates = 0
    for epoch in range(num_epochs):
        # debug log output
        m = f'Epoch {epoch+1}/{num_epochs}'
        print(m, flush=True)

        prev_updates = train(model, device, train_loader, batch_size, optimizer, prev_updates, writer=writer)
        test(model, latent_dim, device, test_loader, prev_updates, writer=writer)

        model_state = model.state_dict()
        file_name = os.path.join(run_dir, f"Model{epoch}.pt")
        torch.save(model_state, file_name)

    interpolate(model, device, run_dir=run_dir)

def main():
    verbose = True
    output_dir = "/home/samir/DevData/WorkingDirectory/Test/VAE/MNIST"
    config = get_config()
    run(output_dir=output_dir, config=config, verbose=verbose)

if __name__ == "__main__":
    main()