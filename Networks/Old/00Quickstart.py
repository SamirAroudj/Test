# This Python file uses the following encoding: utf-8

from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

FashionMNIST_classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

device = None

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def get_device() -> str:
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def get_data():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return (training_data, test_data)

def train_model(loader, model, loss_fn, optimizer):
    size = len(loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def output_data_shape(loader: DataLoader) -> None:
    for X, y in loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

def test_model(loader, model, loss_fn) -> None:
    size = len(loader.dataset)
    num_batches = len(loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def predict(data_point, model: NeuralNetwork, classes: List[str]) -> None:
    model.eval()
    x, gt_label = data_point[0], data_point[1]
    with torch.no_grad():
        predicted_label = model(x)
        predicted_label = predicted_label[0].argmax(0)
        predicted, actual = classes[predicted_label], classes[gt_label]
        temp = "Correctly" if predicted_label == gt_label else "Wrongly"
        print(f'{temp} predicted: "{predicted}", Actual: "{actual}"')

if __name__ == "__main__":
    batch_size = 64

    try:
        print("Trying to load model from data/model.pth.")
        model = NeuralNetwork()
        model.load_state_dict(torch.load("data/model.pth"))

    except:
        # build network model
        print("Building new model from scratch.")
        device = get_device()
        model = NeuralNetwork().to(device)
        print(model)

    # get data
    training_data, test_data = get_data()
    training_loader = DataLoader(training_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    output_data_shape(test_loader)

    # predict given current state
    test_count = 5
    random_indices = torch.randint(0, len(test_data), (test_count,))
    for random_idx in random_indices:
        data_point = test_data[random_idx]
        predict(data_point, model, FashionMNIST_classes)

    # loss type & SGD
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        print("-------------------------------")
        train_model(training_loader, model, loss_fn, optimizer)
        test_model(test_loader, model, loss_fn)
    print("Done networking.")

    torch.save(model.state_dict(), "data/model.pth")
    print("Saved PyTorch Model State to model.pth")
