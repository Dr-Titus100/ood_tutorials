# Load packages
import os
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.datasets import make_classification
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_num_threads(os.cpu_count())

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Wrap the NumPy arrays in a PyTorch data set for use with DataLoader.
class LargeDataset(Dataset): 
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define a massive deep neural network.
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(200, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            *[layer for _ in range(40) for layer in [
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]],
            nn.Linear(2048, 2)
        )

    def forward(self, x):
        return self.net(x)

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from '{filename}' at epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}', starting from scratch.")
        return 0

def train(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # Create 10 million synthetic dataset with 200 feature in total: 150 informative features and 30 redundant/correlated features. The remain. 
    X, y = make_classification(n_samples=10e6, n_features=200, n_informative=150,
                               n_redundant=30, n_classes=2, random_state=42)
    # Split the dataset into 90% training and 10% testing.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    train_dataset = LargeDataset(X_train, y_train)
    test_dataset = LargeDataset(X_test, y_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Load data in batches of 2048. num_workers=4 means parallel data loading with 4 processes (if supported).
    train_loader = DataLoader(train_dataset, batch_size=2048, sampler=train_sampler,
                              num_workers=8, pin_memory=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=2048, sampler=test_sampler,
                             num_workers=8, pin_memory=True, prefetch_factor=4)

    model = DeepNN().to(device)
    model = DDP(model, device_ids=[rank])

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Setup TensorBoard logging
    log_dir = f"runs/ddp_rank{rank}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    num_epochs = 50
    checkpoint_file = f"checkpoint_rank{rank}.pth"
    start_epoch = load_checkpoint(model, optimizer, checkpoint_file)

    # Time measurement
    start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        # TensorBoard logging step
        if rank == 0:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/val", accuracy, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}")

            # Save checkpoint
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, filename=checkpoint_file)

    if rank == 0:
        writer.close()
        end_time = time.time()
        print(f"Training completed in {(end_time - start_time) / 3600:.2f} hours")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)