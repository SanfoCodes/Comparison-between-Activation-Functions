import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    model.to(device)
    train_loss, val_loss, val_accuracy = [], [], []
    gradient_flow = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            gradient_flow.append(np.sqrt(total_norm))

            optimizer.step()
            running_loss += loss.item()

        train_loss.append(running_loss / len(train_loader))

        model.eval()
        val_running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_batch = criterion(outputs, labels)
                val_running_loss += val_loss_batch.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss.append(val_running_loss / len(val_loader))
        val_accuracy.append(100 * correct / total)

    return train_loss, val_loss, val_accuracy, gradient_flow
