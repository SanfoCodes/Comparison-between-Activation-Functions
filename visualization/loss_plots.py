import matplotlib.pyplot as plt

def plot_losses(train_loss, val_loss, activation_name):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(f"Loss Curve - {activation_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(val_acc, activation_name):
    plt.figure(figsize=(8, 5))
    plt.plot(val_acc, label="Validation Accuracy", color='green')
    plt.title(f"Validation Accuracy - {activation_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.show()
