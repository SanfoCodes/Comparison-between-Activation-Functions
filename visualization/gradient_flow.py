import matplotlib.pyplot as plt

def plot_gradient_flow(gradient_norms, activation_name):
    plt.figure(figsize=(8, 5))
    plt.plot(gradient_norms)
    plt.title(f"Gradient Norm Flow - {activation_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.show()
