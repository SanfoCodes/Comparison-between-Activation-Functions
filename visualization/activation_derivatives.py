import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def plot_activation_and_derivative(fn, dfn, name, x_range=(-5, 5)):
    x = torch.linspace(*x_range, steps=200)
    y = fn(x)
    dy = dfn(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x.numpy(), y.numpy(), label=f'{name}')
    plt.plot(x.numpy(), dy.numpy(), label=f"{name}'", linestyle='--')
    plt.title(f'{name} and its Derivative')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define activation functions and derivatives
def relu(x): return F.relu(x)
def d_relu(x): return (x > 0).float()

def sigmoid(x): return torch.sigmoid(x)
def d_sigmoid(x):
    sig = torch.sigmoid(x)
    return sig * (1 - sig)

def tanh(x): return torch.tanh(x)
def d_tanh(x): return 1 - torch.tanh(x) ** 2

def leaky_relu(x): return F.leaky_relu(x, negative_slope=0.01)
def d_leaky_relu(x):
    dx = torch.ones_like(x)
    dx[x < 0] = 0.01
    return dx

def swish(x): return x * torch.sigmoid(x)
def d_swish(x):
    sig = torch.sigmoid(x)
    return sig + x * sig * (1 - sig)
