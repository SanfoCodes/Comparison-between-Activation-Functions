
# 🔍 Comparing Activation Functions in Neural Networks

This repository contains the full codebase, analysis, visualizations, and theoretical evaluation for the project:

**"Mathematical Analysis and Theoretical Trade-offs of Activation Functions in Neural Networks: Convergence Behavior and Gradient Dynamics"**

Developed as part of the DASC 5420 course at TRU.

---

## 📚 Project Summary

This project evaluates and compares the performance of five common activation functions:

- **ReLU**
- **Sigmoid**
- **Tanh**
- **Leaky ReLU**
- **Swish**

We investigate their:
- Convergence speed
- Generalization (accuracy)
- Gradient dynamics (vanishing/exploding)
- Mathematical properties (smoothness, monotonicity, saturation)

---


---

## 🧪 How to Run (Google Colab)

1. Upload this repo or clone it to Google Colab.
2. Make sure CIFAR-10 can be downloaded (via `torchvision.datasets`).
3. Run `main.ipynb` to:
   - Train models with all 5 activation functions
   - Generate and save all loss/accuracy/gradient/derivative plots

---

## 🧠 Theory + Derivations

This repo integrates:
- Formal conditions for faster convergence (e.g., Lipschitz continuity)
- Optimization landscape insights (plateaus, curvature, gradient norms)
- Activation derivative analysis (impact on gradient flow)
- Key theorems from the **TeLU paper** on smooth approximations of ReLU

See `presentation/activation_analysis.md` for all derivations and theoretical write-up.

---

## 💾 Output Plots

All generated plots are saved automatically to:
/MyDrive/tml_project/plots/


You can customize this path in `main.ipynb`.

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Matplotlib
- NumPy
- Google Colab (recommended)

---

## 📬 Contact

Author : Arpitha Thippeswamy 
Team mates :
- Sree Aryan SP
- Zhang Jiyayi
- Shashank Manjunatha

---





