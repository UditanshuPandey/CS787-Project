# AdvGAN Attacks on MNIST & CIFAR-10

This project implements **AdvGAN** to generate adversarial examples in **semi-whitebox** and **black-box** settings on MNIST and CIFAR-10. The goal is to evaluate neural network robustness against GAN-based adversarial attacks.

---

## 🚀 Features
- AdvGAN implementation (Generator + PatchGAN Discriminator)
- Semi-whitebox attacks (gradient access only during training)
- Black-box attacks using **dynamic distillation**
- Supports MNIST and CIFAR-10 datasets
- Attacks on Models A/B/C and ResNet-32

---

## 📊 Results Summary
- **MNIST ASR:**  
  - Semi-whitebox: **88–94%**  
  - Black-box: **83–89%**
- **CIFAR-10 ASR:**  
  - Semi-whitebox: **84.6%**  
  - Black-box: **74.3%**

Key insight: **Higher accuracy ≠ higher robustness**. CIFAR-10 shows lower ASR due to dataset complexity.

---
## Project Structure
