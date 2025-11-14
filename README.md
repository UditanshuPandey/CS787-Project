# AdvGAN Attacks on MNIST & CIFAR-10

This project implements **AdvGAN** to generate adversarial examples in **semi-whitebox** and **black-box** settings on MNIST and CIFAR-10. The goal is to evaluate neural network robustness against GAN-based adversarial attacks.

---

## Features
- AdvGAN implementation (Generator + PatchGAN Discriminator)
- Semi-whitebox attacks (gradient access only during training)
- Black-box attacks using **dynamic distillation**
- Supports MNIST and CIFAR-10 datasets
- Attacks on Models A/B/C and ResNet-32

---

## Results Summary
- **MNIST ASR:**  
  - Semi-whitebox: **88–94%**  
  - Black-box: **83–89%**
- **CIFAR-10 ASR:**  
  - Semi-whitebox: **84.6%**  
  - Black-box: **74.3%**

Key insight: **Higher accuracy ≠ higher robustness**. CIFAR-10 shows lower ASR due to dataset complexity.

---
## Summary of Results

- **MNIST Semi-Whitebox:** 88–94% ASR  
- **MNIST Black-Box:** 83–89% ASR  
- **CIFAR-10 Semi-Whitebox:** 84.6% ASR  
- **CIFAR-10 Black-Box:** 74.3% ASR  
- Dynamic distillation achieves **<2% gap** between substitute and black-box ASR.

---
## Project Structure
1_MNIST_Semi_Whitebox_Attack/
├── checkpoints/
├── advgan_semi_whitebox.py
└── readme.md

2_MNIST_BlackBox_Attack/
├── checkpoints/
├── blackbox_dynamic_distillation.py
└── readme.md

3_CIFAR_Semi_Whitebox_Attack/
├── checkpoints/
├── advgan_semi_whitebox_cifar10.py
└── readme.md

4_CIFAR_BlackBox_Attack/
├── checkpoints/
├── CIFAR_Dynamic_Distillation.py
└── readme.md

Plots/
├── plot1_mnist_asr.png
├── plot2_cifar_asr.png
├── plot3_accuracy_vs_asr.png
├── plot4_sub_vs_blackbox.png
├── plot5_dataset_complexity.png
├── plot6_asr_consistency.png
└── readme.md
