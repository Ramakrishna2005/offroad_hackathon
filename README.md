# offroad_hackathon
Trained Dinov2 Vision Model for Desert Areas to detect the Objects - Offroad Hackathon
# 🚙 Off-Road Semantic Segmentation via DINOv2 & ConvNeXt

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

An autonomous vehicle perception pipeline designed to classify unpaved, off-road terrain in real-time. This project leverages a foundational Vision Transformer (**DINOv2**) as a feature extractor, paired with a custom **ConvNeXt-style** decoding head to perform dense, 10-class semantic segmentation.

## 🧠 Architecture & Optimizations

Building an off-road semantic segmentation model under strict hackathon time constraints required a highly optimized, high-throughput pipeline. 

* **Backbone:** Facebook's `dinov2_vits14` (frozen pre-trained weights for robust spatial geometry recognition).
* **Segmentation Head:** A lightweight, ConvNeXt-inspired convolutional decoder.
* **Loss Function:** A custom composite function combining **Cross-Entropy Loss** and **Multiclass Dice Loss** to aggressively penalize poor recall on rare minority classes (like logs and rocks).
* **High-Performance Training:**
  * **Dual-GPU DataParallel:** Batch workloads split dynamically across Kaggle's dual NVIDIA T4 GPUs.
  * **Automatic Mixed Precision (AMP):** 32-bit floating-point operations were downcast to 16-bit (FP16) using PyTorch's `autocast` and `GradScaler`, doubling training speed and slashing VRAM consumption.
  * **Single-Loop Validation:** Heavy Train-IoU metrics were dynamically bypassed to skip redundant 7-minute evaluation loops, relying purely on unseen validation data for model checkpointing.
* **Inference:** Employs **Test-Time Augmentation (TTA)** via horizontal tensor flipping to increase prediction confidence and spatial consistency.

## 🗺️ Object Classes
The model categorizes the off-road environment into 10 distinct navigational classes:
1. `Background`
2. `Trees`
3. `Lush Bushes`
4. `Dry Grass`
5. `Dry Bushes`
6. `Ground Clutter`
7. `Logs`
8. `Rocks`
9. `Landscape`
10. `Sky`

## ⚙️ Installation & Requirements

```bash
pip install torch torchvision numpy matplotlib opencv-python segmentation-models-pytorch tqdm pillow
