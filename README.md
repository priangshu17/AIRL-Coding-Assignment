# Vision Transformer on CIFAR-10 (Q1)

This repository contains the solution for **Q1** of the AIRL Internship coding assignment.  
The task was to implement a Vision Transformer (ViT) from scratch in PyTorch and train it on CIFAR-10.

---

## How to Run in Colab
1. Open `q1.ipynb` in Google Colab.  
2. Go to `Runtime → Run all`.  
3. No additional `pip install` is required; the notebook runs end-to-end with default Colab dependencies.  

---

## Best Model Configuration

| Parameter        | Value   |
|------------------|---------|
| Image Size       | 32      |
| Patch Size       | 4       |
| In Channels      | 3       |
| Num Classes      | 10      |
| Embed Dim        | 192     |
| Depth (Layers)   | 8       |
| Num Heads        | 12      |
| MLP Dim          | 256     |
| Batch Size       | 128     |
| Epochs           | 50      |
| Base LR          | 3e-4    |
| Weight Decay     | 0.01    |
| Warmup Epochs    | 5       |
| Optimizer        | AdamW   |
| LR Scheduler     | Cosine decay with warmup (LambdaLR) |

---

## Results

| Epoch | Test Accuracy (%) |
|-------|--------------------|
| 49    | **82.04** |

---

## Bonus: Analysis

- **Augmentation**: Adding random crop + horizontal flip improved generalization by ~3% (72% → 75%).  
- **Optimizer**: Switching from Adam to AdamW with weight decay boosted accuracy further to ~77%.  
- **Learning Rate Schedule**: Warmup + cosine decay helped stabilize training and pushed accuracy above 80%.  
- **Model Capacity**: Increasing embed dim from 128 → 192 and depth from 6 → 8 improved performance to ~82%.   

---

## Files

- `q1.ipynb` → Implementation of Vision Transformer on CIFAR-10  
- `q2.ipynb` → Text-Driven Segmentation with SAM 2  
- `README.md` → Instructions, configs, and results
