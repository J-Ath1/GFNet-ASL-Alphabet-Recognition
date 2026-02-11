# GFNet (PyTorch)

Image classification project using **GFNet (Global Filter Networks)** trained in **PyTorch**. Training was run on an **NVIDIA A100 (Colab)** and completed in ~**3 hours**.

## Highlights

- **Framework:** PyTorch  
- **Hardware:** A100 (Google Colab)  
- **Train time:** ~3 hours  
- **Accuracy:**
  - Reached **96% validation accuracy** initially
  - Fine-tuned to **100% validation accuracy** using cosine annealing with **lr = 1e-4** (10% of initial lr)
  - **100% test accuracy** on provided test set (**112 images**)
  - **97% test accuracy** on an additional evaluation set of **5600 images** (200 per class) sampled from unused training data

## Why GFNet?

GFNet replaces attention with **learned global filters**, enabling more computation within a fixed budget:

- Attention is **O(n²)**, while GFNet’s global filtering is **O(n log n)** (via FFT)
- Global filters capture spatial dependencies in “one shot”, allowing the model to use the *full image* effectively
- Learned frequency responses quickly emphasize discriminative (“dichotomizing”) frequencies during training

## Known Failure Modes / Errors

Most remaining class-specific errors appear driven by:

1. **Class overlap in embedding space:** UMAP embeddings (computed from PCA-compressed data) show significant overlap among:
   - **A / F**
   - **M / N**
   - **R / S / V**
2. **Potential class imbalance:** Random selection from a reduced training subset may have introduced imbalance.

## Suggested Improvements

To push test performance higher (especially for **A/F/M/N/R/S/V**):

- Further fine-tuning at a **lower learning rate**
- Increase/adjust **dropout**
- Continue **cosine annealing**
- Apply targeted sampling / class-balancing strategies with emphasis on those confusing classes

---

## Model Architecture

### Input / Tokenization

- **Input:** `96 × 96` grayscale
- **Patch embedding:** initial convolution with **kernel=8, stride=8**
- Produces:
  - **144 tokens** (12×12)
  - **token dimension:** 256

### Positional Embeddings

- Learned positional embeddings of shape **144 × 256**
- **Dropout:** 0.2

### Transformer Block (per block)

Each block contains:

1. **LayerNorm**
2. **Global Filter**
   - Applied over the **12×12 token map**
   - Uses **2D FFT**
   - Learnable complex-valued weights of shape **7 × 7 × 256**
3. **DropPath**
   - Linearly scheduled up to **0.1**
4. **LayerNorm**
5. **MLP**
   - 256 -> 1024 (Linear)
   - **GELU**
   - **Dropout:** 0.2
   - 1024 -> 256 (Linear)
   - **Dropout:** 0.2

### Classification Head

- **LayerNorm**
- Linear projection: **256 -> 28 classes**

---

## Training Setup

- **Loss:** Cross Entropy
- **Batch size:** 512
- **Optimizer:** AdamW
  - Initial learning rate: **1e-3**
  - Weight decay (base stage): **1e-6**
  - Gradient clipping enabled

### LR Schedule

- **Linear warmup:** first **5 epochs**, lr ramped from 0 -> 1e-3
- **Cosine annealing:** epochs **5 -> 20**, lr decayed from 1e-3 -> 0

### Fine-Tuning (“Polishing” Stage)

- Fine-tuned from ~96% -> **100% validation**
- **lr = 1e-4** (10% of initial lr)
- Weight decay increased:
  - **1e-6 -> 1e-5**

---

## Training Efficiency Notes (A100)

To speed up training on A100:

- Enabled **TF32**
  - (19-bit precision format with 10-bit mantissa)
  - Reported substantial speedups for relevant ops in PyTorch on Ampere GPUs
- Allowed **cuDNN TF32** for convolution kernels
- Set matrix multiply precision to **"medium"** (TF32 for most matmuls)
- Enabled **cuDNN benchmark**
  - Profiles multiple algorithms on first call
  - Caches fastest algorithm for subsequent runs

---

## Results Summary

| Evaluation Set | Size | Accuracy |
|---|---:|---:|
| Provided test set | 112 images | **100%** |
| Unused training sample | 5600 images | **97%** |

---

## Notes

This README captures the current experimental setup and observed results. If you add scripts/notebooks later, consider extending with:
- dataset preparation instructions
- exact command lines / config files
- checkpoints + reproducibility notes (seed, deterministic flags)
