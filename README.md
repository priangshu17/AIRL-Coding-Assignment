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

# Text-Driven Segmentation with SAM 2 (Q2)

A complete end-to-end pipeline for text-based image segmentation that combines CLIPSeg for text-to-region detection with SAM 2 for precise mask generation.

## Pipeline Overview
The segmentation pipeline works in three stages:

Text Prompt → CLIPSeg (Heatmap Generation) → Seed Conversion → SAM 2 (Mask Refinement) → Final Mask

## Stage 1: Text-to-Heatmap (CLIPSeg)

Takes a natural language text prompt (e.g., "dog", "red car", "person wearing hat")
Generates a probability heatmap indicating where the described object likely appears
Uses CLIP-based vision-language understanding to match text semantics with image regions

## Stage 2: Seed Generation

Converts the CLIPSeg heatmap into prompts for SAM 2
Two modes available:

Box mode (default): Extracts bounding box from thresholded heatmap
Point mode: Identifies region centroids as point prompts


Threshold parameter controls sensitivity (lower = more permissive, higher = more precise)

## Stage 3: Mask Refinement (SAM 2)

Uses the generated seeds (box or points) as input prompts
Produces high-quality, pixel-precise segmentation masks
Leverages SAM 2's powerful boundary detection and object understanding

## Installation
The pipeline requires minimal setup and runs entirely in Google Colab:

### Install dependencies
!pip install git+https://github.com/facebookresearch/segment-anything-2.git
!pip install transformers torch torchvision opencv-python matplotlib pillow scikit-image

### Download SAM 2 checkpoint
!wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

## Usage 
# Initialize pipeline
pipeline = TextPromptedSAM2_CLIPSeg()

### Segment an object
masks, heatmap, image = pipeline.segment(
    image_path="photo.jpg",
    text_prompt="dog",
    threshold=0.4,
    use_box=True
)

### Visualize results
pipeline.visualize(image, masks, heatmap, "dog")

## Limitations
1. CLIPSeg Heatmap Quality

CLIPSeg generates coarse, low-resolution heatmaps (not pixel-perfect)
Works best for objects with clear visual distinction from background
May struggle with small objects, occlusions, or ambiguous regions
Performance depends heavily on how well the text prompt matches visual features

2. Text Prompt Specificity

Requires descriptive, specific prompts for best results
Generic terms like "object" or "thing" produce poor heatmaps
Works better with concrete nouns: "dog", "car", "tree" vs. abstract concepts
May misinterpret ambiguous prompts (e.g., "bank" could mean riverbank or financial institution)

3. Domain-Specific Limitations

Medical imaging: not trained on medical data, may miss pathologies
Aerial/satellite imagery: struggles with top-down perspectives
Abstract art: limited understanding of non-photorealistic content
Text in images: cannot segment based on written text content

## Files

- `q1.ipynb` → Implementation of Vision Transformer on CIFAR-10  
- `q2.ipynb` → Text-Driven Segmentation with SAM 2  
- `README.md` → Instructions, configs, and results


