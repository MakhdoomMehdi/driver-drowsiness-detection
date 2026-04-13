# Driver Drowsiness Detection using CNN + Attention

A deep learning pipeline that detects driver drowsiness from face images using a **ResNet-50** backbone enhanced with **CBAM (Convolutional Block Attention Module)** for channel and spatial attention.

## Architecture

```
Image Input
     │
Face Detection (MTCNN)
     │
Face Crop + Resize (224×224)
     │
CNN Backbone (ResNet-50, ImageNet pretrained)
     │
CBAM Attention (Channel + Spatial)    ← novelty component
     │
Global Average Pooling
     │
Fully Connected (2048 → 512 → 1)
     │
Sigmoid → Drowsy / Non-Drowsy
```

**Key design choices:**
- **MTCNN** isolates the face region before classification, removing background noise
- **CBAM** lets the model learn to focus on discriminative regions (eyes, mouth) rather than treating all spatial locations equally
- **Leakage-safe splitting** — train/val split is done by sequence prefix so near-duplicate frames from the same clip never appear in both sets
- **Caching at every stage** — intermediate results are saved to Google Drive so expensive steps (face detection, training) only run once

## Dataset

[Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) — ~41,000 PNG face images across two classes:

| Class | Count |
|-------|-------|
| Drowsy | ~22,300 |
| Non-Drowsy | ~19,400 |

## Results

| Metric | Value |
|--------|-------|
| Validation AUC-ROC | ~0.96–0.98 |
| Validation Accuracy | ~93% |
| Training Time | ~45–75 min on T4 GPU (early stops around epoch 8–10) |

The notebook produces a confusion matrix, ROC curve, classification report, and spatial attention heatmaps showing which face regions the model focuses on.

## Project Structure

```
├── demo.ipynb          # Full pipeline notebook (run on Google Colab)
├── requirements.txt    # Python dependencies
├── HOW_TO_RUN.md       # Detailed step-by-step guide with expected outputs
└── README.md
```

**On Google Drive (created at runtime):**

```
My Drive/datasets/
├── Driver Drowsiness Dataset (DDD)/    # Raw dataset
│   ├── Drowsy/
│   └── Non Drowsy/
├── processed_faces/                    # MTCNN face crops (cached)
├── best_model.pt                       # Best model checkpoint
├── drowsiness_best_model.pt           # Final saved model
└── pipeline_results/                   # Cached intermediate results
    ├── step3_records.pkl
    ├── step4_split.pkl
    ├── step5_faces.pkl
    ├── step9_history.pkl
    ├── step11_eval.pkl
    └── step12_attention.pkl
```

## How to Run

### Prerequisites

- A Google account with Google Drive
- Google Colab (free tier GPU is sufficient): [colab.research.google.com](https://colab.research.google.com)

### Setup

1. Download the [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) from Kaggle
2. Upload it to your Google Drive at:
   ```
   My Drive/datasets/Driver Drowsiness Dataset (DDD)/
       ├── Drowsy/          (PNG images)
       └── Non Drowsy/      (PNG images)
   ```
3. Upload `demo.ipynb` to Colab (or open it directly from Drive)
4. Set the runtime to **GPU**: Runtime → Change runtime type → GPU

### Running the Notebook

Run all cells top to bottom. The notebook is split into 14 steps:

| Step | What it does | First run | Subsequent runs |
|------|-------------|-----------|-----------------|
| 1 | Install dependencies, mount Drive | ~30s | ~30s |
| 2 | Configuration and imports | instant | instant |
| 3 | Scan dataset and build image index | ~10s | cached |
| 4 | Leakage-safe train/val split | ~1s | cached |
| 5 | MTCNN face detection and cropping | **30–60 min** | cached |
| 6 | Create PyTorch DataLoaders | ~5s | ~5s |
| 7 | Build ResNet-50 + CBAM model | ~5s | ~5s |
| 8 | Set up loss, optimizer, scheduler | instant | instant |
| 9 | Training loop with early stopping | **45–75 min** | cached |
| 10 | Plot training curves | instant | instant |
| 11 | Evaluation (confusion matrix, ROC) | ~1 min | cached |
| 12 | Attention map visualization | ~30s | cached |
| 13 | Single-image inference demo | ~2s | ~2s |
| 14 | Save final model to Drive | ~5s | ~5s |

Steps marked **cached** save their results to `pipeline_results/` on Drive. On re-runs they load instantly instead of recomputing. To force a fresh run of any step, delete the corresponding `.pkl` file from `My Drive/datasets/pipeline_results/`.

### Local Setup (Optional)

```bash
pip install -r requirements.txt
```

You will need to adjust `DATA_ROOT` and other Drive paths in Step 2 to point to your local dataset location.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 224×224 |
| Batch size | 32 |
| Epochs (max) | 15 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | Cosine annealing (eta_min=1e-6) |
| Early stopping patience | 5 epochs |
| Backbone | ResNet-50 (IMAGENET1K_V2) |
| Attention | CBAM (reduction=16, spatial_kernel=7) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Using device: cpu` | Runtime → Change runtime type → GPU |
| `WARNING: directory not found` | Check Drive path matches exactly: `My Drive/datasets/Driver Drowsiness Dataset (DDD)/Drowsy` and `Non Drowsy` |
| Colab disconnects during face caching | Re-run Step 5 — already processed images are skipped |
| Out of memory | Reduce `BATCH_SIZE` from 32 to 16 in Step 2 |
| Want to retrain from scratch | Delete `pipeline_results/` folder and `best_model.pt` from Drive |

## Dependencies

- PyTorch >= 2.1
- torchvision >= 0.16
- facenet-pytorch >= 2.5.3
- scikit-learn >= 1.3
- matplotlib >= 3.8
- Pillow >= 10.0
- tqdm >= 4.66
- numpy >= 1.24
