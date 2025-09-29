# Garment Color Correction

A PyTorch-based system for recoloring garments in on-model images using still product images as reference. The system uses deep learning to transfer color and texture properties while preserving garment shape and background.

## Overview

This project implements a three-model pipeline:

1. **Model 1 (Recoloring)**: A U-Net with cross-attention that transfers color from still images to on-model images
2. **Model 2 (Still Segmentation)**: Segments garments in still product images
3. **Model 3 (On-Model Segmentation)**: Segments garments in on-model images

## Key Features

- **Cross-attention mechanism** with 2D sinusoidal positional encoding
- **Color-accurate loss functions** including Delta E 1976 and perceptual losses
- **Light degradation simulation** for robust training (HSV, LAB, RGB modes)
- **Comprehensive evaluation metrics** including CIEDE2000, SSIM, and PSNR
- **Production-ready CLI tools** for training and inference
- **Extensive test coverage** for all components
- **Advanced data augmentation** with Albumentations for segmentation models
- **Dual-input segmentation** with cross-attention for improved on-model segmentation
- **Progressive learning rate scheduling** and early stopping
- **High-resolution output** (1024x1024) with intelligent upsampling
- **Hybrid checkpoint saving** based on multiple metrics
- **Training visualization** with automatic plot generation

## Installation

### Requirements

- Python 3.10+
- PyTorch ≥ 2.2
- CUDA-compatible GPU (recommended)
- Additional dependencies: numpy, opencv-python, scikit-image, einops, pyyaml, rich, tqdm, pytest, seaborn, scipy, Pillow, albumentations

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd garment-color-correction

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Dataset Structure

The expected dataset structure is:

```
dataset/
├── train/
│   ├── item_001/
│   │   ├── still.jpg
│   │   ├── mask_still.png
│   │   ├── on_model.jpg
│   │   └── mask_on_model.png
│   └── item_002/
│       └── ...
├── val/
│   └── ... (same structure as train)
└── test/
    ├── item_100/
    │   ├── still.jpg
    │   └── on_model.jpg  # No masks provided
    └── ...
```

### Data Conventions

- **Images**: RGB JPEG format, will be resized to specified dimensions
- **Masks**: Grayscale PNG format where 255 = garment, 0 = background
- **Training data**: Includes both images and ground-truth masks
- **Test data**: Only images provided; masks are predicted by Models 2 and 3

## Usage

### Training

Train the three models in sequence:

```bash
# 1. Train segmentation for still images (Model 2)
python -m src.cli.train_model2 --config configs/model2.yaml

# 2. Train segmentation for on-model images (Model 3)
python -m src.cli.train_model3 --config configs/model3.yaml

# 3. Train recoloring model (Model 1)
python -m src.cli.train_model1 --config configs/model1.yaml
```

### Inference

The inference pipeline consists of three steps:

```bash
# Step 1: Generate masks for test images (output: 1024x1024)
python -m src.cli.infer_masks \
    --data_root dataset/test \
    --ckpt_still runs/model2_seg_still/train_20250927_070746/best.pth \
    --ckpt_onmodel runs/model3_seg_onmodel/train_20250929_080636/best.pth \
    --overwrite

# Step 2: Create degraded on-model images (output: 1024x1024)
python -m src.cli.degrade_on_model \
    --data_root dataset/test \
    --mode hsl --magnitude 0.20

# Step 3: Generate corrected images (output: 1024x1024)
python -m src.cli.infer_recolor \
    --data_root dataset/test \
    --ckpt runs/model1_recolor/train_20250925_234945/best.pth \
    --config configs/model1.yaml \
    --use_degraded --overwrite

# Step 4: Visualize results (optional)
python -m src.cli.visualize_results \
    --data_root dataset/test \
    --output_dir results/visualizations
```

## Configuration

### Model 1 (Recoloring) - `configs/model1.yaml`

```yaml
data_root: dataset
img_size: 512
train:
  batch_size: 4
  epochs: 60
  lr: 0.00005
  weight_decay: 0.0001
  amp: true
  num_workers: 2
  seed: 42
model:
  base_channels: 96
  num_attn_blocks: 3
  num_heads: 8
  use_gan: false
loss_weights:
  w_l1: 1.0
  w_de: 0.5
  w_perc: 0.01
  w_gan: 0.0
degrade:
  enable: true
  mode: mixed
  magnitude: 0.06
val:
  every_n_epochs: 2
scheduler:
  type: "cosine_annealing"
  T_max: 60
  eta_min: 0.00001
early_stopping:
  enabled: true
  patience: 15
  min_delta: 0.001
  mode: "min"
  restore_best_weights: true
save_dir: runs/model1_recolor
```

### Models 2 & 3 (Segmentation) - `configs/model2.yaml`, `configs/model3.yaml`

```yaml
data_root: dataset
img_size: 512
train:
  batch_size: 16
  epochs: 30
  lr: 0.0001
  weight_decay: 0.0001
  amp: true
  num_workers: 4
  seed: 42
model:
  type: "enhanced"  # or "dual_input" for Model 3
  base_channels: 96
  use_attention: true
  dropout: 0.2
seg:
  threshold: 0.4
augment:
  hflip: true
  rotate_deg: 15
  scale: [0.8, 1.2]
  brightness: [0.8, 1.2]
  contrast: [0.8, 1.2]
  elastic_transform: true
  grid_distortion: true
  perspective_transform: true
  hue_shift: true
  saturation_shift: true
  gaussian_noise: true
  cutout: true
  mixup: true
  random_erasing: true
val:
  every_n_epochs: 2
scheduler:
  type: "cosine_annealing"
  T_max: 30
  eta_min: 0.00001
early_stopping:
  enabled: true
  patience: 8
  min_delta: 0.001
  mode: "max"
  restore_best_weights: true
save_dir: runs/model2_seg_still  # or runs/model3_seg_onmodel
```

## Architecture Details

### Model 1: Recoloring U-Net

- **Dual encoders** for on-model and still images
- **Cross-attention bottleneck** where query comes from on-model features and key/value from still features
- **Single decoder** with skip connections from on-model encoder
- **Mask-aware compositing** to preserve background regions

### Models 2 & 3: Segmentation U-Net

- **Enhanced U-Net** with spatial attention in bottleneck and skip connections
- **Dual-input architecture** (Model 3) with cross-attention for improved on-model segmentation
- **Advanced loss functions** including BCE, Dice, Focal, Tversky, and Boundary losses
- **Comprehensive data augmentation** with Albumentations (elastic transforms, perspective, noise, etc.)
- **Multi-task learning** for dual-input models (main + auxiliary segmentation)
- **Post-processing** with configurable thresholding

## Loss Functions

### Recoloring Loss (Model 1)

```
L = w_l1 * L1_masked + w_de * ΔE76_masked + w_perc * Perceptual_luma + w_gan * GAN_loss
```

- **Masked L1**: Pixel-wise L1 loss in garment regions only
- **Delta E 1976**: Perceptually-motivated color difference in LAB space
- **Perceptual loss**: VGG19-based feature matching on luminance channel
- **Optional GAN loss**: Adversarial loss for enhanced realism

### Segmentation Loss (Models 2 & 3)

**Basic Models:**
```
L = BCE(logits, target) + Dice(sigmoid(logits), target)
```

**Enhanced Models:**
```
L = w_bce * BCE + w_dice * Dice + w_focal * Focal + w_tversky * Tversky + w_boundary * Boundary
```

**Dual-Input Models (Model 3):**
```
L = w_main * L_main + w_aux * L_aux
```
Where each component loss includes all advanced loss functions.

## Evaluation Metrics

### Color Accuracy
- **Delta E 1976**: CIE color difference (lower is better)
- **CIEDE2000**: Advanced perceptual color difference
- **Statistics**: Mean, median, p90, p95, p99 percentiles

### Image Quality
- **PSNR**: Peak signal-to-noise ratio on luminance
- **SSIM**: Structural similarity index (planned)

### Segmentation Quality
- **IoU**: Intersection over Union
- **Dice coefficient**: Harmonic mean of precision and recall
- **Precision/Recall**: Standard classification metrics

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_color.py -v          # Color conversion tests
pytest tests/test_degradation.py -v    # Degradation function tests
pytest tests/test_model1_shapes.py -v  # Model 1 architecture tests
pytest tests/test_seg_threshold.py -v  # Segmentation tests
```

### Code Quality

The project follows Python best practices:

- **Type hints** throughout the codebase
- **Comprehensive docstrings** for all functions and classes
- **Linting** with flake8 and isort
- **Line length**: 100 characters maximum

## Technical Details

### Light Degradation

The system applies controlled degradation to on-model images during training to improve robustness:

- **HSV mode**: ±2° hue, ±3% saturation/value
- **LAB mode**: ±2 L*, ±1.5 a*/b*
- **RGB mode**: ±4/255 per channel
- **HSL mode**: ±5° hue, ±80% saturation, ±60% lightness (enhanced)
- **Mixed mode**: Combines multiple degradation types for maximum robustness

### Cross-Attention with Positional Encoding

The recoloring model uses 2D sinusoidal positional encoding to help the attention mechanism understand spatial relationships:

```python
# 2D sinusoidal encoding
pos_h = sin/cos(pos_y / 10000^(2i/d))
pos_w = sin/cos(pos_x / 10000^(2i/d))
pos_embed = concat([pos_h, pos_w])
```

### Memory and Performance

- **Mixed precision training** (AMP) supported for faster training
- **Gradient checkpointing** available for large models
- **Efficient data loading** with configurable workers
- **GPU memory optimization** through careful tensor management
- **Progressive learning rate scheduling** with cosine annealing, step, and exponential decay
- **Early stopping** to prevent overfitting and save training time
- **Hybrid checkpoint saving** based on multiple metrics (Delta E, total loss)
- **Training visualization** with automatic plot generation and timestamped directories
- **High-resolution inference** with intelligent upsampling from 512x512 to 1024x1024

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in config files
   - Enable gradient checkpointing
   - Use smaller model architectures

2. **Color conversion errors**
   - Ensure input images are in [0, 1] range
   - Check for NaN values in LAB conversion
   - Verify mask values are binary {0, 1}

3. **Training instability**
   - Reduce learning rate
   - Increase gradient clipping
   - Check loss weight balance

4. **Model parameter mismatch in inference**
   - Use `--config` flag to specify model configuration
   - Check that checkpoint matches model architecture
   - Verify model type (basic/enhanced/dual_input)

5. **Low segmentation IoU**
   - Increase data augmentation intensity
   - Adjust loss weights (increase boundary weight)
   - Use dual-input architecture for Model 3
   - Increase training epochs with early stopping

6. **HSL degradation not visible**
   - Increase magnitude (default: 0.20)
   - Check mask coverage percentage
   - Verify degradation is applied to correct regions

### Performance Tips

- Use **SSD storage** for datasets to improve I/O
- Set `num_workers` based on CPU cores (typically 4-8)
- Enable `pin_memory=True` for faster GPU transfers
- Use **AMP** for ~30% speedup on modern GPUs
- **Enable early stopping** to save training time
- **Use progressive learning rate scheduling** for better convergence
- **Monitor training plots** to identify overfitting early
- **Use hybrid checkpoint saving** for better model selection
- **Enable data augmentation** for improved generalization
- **Use dual-input architecture** for challenging segmentation tasks

## Citation

If you use this code in your research, please cite:

```bibtex
@software{garment_color_correction,
  title={Garment Color Correction with Cross-Attention},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/garment-color-correction}
}
```

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- scikit-image for color space conversion utilities
- Rich library for beautiful CLI interfaces
- Albumentations for advanced data augmentation
- The computer vision community for foundational research in image-to-image translation
- U-Net architecture for semantic segmentation
- Attention mechanisms for cross-modal learning
