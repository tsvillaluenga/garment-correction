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
- **Light degradation simulation** for robust training
- **Comprehensive evaluation metrics** including CIEDE2000, SSIM, and PSNR
- **Production-ready CLI tools** for training and inference
- **Extensive test coverage** for all components

## Installation

### Requirements

- Python 3.10+
- PyTorch ≥ 2.2
- CUDA-compatible GPU (recommended)

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
# Step 1: Generate masks for test images
python -m src.cli.infer_masks \
    --data_root dataset/test \
    --ckpt_still runs/model2_seg_still/best.pth \
    --ckpt_onmodel runs/model3_seg_onmodel/best.pth \
    --img_size 512 --thresh 0.5

# Step 2: Create degraded on-model images
python -m src.cli.degrade_on_model \
    --data_root dataset/test \
    --mode lab --magnitude 0.04 \
    --skip_if_exists

# Step 3: Generate corrected images
python -m src.cli.infer_recolor \
    --data_root dataset/test \
    --ckpt_model1 runs/model1_recolor/best.pth \
    --img_size 512 \
    --save_dir results/test \
    --use_degraded

# Step 4: Evaluate results
python -m src.cli.evaluate_corrected \
    --data_root dataset/test \
    --results_dir results/test \
    --save_csv results/metrics.csv
```

## Configuration

### Model 1 (Recoloring) - `configs/model1.yaml`

```yaml
data_root: dataset
img_size: 512
train:
  batch_size: 8
  epochs: 60
  lr: 1.0e-4
  weight_decay: 0.0
  amp: true
  num_workers: 8
  seed: 42
model:
  base_channels: 64
  num_attn_blocks: 2
  num_heads: 4
  use_gan: false
loss_weights:
  w_l1: 1.0
  w_de: 1.0
  w_perc: 0.1
  w_gan: 0.0
degrade:
  enable: true
  mode: lab
  magnitude: 0.04
val:
  every_n_epochs: 1
save_dir: runs/model1_recolor
```

### Models 2 & 3 (Segmentation) - `configs/model2.yaml`, `configs/model3.yaml`

```yaml
data_root: dataset
img_size: 512
train:
  batch_size: 8
  epochs: 50
  lr: 1.0e-4
  weight_decay: 0.0
  amp: true
  num_workers: 8
  seed: 42
seg:
  threshold: 0.5
augment:
  hflip: true
  rotate_deg: 5
  scale: [0.95, 1.05]
val:
  every_n_epochs: 1
save_dir: runs/model2_seg_still  # or runs/model3_seg_onmodel
```

## Architecture Details

### Model 1: Recoloring U-Net

- **Dual encoders** for on-model and still images
- **Cross-attention bottleneck** where query comes from on-model features and key/value from still features
- **Single decoder** with skip connections from on-model encoder
- **Mask-aware compositing** to preserve background regions

### Models 2 & 3: Segmentation U-Net

- **Lightweight U-Net** architecture optimized for binary segmentation
- **Combined BCE + Dice loss** for robust training
- **Data augmentation** including horizontal flips, rotation, and scaling
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

```
L = BCE(logits, target) + Dice(sigmoid(logits), target)
```

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

### Performance Tips

- Use **SSD storage** for datasets to improve I/O
- Set `num_workers` based on CPU cores (typically 4-8)
- Enable `pin_memory=True` for faster GPU transfers
- Use **AMP** for ~30% speedup on modern GPUs

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
- The computer vision community for foundational research in image-to-image translation
