# RailTrack-DaViT

A deep learning model for rail track defect detection using the DaViT (Data-efficient Vision Transformer) architecture.

## Overview

This project implements a binary classification model to detect defects in rail tracks using the DaViT vision transformer architecture. The model is trained on a dataset of rail track images labeled as either "Defective" or "Non Defective".

## Features

- Data-efficient Vision Transformer (DaViT) model for image classification
- Transfer learning with pre-trained weights
- Data augmentation techniques (random horizontal/vertical flips, rotation)
- OneCycleLR learning rate scheduler for efficient training
- Two-phase training strategy (frozen backbone + fine-tuning)
- Comprehensive evaluation metrics (accuracy, precision, recall, F1 score)
- Visualization tools for training progress and results

## Project Structure

```
RailTrack-DaViT/
│
├── main.py                  # Main entry point for training and evaluation
├── model.py                 # DaViT model implementation
├── dataset.py               # Dataset loading and preprocessing
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
└── utils/                   # Utility functions
    ├── __init__.py
    ├── trainer.py           # Training functions
    ├── evaluation.py        # Evaluation metrics and visualization
    ├── visualization.py     # Plotting functions
    └── saving.py            # Model and results saving utilities
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/phaphuang/RailTrack-DaViT.git
cd RailTrack-DaViT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model with default parameters:

```bash
python main.py
```

With custom parameters:

```bash
python main.py --train_dir path/to/train --test_dir path/to/test --image_size 224 --batch_size 16 --num_epochs 90 --fine_tune_epochs 10
```

### Dataset Structure

The dataset should be organized as follows:

```
train/
├── Defective/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Non Defective/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

test/
├── Defective/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Non Defective/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Results

The model achieves high accuracy in detecting rail track defects. Training results, including loss curves, accuracy metrics, and confusion matrices, are saved in the output directory.

## Citation

If you use this code for your research, please cite:

```
@article{phaphuangwittayakul2024railtrack,
  title={RailTrack-DaViT: A Vision Transformer-Based Approach for Automated Railway Track Defect Detection},
  author={Phaphuangwittayakul, Aniwat and Harnpornchai, Napat and Ying, Fangli and Zhang, Jinming},
  journal={Journal of Imaging},
  volume={10},
  number={8},
  pages={192},
  year={2024},
  publisher={MDPI}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
