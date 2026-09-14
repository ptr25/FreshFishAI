# FishFreshAI

Fish freshness detection using deep learning.

## Project Structure

- `train.py` - Model training script
- `evaluate.py` - Model evaluation
- `predict.py` - Prediction/inference
- `model.py` - Model architecture
- `utils.py` - Utility functions
- `results/` - Training results and graphs

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```

### Prediction
```bash
python predict.py --image path/to/image.jpg
```

## Dataset

The dataset contains fish images categorized by freshness levels. Due to its size (13GB), the dataset is not included in this repository.
