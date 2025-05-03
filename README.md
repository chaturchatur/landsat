# Landsat Satellite Image Analysis

This project uses deep learning to analyze Landsat satellite imagery for land use and land cover (LULC) classification. It leverages the EuroSAT dataset and PyTorch for training a ResNet50 model with pre-trained weights from Sentinel-2 RGB imagery.

## Features

- Land Use and Land Cover (LULC) classification using satellite imagery
- Pre-trained ResNet50 model with Sentinel-2 RGB weights
- Support for various land cover classes including:
  - Annual Crop
  - Forest
  - Herbaceous Vegetation
  - Highway
  - Industrial
  - Pasture
  - Permanent Crop
  - Residential
  - River
  - Sea/Lake

## Prerequisites

- Docker Desktop
- Git

## Project Structure

```
.
├── backend/
│   ├── data/               # Data directory (mounted as volume)
│   ├── models/            # Trained models (mounted as volume)
│   ├── uploads/           # Upload directory (mounted as volume)
│   ├── sample_predictions/# Prediction outputs (mounted as volume)
│   ├── main.py           # Main training script
│   ├── result.py         # Results processing script
│   └── requirements.txt  # Python dependencies
├── Dockerfile            # Docker configuration
└── .dockerignore        # Docker ignore rules
```

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd landsat
   ```

2. Build the Docker image:
   ```bash
   docker build -t landsat-app .
   ```

## Usage

### Running the Training Script

```bash
docker run -it \
  -v "$(pwd)/backend/data:/app/data" \
  -v "$(pwd)/backend/models:/app/models" \
  -v "$(pwd)/backend/uploads:/app/uploads" \
  -v "$(pwd)/backend/sample_predictions:/app/sample_predictions" \
  landsat-app main
```

### Running the Results Script

```bash
docker run -it \
  -v "$(pwd)/backend/data:/app/data" \
  -v "$(pwd)/backend/models:/app/models" \
  -v "$(pwd)/backend/uploads:/app/uploads" \
  -v "$(pwd)/backend/sample_predictions:/app/sample_predictions" \
  landsat-app result
```

### Getting an Interactive Shell

```bash
docker run -it \
  -v "$(pwd)/backend/data:/app/data" \
  -v "$(pwd)/backend/models:/app/models" \
  -v "$(pwd)/backend/uploads:/app/uploads" \
  -v "$(pwd)/backend/sample_predictions:/app/sample_predictions" \
  landsat-app shell
```

## Data

The project uses the EuroSAT dataset, which contains 27,000 labeled Sentinel-2 satellite images covering 10 different land use and land cover classes. The data is organized in the following structure:

```
data/
└── EuroSAT/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```

## Model Architecture

The project uses a ResNet50 model pre-trained on Sentinel-2 RGB imagery. The model is fine-tuned on the EuroSAT dataset for LULC classification.

## Dependencies

Key Python packages used in this project:

- torchgeo
- lightning
- torch
- timm
- matplotlib
- numpy
- rasterio
- scikit-learn
- geopandas
- earthengine-api
- geemap

## License

This project is licensed under the MIT License - see the LICENSE file for details.
