# Landsat Satellite Image Analysis

This project uses deep learning to analyze Landsat satellite imagery for land use and land cover (LULC) classification. It leverages the EuroSAT dataset and PyTorch for training a ResNet50 model with pre-trained weights from Sentinel-2 RGB imagery.

---

## Features

- **Land Use and Land Cover (LULC) classification** using satellite imagery
- **Pre-trained ResNet50 model** with Sentinel-2 RGB weights
- Support for various land cover classes:
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

---

## Prerequisites

- Docker Desktop (recommended)
- Git
- (Optional) Python 3.8+ and pip if running outside Docker

---

## Google Earth Engine Initialization & Setup

This project uses Google Earth Engine (GEE) for satellite image retrieval and geospatial processing.  
**You must authenticate and enable the required services before running any code that uses GEE.**

**1. Enable Google Earth Engine and Cloud Project**

- Sign up at [Google Earth Engine](https://signup.earthengine.google.com/)
- Go to the [Google Cloud Console](https://console.cloud.google.com/)
- Create/select a project
- Enable the **Earth Engine API** (and optionally Google Drive/Cloud Storage APIs)

**2. Install the Google Cloud SDK**

```sh
gcloud init
gcloud auth login
```

**3. Authenticate Earth Engine**

```sh
earthengine authenticate
# or for a specific project
earthengine authenticate --project <your-project-id>
```

**4. (Optional) Set the Project in Your Code**

```python
import ee
ee.Initialize(project='your-project-id')
```

**You must complete these steps before running scripts that use Google Earth Engine (such as `result.py`).**

---

## Pulling the Docker Image from Docker Hub

You can use the pre-built Docker image without building it yourself:

```bash
docker pull chaturx3/landsat-app-simple
```

---

## OS Support and Device Configuration

This project supports Mac (Apple Silicon/Intel), Windows, and Linux. You may need to comment or uncomment certain code blocks in `main.py` depending on your operating system and hardware:

- **For Mac (MPS/CPU):**

  - Use the code block labeled `# for macs` (default in the script):
    ```python
    model_cpu = model.to('cpu') # move to cpu for summary
    torchsummary.summary(model, (3, 224, 224))
    model = model_cpu.to(device) # move back to mps for training
    ```
  - Use the training loop under `# for mac since mixed precision training is not supported`.

- **For Windows (CUDA):**

  - Uncomment the code block labeled `# for windows`:
    ```python
    model = model.to(device) # move model to gpu
    torchsummary.summary(model, (3, 224, 224))
    ```
  - Uncomment the mixed precision training loop under `# mixed precision training for windows` and comment out the Mac-specific training loop.

- **For Linux (CUDA/CPU):**
  - Use the Windows (CUDA) code if you have an NVIDIA GPU, or the Mac/CPU code if you do not.

**Instructions:**

- Open `backend/main.py`.
- Comment or uncomment the relevant code blocks according to your OS and hardware.
- Only one device/training loop block should be active at a time.

---

## Usage

### Run the Complete Pipeline (Download Data, Train, and Predict)

After pulling the image, simply run:

```bash
docker run -it chaturx3/landsat-app-simple:latest
```

This will execute the full pipeline: download the data, train the model, and run predictions, all inside the container.

> **Note:** No volume mounts are needed. All code, data, and models are included in the image.

### (Optional) Get an Interactive Shell

If you want to open a shell inside the container for debugging or exploration:

```bash
docker run -it chaturx3/landsat-app-simple:latest /bin/bash
```

---

## Project Structure

```
landsat/
├── backend/
│   ├── data/               # Data directory (mounted as volume)
│   ├── models/             # Trained models (mounted as volume)
│   ├── uploads/            # Upload directory (mounted as volume)
│   ├── sample_predictions/ # Prediction outputs (mounted as volume)
│   ├── main.py             # Main training script
│   ├── result.py           # Results processing script
│   └── requirements.txt    # Python dependencies
├── Dockerfile              # Docker configuration
└── .dockerignore           # Docker ignore rules
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd landsat
```

### 2. Build the Docker image (Recommended)

```bash
docker build -t landsat-app .
```

### 3. (Alternative) Install Python dependencies with pip (if not using Docker)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r backend/requirements.txt
```

---

## Data

The project uses the EuroSAT dataset, which contains 27,000 labeled Sentinel-2 satellite images covering 10 different land use and land cover classes. The data is organized as follows:

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

---

## Model Architecture

- **Model:** ResNet50 pre-trained on Sentinel-2 RGB imagery
- **Task:** Fine-tuned on EuroSAT for LULC classification

---

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

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
