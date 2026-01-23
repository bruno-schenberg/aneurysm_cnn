# 3D CNN for Aneurysm Detection

This project provides a complete pipeline for training and evaluating 3D Convolutional Neural Networks (CNNs) for aneurysm detection in medical imaging data. The workflow is designed to be modular and experiment-driven, allowing for easy configuration and comparison of different models, data preprocessing techniques, and balancing strategies.

## Features

*   **Modular Data Pipeline**: Separate scripts for cleaning raw data and generating training-ready datasets.
*   **Experiment-Driven Training**: Define and manage multiple training runs through a simple `experiments.json` configuration file.
*   **Configurable Training**: Easily adjust hyperparameters, model architecture, data balancing, and dataset choice for each experiment.
*   **Cross-Validation**: Built-in support for k-fold cross-validation to ensure robust model evaluation.
*   **Hold-Out Test Set**: Option to reserve a final test set for unbiased performance assessment after model selection.
*   **Reproducibility**: Track experiments with a fixed random seed and organized output directories.

## Core Libraries

This project is built using Python and relies on several key libraries for deep learning and medical image processing.

*   **PyTorch**: The primary deep learning framework used for building and training the 3D CNN models.
*   **MONAI**: A PyTorch-based framework for deep learning in healthcare imaging, used for data loading, transformations, and models.
*   **NiBabel**: For reading and writing medical imaging formats, specifically NIfTI (`.nii.gz`).
*   **pydicom**: Used in the `data_cleaner.py` script to parse and validate raw DICOM files.
*   **NumPy & SciPy**: For numerical operations and scientific computing, particularly in image manipulation and data preprocessing.

### Installation

It is recommended to install these dependencies using `pip`. You can create a `requirements.txt` file to manage the packages for easy setup.
```bash
pip install -r requirements.txt
```

## Project Structure

```
aneurysm_cnn/
├── experiments/                # Output directory for all training results
├── data_engine/
│   ├── src/
│   │   ├── class_utils.py      # Joins classification data (e.g., aneurysm labels) from classes.csv.
│   │   ├── dicom_utils.py      # Validates DICOM series, sorts slices, and extracts key metadata.
│   │   ├── file_utils.py       # Organizes raw data folders, standardizes names, and identifies data paths.
│   │   ├── nifti_utils.py      # Converts validated DICOM series into NIfTI (.nii.gz) format.
│   │   └── nifti_resize.py     # Resizes and pads NIfTI images to a uniform, isotropic shape.
│   │
│   ├── dataset_gen.py          # Generates resized datasets (e.g., isotropic) from NIfTI files.
│   ├── data_cleaner.py         # Main pipeline to validate, clean, and convert raw DICOM data to NIfTI.
│   ├── requirements.txt        # Lists all Python package dependencies for dataset preparation.
│   └── classes.csv             # Manifest file with exam IDs, classification labels, and metadata.
│
├── training_engine/
│   ├── src/
│   │   ├── data_preprocess.py  # Handles data loading, MONAI transforms (augmentation), and k-fold splitting.
│   │   ├── models.py           # Defines 3D CNN architectures (e.g., R3D-18, UNet3D-Classifier).
│   │   ├── orchestrator.py     # Manages the execution of training folds/experiments
│   │   ├── plots.py            # Generates evaluation plots (e.g., confusion matrix, ROC) and metric reports.
│   │   └── training.py         # Implements the core training and validation loops per epoch.
│   │
│   ├── train_models.py         # Main entry point to run all training experiments defined in `experiments.json`.
│   ├── requirements.txt        # Lists all Python package dependencies for training and evaluating the models.
│   └── experiments.json        # Configuration file defining the list of experiments to run.
│
├── .gitignore                  # Specifies files/directories for Git to ignore (e.g., outputs, caches).
└── README.md                   # This file.
```

## Workflow

The project follows a three-step workflow: Data Preparation, Experiment Configuration, and Training.

### 1. Data Preparation

This stage involves cleaning the raw medical scans and generating datasets suitable for training.

*   **`data_cleaner.py`**: This script should be run first. It is responsible for initial preprocessing tasks such as filtering out corrupted files, standardizing formats.
*   **`dataset_gen.py`**: After cleaning, this script generates the final datasets. It can create multiple versions of the data with different preprocessing steps, such as resampling, cropping, or shrinking, as defined by the paths in `train_models.py`.

### 2. Configure Experiments

All training runs are defined in the `experiments.json` file. Each object in the JSON array represents a single experiment with a specific configuration.

The main script (`train_models.py`) uses a default configuration, which can be overridden by the parameters you set in `experiments.json`.

**Required Parameters in `experiments.json`:**

*   `name`: A unique name for the experiment (e.g., "ResNet_Balanced_DatasetA").
*   `model`: The name of the model architecture to use.
*   `balancing`: The data balancing strategy (e.g., "oversampling", "none").
*   `data_path_key`: The key corresponding to the desired dataset (e.g., "DATASET_A", "DATASET_B").

**Example `experiments.json`:**

```json
[
  {
    "name": "3D-ResNet_With-Oversampling_On-Cropped-Data",
    "model": "ResNet3D",
    "balancing": "oversampling",
    "data_path_key": "DATASET_A",
    "LEARNING_RATE": 0.001,
    "EPOCHS": 50
  },
  {
    "name": "SimpleCNN_No-Balancing_On-Shrunk-Data",
    "model": "SimpleCNN",
    "balancing": "none",
    "data_path_key": "DATASET_B",
    "EPOCHS": 30
  }
]
```

### 3. Run Training

Once your data is prepared and your experiments are configured, you can start the training process.

Execute the main training script from your terminal:
```bash
python train_models.py
```

The script will:
1.  Load the experiment definitions from `experiments.json`.
2.  Prepare a configuration for each experiment, merging it with the defaults.
3.  Call the `run_all_experiments` orchestrator, which will loop through each configuration and run the training and evaluation, including k-fold cross-validation.

### 4. Review Results

All outputs, including trained model weights, logs, and performance metrics for each fold, will be saved in the `./experiments` directory. Each experiment will have its own sub-directory named after the `name` you provided in the configuration.

## Configuration Details

You can modify the default training behavior by changing the `DEFAULT_CONFIG` dictionary in `train_models.py` or by overriding parameters in `experiments.json` for specific runs.

Key parameters include:

*   `LEARNING_RATE`: The learning rate for the optimizer.
*   `BATCH_SIZE`: The number of samples per training batch.
*   `EPOCHS`: The number of training epochs.
*   `N_SPLITS`: The number of folds for cross-validation.
*   `QUICK_TEST`: If `True`, runs only one fold of one experiment for fast debugging.
*   `HOLD_OUT_TEST_SET`: If `True`, the final test set is held out, and evaluation during cross-validation is performed on the validation set.

This structured approach allows for systematic exploration and robust evaluation of models for aneurysm detection.