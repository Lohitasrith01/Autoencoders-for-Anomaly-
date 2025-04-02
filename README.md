## Autoencoder for Anomaly Detection

## Overview
This repository demonstrates an implementation of an **Autoencoder-based Anomaly Detection** model using different architectures to analyze time-series data. Specifically, this model identifies anomalies in tweet volumes related to Apple Inc. (`AAPL`) from the Numenta Anomaly Benchmark (NAB). The goal is to detect outliers that deviate significantly from typical patterns in time-series data.

## Anomaly Detection with Autoencoders
Autoencoders are a type of neural network used to learn efficient encodings of input data, and they can be used for anomaly detection by analyzing the reconstruction error between the input and output. In the context of this project, the model learns the structure of normal tweet volume and flags unusual patterns (anomalies) based on high reconstruction error.

## Data
The data used in this project comes from the **Numenta Anomaly Benchmark (NAB)**, specifically Apple’s tweet volume. The dataset contains timestamps and corresponding tweet volumes, sampled at 5-minute intervals. The main variable for anomaly detection is the tweet volume, which can exhibit sudden spikes or drops that might indicate anomalies.

The dataset can be downloaded from the following link:
[NAB Dataset - Twitter AAPL](https://www.kaggle.com/boltzmannbrain/nab)

## Autoencoder Architectures
We experiment with three types of autoencoder architectures for anomaly detection:

1. **Dense Autoencoder** - A simple fully connected architecture.
2. **LSTM Autoencoder** - For time-series data, leveraging Long Short-Term Memory layers.
3. **Conv1D Autoencoder** - A convolutional approach for capturing local temporal patterns.

### Architecture Details

#### 1. Dense Autoencoder
- **Encoder**: Two fully connected layers with ReLU activations.
- **Decoder**: Symmetric fully connected layers with ReLU activations.
  
#### 2. LSTM Autoencoder
- **Encoder**: LSTM layer to capture temporal dependencies in the sequence.
- **Decoder**: LSTM layer for sequence reconstruction.
  
#### 3. Conv1D Autoencoder
- **Encoder**: Two Conv1D layers with ReLU activations to capture local features.
- **Decoder**: Two Conv1D layers for sequence reconstruction.

### Model Evaluation
The models are evaluated based on the following metrics:
- **Mean Squared Error (MSE)** between the input and output (reconstruction error).
- **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.
- **R² Score** to measure how well the model generalizes.

### Anomaly Detection
Anomalies are detected by computing the reconstruction error between the input and output. A threshold (95th percentile of the reconstruction errors) is used to classify data points as anomalies.

The best-performing model was the **Conv1D Autoencoder**, which exhibited the lowest validation loss and was most sensitive to anomalies in the data.

## Setup
### Requirements
- Python 3.x
- PyTorch
- Pandas
- NumPy
- Matplotlib, Seaborn
- scikit-learn
- torchinfo

Install required libraries:

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn torchinfo
```
## Key Features
- Multiple Autoencoder Architectures: Experiment with Dense, LSTM, and Conv1D autoencoders.

- Anomaly Detection: Detect anomalies based on the reconstruction error.

- Model Evaluation: Evaluate models using various regression metrics (MAE, RMSE, R²).

- Visualization: Visualize the anomaly detection results with reconstruction error histograms and time-series plots.

## Results
The Conv1D Autoencoder achieved the best results, with the following performance on the test set:

- **Train Loss**: 0.000026

- **Validation Loss**: 0.000006

- **Test Loss**: 0.000111

- **MAE**: 0.001896

- **RMSE**: 0.010520

- **R² Score**: 0.9327

This model detected 118 anomalies out of 2,356 test samples, showing that it effectively learned the normal patterns in tweet volume and flagged unusual behavior.

## Plotting Anomalies
The notebook includes visualizations showing the detected anomalies in tweet volume over time:

- **Anomaly Detection Plot**: Visualize detected anomalies in the original time-series data.

- **Reconstruction Error Distribution**: Plot the distribution of reconstruction errors to understand the threshold for anomaly detection.

## Conclusion
This implementation shows how autoencoders, particularly Conv1D-based architectures, can be used effectively for anomaly detection in time-series data. The project provides insights into the structure of normal tweet volume patterns for AAPL, enabling the detection of significant deviations, such as spam or unexpected events.
