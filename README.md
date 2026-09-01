# ASL to Text Converter: Real-Time Sign Language Recognition

A real-time American Sign Language (ASL) recognition system capable of identifying 250 distinct signs. This project processes spatial-temporal graph data from hand, pose, and facial landmarks into an optimized neural network architecture, translating continuous sign language into text.

##  Project Progression & Pipeline
This repository is structured to demonstrate an end-to-end Machine Learning lifecycle, progressing from a simple baseline to a highly optimized, competition-grade architecture.

1. **`01_EDA_Baseline_and_Feature_Engineering.ipynb`**
   * **Exploratory Data Analysis:** Analyzes raw Parquet data containing 543 spatial landmarks per frame.
   * **Baseline Model:** Establishes a Bidirectional LSTM baseline trained on 126 raw coordinate features.
   * **Feature Engineering:** Introduces frame-to-frame velocity kinematics, doubling the feature set to 252 to capture motion dynamics and significantly improve baseline accuracy.
2. **`02_Local_Preprocessing_CNN_Transformer.ipynb`**
   * **Architecture Upgrade:** Transitions to an advanced hybrid architecture featuring Depthwise-separable 1D Convolutions, Efficient Channel Attention (ECA), and Multi-Head Self-Attention (Transformer) blocks.
   * **Pipeline Optimization:** Implements multithreaded local preprocessing to extract 130 essential landmarks, apply translation invariance (nose-centric normalization), and compute motion lags.
3. **`03_TFRecord_Dynamic_Augmentation_Model.ipynb`**
   * **Data Serialization:** Converts processed datasets into compressed TFRecord shards for high-throughput `tf.data` pipeline feeding.
   * **Dynamic Augmentation:** Applies on-the-fly spatial and temporal augmentations (e.g., temporal masking, spatial affine transformations, horizontal flipping) to prevent overfitting and improve model generalization.

##  Real-Time Live Inference
The repository includes a live webcam inference script that utilizes Google MediaPipe to extract coordinates in real-time and passes them through the trained model.

### Dependencies
Ensure you have Python 3.9+ installed, then install the required packages:
```bash
pip install tensorflow mediapipe opencv-python numpy pandas
