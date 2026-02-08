# GP Project - EEG Model Evaluation Interface

This is a web-based interface for evaluating and visualizing EEG classification models (specifically ATCNet) on Raspberry Pi or local machines. It allows researchers to test different model formats (Keras, TFLite) and strategies (Subject Dependent vs. Subject Independent).

## Features

### 1. Model Evaluation Strategies
*   **Subject Dependent**: Test models trained specifically for a single subject (Subjects 1-9).
*   **Subject Independent (LOSO)**: Test Leave-One-Subject-Out models. currently configured to evaluate on **Subject 1** (tested against a model trained on subjects 2-9).

### 2. Model Format Support
*   **Original (Keras)**: Full precision models (`.keras`).
*   **TFLite Variants**: Optimized models for edge deployment:
    *   Float16 (`FP16`)
    *   Dynamic Range (`Dynamic`)
    *   Integer8 (`INT8`)
    *   Quantized (`quantized`) - *Available in Subject Dependent mode only*

### 3. Interactive Visualization
*   Displays all **22 EEG Channels** using standard BCI Competition IV 2a channel ordering (Fz, C3, Cz, Pz, etc.).
*   Distinct colors for each channel with toggleable visibility.
*   Zoom and hover capabilities using Chart.js.

### 4. Analysis Modes
*   **Single Trial Analysis**: 
    *   Load random trials or jump to a specific trial index.
    *   Run real-time inference.
    *   View predicted class, confidence score, and inference time (ms).
    *   Visual "Correct/Wrong" feedback.
*   **Batch Evaluation (Run All)**:
    *   Process the entire test set for the selected subject.
    *   Calculate **Accuracy (%)** and **Average Inference Time (ms)**.
    *   Generate a detailed results table for every trial.

## Optimization Features
*   **Keras Loading**: Optimized to load the model only once per batch to ensure faster execution (~10ms/trial) while maintaining accuracy matching the standalone script.
*   **Frontend Filtering**: Automatically hides incompatible model options (e.g., hiding 'Quantized' when in LOSO mode) to prevent user error.

## Installation & Usage

1.  **Requirements**:
    *   Python 3.11+
    *   TensorFlow / Keras
    *   Flask
    *   NumPy, SciPy

2.  **Dataset**:
    *   Ensure `BCI2a/` folder is present in the root directory.

3.  **Running the Interface**:
    ```bash
    python3.11 web_interface/app.py
    ```

4.  **Access**:
    *   Open your browser and navigate to `http://localhost:5001` (or the Pi's IP address).

## Project Structure
*   `web_interface/`: Contains the Flask app and frontend assets.
    *   `app.py`: Backend logic (API endpoints, model loading).
    *   `templates/index.html`: Main dashboard UI.
    *   `static/`: CSS styles and JavaScript logic.
*   `models/`: Directory containing trained `.keras` and `.tflite` models.
*   `main.py` / `preprocess.py`: Core logic for inference and data loading.
