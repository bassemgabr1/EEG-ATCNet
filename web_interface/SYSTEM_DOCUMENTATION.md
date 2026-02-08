# System Documentation: EEG ATCNet Evaluation Interface (GP Project)

This document provides a comprehensive technical overview of the EEG Evaluation System, designed for testing and visualizing ATCNet models on the BCI Competition IV 2a dataset.

---

## 1. System Architecture

The system follows a client-server architecture:

*   **Frontend**: A responsive web interface (HTML/CSS/JS) that manages user interaction, visualization (Chart.js), and state (Subject/Model selection).
*   **Backend**: A Flask (Python) server that handles model loading, data preprocessing, and inference.
*   **Data Layer**: Direct file system access to the BCI2a dataset (`.mat` files) and trained models (`.keras`, `.tflite`).

### Data Flow
1.  **Selection**: User selects a Subject (1-9) and Model Format (e.g., Original, TFLite INT8).
2.  **Loading**: 
    *   Backend loads the corresponding model file from `models/`.
    *   Backend loads the test data for that subject using `preprocess.py`.
3.  **Inference**:
    *   **Single**: User requests a trial -> Backend slices data -> Model predicts -> Result returned.
    *   **Batch**: Backend iterates through all test trials -> Aggregates results -> Returns accuracy & speed metrics.

### GPIO Control (Raspberry Pi 5)
*   **Trigger**: When the model predicts **"Right hand"**.
*   **Action**: Flashes the **Built-in Activity LED (Green)** for 1 second.
*   **Implementation**: Uses `gpiozero` with `LED('ACT')`.
*   **Fallback**: Includes a mock simulation mode for development on non-Pi devices.

---

## 2. Models & File Structure

The system is designed to work with specific naming conventions in the `models/` directory.

### Directory Structure
```
EEG-ATCNet/
├── models/
│   ├── subject-dependent/      # Models trained on specific subjects
│   │   ├── subject_1_org.keras         # Original Keras Model (Float32)
│   │   ├── subject_1_quantized.tflite  # Post-training Quantization
│   │   ├── subject_1_Dynamic.tflite    # Dynamic Range Quantization
│   │   ├── subject_1_FP16.tflite       # Float16 Quantization
│   │   └── subject_1_INT8.tflite       # Integer8 Quantization
│   │   └── ... (Repeats for subjects 1-9)
│   └── subject-independent/    # LOSO Models (Leave-One-Subject-Out)
│       └── subject_1_org.keras         # Model tested on Subj 1, trained on 2-9
```

### Supported Model Formats
1.  **Original (.keras)**:
    *   **Type**: TensorFlow/Keras SavedModel.
    *   **Use Case**: Baseline accuracy, research validation.
    *   **Optimization**: During batch processing, this model is loaded *once* into memory to ensure high-speed evaluation (~10ms/trial).
2.  **TFLite (.tflite)**:
    *   **Type**: TensorFlow Lite (FlatBuffers).
    *   **Variants**:
        *   **FP16**: Half-precision floating point (2x smaller, slightly faster).
        *   **Dynamic**: Weights quantized to 8-bit, computations in float.
        *   **INT8**: Full integer quantization (smallest, fastest, lowest power).
        *   **Quantized**: Generic post-training quantization.
    *   **Use Case**: Edge deployment (Raspberry Pi), low latency.

---

## 3. API Endpoints

The Flask server exposes the following RESTful endpoints:

### `GET /api/init`
*   **Purpose**: Returns initial configuration data to the frontend.
*   **Response**:
    ```json
    {
      "subjects": [1, 2, ... 9],
      "model_types": [
        {"id": "original", "name": "Original (Keras)"},
        {"id": "int8", "name": "Integer8 (TFLite)"},
        ...
      ]
    }
    ```

### `POST /api/load_model`
*   **Purpose**: Loads a specific model and the corresponding test dataset into memory.
*   **Payload**:
    ```json
    {
      "subject": 1,
      "loso": false,       // true = Subject Independent
      "model_type": "int8"
    }
    ```
*   **Logic**:
    *   Constructs filepath based on naming convention (e.g., `subject_1_INT8.tflite`).
    *   Calls `preprocess.get_data()` to load test data for Subject 1.
    *   If **Keras**: Pre-loads model graph if needed.
    *   If **TFLite**: Verifies file existence.

### `POST /api/get_trial`
*   **Purpose**: Retrieves EEG data for visualization and specific trial metadata.
*   **Payload**: `{"mode": "random" | "index", "index": 0}`
*   **Response**:
    ```json
    {
      "trial_index": 42,
      "total_trials": 515,
      "eeg_data": [[...], ...], // 22 channels x 1125 timepoints
      "true_label": 0,
      "true_label_name": "Left hand"
    }
    ```

### `POST /api/predict`
*   **Purpose**: Runs inference on a single trial.
*   **Payload**: `{"index": 42}`
*   **Response**:
    ```json
    {
      "predicted_class": 0,
      "predicted_label": "Left hand",
      "confidence": 0.98,
      "inference_time_ms": 12.5,
      "correct": true
    }
    ```

### `POST /api/predict_all`
*   **Purpose**: Runs batch evaluation on the entire loaded test set.
*   **Payload**: `{}` (Uses currently loaded model/data)
*   **Internal Logic**:
    *   **Keras Optimization**: Loads the model *once* before the loop using `tensorflow.keras.models.load_model` with custom objects (`MultiHeadAttention_LSA`, etc.).
    *   **TFLite Logic**: Creates a new interpreter for each trial (fast creation) or reuses it.
*   **Response**:
    ```json
    {
      "accuracy": 76.5,
      "avg_time_ms": 9.2,
      "total_trials": 288,
      "results": [
        {"id": 0, "true": "Left", "pred": "Left", "correct": true, "time_ms": 8},
        ...
      ]
    }
    ```

---

## 4. Frontend Logic

### State Management (`script.js`)
*   **state object**: Tracks `subject`, `loso` (strategy), `modelType`, and `currentTrialIndex`.
*   **Event Listeners**: Updates UI immediately upon selection changes (e.g., disabling Subject selector when `loso=true`).

### Visualization (`Chart.js`)
*   **All Channels**: Renders 22 lines corresponding to the 10-20 system channels in BCI2a (Fz, FC3, FC1... P2, POz).
*   **Performance**: Uses simplified point rendering (`pointRadius: 0`) and `interaction: { mode: 'nearest' }` to maintain high frame rates even with dense EEG data.

### Filtering Logic
*   When **Subject Independent** is selected:
    *   Subject selector is locked to **1**.
    *   Model dropdown filters out **Quantized** options (showing only Original/compatible types), preventing file-not-found errors.

---

## 5. Deployment Notes (Raspberry Pi)

1.  **Dependencies**:
    *   The system uses `tensorflow` (heavy) or `tflite-runtime` (light).
    *   For Pi, `tflite-runtime` is recommended for TFLite models, but `tensorflow` is required for Keras models.
2.  **Performance**:
    *   **Keras Models**: ~1-2s loading, ~100-200ms inference (optimized to ~10ms in batch).
    *   **TFLite Models**: Instant loading, ~5-20ms inference depending on quantization (INT8 is fastest).

## 6. Verification Steps
To verify the system is working correctly:
1.  **Load Model**: Select Topic 1, Dependent, Original. Click Load. Status should turn green ("Ready").
2.  **Visual Check**: Chart should populate with 22 distinct colored lines.
3.  **Single Pred**: Click "Run Prediction". Check if result makes sense (Confidence > 25%).
4.  **Batch Run**: Click "Batch Evaluation" -> "Run All Trials". 
    *   Watch progress.
    *   Verify final accuracy is >60% (random chance is 25%).
    *   Download or view the table for detailed metrics.
