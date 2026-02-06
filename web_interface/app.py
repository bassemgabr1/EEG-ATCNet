import sys
import os
import time
import numpy as np
import json
from flask import Flask, render_template, request, jsonify

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import get_data
# Import inference functions from main.py
# We might need to slightly refactor main.py or just import the functions if they are clean.
# Looking at main.py, run_inference, select_trial, etc are available.
from main import run_inference, select_trial

app = Flask(__name__)

# --- Global State ---
CURRENT_MODEL = None
CURRENT_MODEL_PATH = None
CURRENT_DATA = {
    "X_test": None,
    "y_test": None, # One-hot
    "y_test_labels": None # Integer class indices
}
CURRENT_CONFIG = {
    "subject": 1,
    "loso": False
}

CLASSES_LABELS = ['Left hand', 'Right hand', 'Foot', 'Tongue']

MODELS_DIR_DEP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'subject-dependent')
MODELS_DIR_INDEP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'subject-independent')

# --- Helper Functions ---

def load_dataset_if_needed(subject, loso):
    global CURRENT_DATA, CURRENT_CONFIG
    
    # Reload if subject or strategy changed, or if data is missing
    if (CURRENT_DATA["X_test"] is None or 
        CURRENT_CONFIG["subject"] != subject or 
        CURRENT_CONFIG["loso"] != loso):
        
        print(f"Loading data for Subject {subject}, LOSO={loso}...")
        try:
            # Note: preprocess.get_data returns: X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot
            # We only need test data
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BCI2a/')
            _, _, _, X_test, y_test_labels, y_test_onehot = get_data(
                data_path,
                subject=subject-1,
                dataset='BCI2a',
                LOSO=loso,
                isStandard=True
            )
            CURRENT_DATA["X_test"] = X_test
            CURRENT_DATA["y_test"] = y_test_onehot
            CURRENT_DATA["y_test_labels"] = y_test_labels
            CURRENT_CONFIG["subject"] = subject
            CURRENT_CONFIG["loso"] = loso
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e

def get_model_path(subject, loso, model_type):
    # model_type examples: "org", "quantized", "Dynamic", "FP16", "INT8"
    # Filename patterns based on ls:
    # subject-dependent: subject_1_org.keras, subject_1_quantized.tflite, etc.
    # subject-independent: subject_1_org.keras (assumed same naming convention)
    
    base_dir = MODELS_DIR_INDEP if loso else MODELS_DIR_DEP
    
    # Map UI model types to filenames
    # UI: Original -> org.keras
    # UI: Quantized -> quantized.tflite
    # UI: Dynamic Range -> Dynamic.tflite
    # UI: Float16 -> FP16.tflite
    # UI: Integer8 -> INT8.tflite
    
    ext = ".tflite"
    suffix = model_type
    
    if model_type == "original":
        suffix = "org"
        ext = ".keras"
    elif model_type == "quantized":
        suffix = "quantized"
    elif model_type == "dynamic":
        suffix = "Dynamic"
    elif model_type == "fp16":
        suffix = "FP16"
    elif model_type == "int8":
        suffix = "INT8"
        
    filename = f"subject_{subject}_{suffix}{ext}"
    return os.path.join(base_dir, filename)

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init', methods=['GET'])
def init_api():
    # Return available options
    return jsonify({
        "subjects": list(range(1, 10)),
        "model_types": [
            {"id": "original", "name": "Original (Keras)"},
            {"id": "quantized", "name": "Quantized (TFLite)"},
            {"id": "dynamic", "name": "Dynamic Range (TFLite)"},
            {"id": "fp16", "name": "Float16 (TFLite)"},
            {"id": "int8", "name": "Integer8 (TFLite)"}
        ]
    })

@app.route('/api/load_model', methods=['POST'])
def load_model_route():
    global CURRENT_MODEL_PATH
    data = request.json
    subject = int(data.get('subject', 1))
    loso = data.get('loso', False)
    model_type = data.get('model_type', 'original')
    
    try:
        # 1. Locate Model
        model_path = get_model_path(subject, loso, model_type)
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {os.path.basename(model_path)}"}), 404
        
        # 2. Check if we can load it (lazy check, actual load happens in inference usually for TFLite, 
        # but for Keras we might want to pre-load if we were keeping it in memory.
        # Given existing main.py structure, run_keras loads it every time? 
        # Wait, main.py: run_keras -> model = load_model(model_path). 
        # This is SLOW if done per prediction. 
        # Ideally we should load it once globally. 
        # I will implement global loading here for Keras. TFLite is fast to load usually.
        
        print(f"Set current model to: {model_path}")
        CURRENT_MODEL_PATH = model_path
        
        # 3. Pre-load Data for this subject
        start_time = time.time()
        load_dataset_if_needed(subject, loso)
        data_load_time = time.time() - start_time
        
        return jsonify({
            "message": f"Model selected: {os.path.basename(model_path)}",
            "data_load_time": data_load_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_trial', methods=['POST'])
def get_trial_route():
    data = request.json
    trial_mode = data.get('mode', 'random') # 'random' or 'index'
    trial_index = data.get('index', 0)
    
    if CURRENT_DATA["X_test"] is None:
        return jsonify({"error": "Data not loaded. Load a model first."}), 400
        
    num_trials = CURRENT_DATA["X_test"].shape[0]
    
    if trial_mode == 'random':
        idx = np.random.randint(0, num_trials)
    else:
        idx = int(trial_index)
        if idx >= num_trials: idx = num_trials - 1
        
    # Get Data
    # X_test shape: (Trials, 1, Channels, Timepoints) -> (Trials, 1, 22, 1125)
    # We want to return specific trial data for visualization
    # Let's return just the first channel or a few channels for viz, or all?
    # All is fine, it's 22 * 1125 floats. approx 100KB JSON. Acceptable.
    
    trial_data = CURRENT_DATA["X_test"][idx, 0, :, :] # Shape (22, 1125)
    true_label = int(CURRENT_DATA["y_test_labels"][idx])
    
    return jsonify({
        "trial_index": idx,
        "total_trials": num_trials,
        "eeg_data": trial_data.tolist(), # List of 22 lists
        "true_label": true_label,
        "true_label_name": CLASSES_LABELS[true_label]
    })

@app.route('/api/predict', methods=['POST'])
def predict_route():
    data = request.json
    trial_index = data.get('index')
    
    if CURRENT_MODEL_PATH is None:
        return jsonify({"error": "No model loaded"}), 400
    if CURRENT_DATA["X_test"] is None:
        return jsonify({"error": "Data not loaded"}), 400
        
    # Prepare Input
    # select_trial returns: x (1, 1, 22, 1125), y (1, 4), index
    x, y_true, idx = select_trial(CURRENT_DATA["X_test"], CURRENT_DATA["y_test"], trial_index)
    
    try:
        # Run Inference
        # run_inference returns: y_pred (1, 4), time
        y_pred_prob, inference_time = run_inference(CURRENT_MODEL_PATH, x)
        
        pred_class = int(np.argmax(y_pred_prob))
        true_class = int(np.argmax(y_true))
        confidence = float(y_pred_prob[0][pred_class])
        
        return jsonify({
            "predicted_class": pred_class,
            "predicted_label": CLASSES_LABELS[pred_class],
            "true_class": true_class,
            "true_label": CLASSES_LABELS[true_class],
            "correct": (pred_class == true_class),
            "confidence": confidence,
            "inference_time_ms": inference_time * 1000
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_all', methods=['POST'])
def predict_all_route():
    # Run all trials and return table data
    if CURRENT_MODEL_PATH is None:
        return jsonify({"error": "No model loaded"}), 400
    if CURRENT_DATA["X_test"] is None:
        return jsonify({"error": "Data not loaded"}), 400
        
    results = []
    correct_count = 0
    total_time = 0
    
    X_test = CURRENT_DATA["X_test"]
    y_test = CURRENT_DATA["y_test"] # one hot
    
    # We need to optimize this loop. Loading Keras model inside loop is bad.
    # Logic in `main.py` -> `run_keras` -> `load_model` is effectively loading it EVERY CALL.
    # We MUST optimize this for loop usage.
    
    # Check model type
    ext = os.path.splitext(CURRENT_MODEL_PATH)[1].lower()
    model_instance = None
    is_tflite = False
    
    try:
        if ext == ".tflite":
            is_tflite = True
        elif ext in [".keras", ".h5"]:
            # Load ONCE
            from keras import config
            config.enable_unsafe_deserialization()
            from tensorflow.keras.models import load_model
            import models # Required for custom layers - ensures they are registered
            
            print(f"Pre-loading Keras model from {CURRENT_MODEL_PATH}...")
            model_instance = load_model(CURRENT_MODEL_PATH)
            
    except Exception as e:
        print(f"Model load error: {e}")
        return jsonify({"error": f"Failed to load model: {e}"}), 500
    
    num_trials = X_test.shape[0]

    for i in range(num_trials):
        x = X_test[i:i+1] # (1, 1, 22, 1125)
        y = y_test[i:i+1]
        true_class = int(np.argmax(y))
        
        try:
            if is_tflite:
                # TFLite inference (still per-trial load as optimized in main.py logic/lower overhead)
                y_pred_prob, inf_time = run_inference(CURRENT_MODEL_PATH, x)
                total_time += inf_time
            else:
                # Keras optimized inference
                t_start = time.time()
                y_pred_prob = model_instance.predict(x, verbose=0)
                t_end = time.time()
                inf_time = t_end - t_start
                total_time += inf_time
            
            pred_class = int(np.argmax(y_pred_prob))
            is_correct = (pred_class == true_class)
            if is_correct: correct_count += 1
            
            results.append({
                "id": i,
                "true": CLASSES_LABELS[true_class],
                "pred": CLASSES_LABELS[pred_class],
                "correct": is_correct,
                "time_ms": round(inf_time * 1000, 2)
            })
            
        except Exception as e:
            print(f"Error processing trial {i}: {e}")
            results.append({
                "id": i,
                "true": CLASSES_LABELS[true_class],
                "pred": "ERROR",
                "correct": False,
                "time_ms": 0
            })
        
    accuracy = (correct_count / num_trials) * 100
    avg_time = (total_time / num_trials) * 1000
    
    return jsonify({
        "accuracy": accuracy,
        "avg_time_ms": avg_time,
        "total_trials": num_trials,
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
