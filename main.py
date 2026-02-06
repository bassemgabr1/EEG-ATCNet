import sys
import os
import time
import numpy as np
import tensorflow as tf

from preprocess import get_data
tf.config.set_visible_devices([], 'GPU')

# ------------------------------
# Dataset parameters
# ------------------------------
in_samples = 1125
n_channels = 22
n_classes = 4
classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
data_path = './BCI2a/'


# =====================================================
# Load dataset
# =====================================================
def load_test_data(subj,LOSO=False):
    _, _, _, X_test, _, y_test_onehot = get_data(
        data_path,
        subject=subj,
        dataset='BCI2a',
        LOSO=False,
        isStandard=True
    )
    return X_test, y_test_onehot


# =====================================================
# Select one trial
# =====================================================
def select_trial(X_test, y_test, index=None):
    if index is None:
        index = np.random.randint(0, X_test.shape[0])

    x = X_test[index:index+1]
    y = y_test[index:index+1]

    return x, y, index


# =====================================================
# TFLite inference
# =====================================================
def run_tflite(model_path, x_input):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantization handling
    if input_details[0]['dtype'] == np.float32:
        x_input = x_input.astype(np.float32)

    elif input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        x_input = x_input / scale + zero_point
        x_input = x_input.astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], x_input)

    start = time.time()
    interpreter.invoke()
    end = time.time()

    y_pred = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        y_pred = scale * (y_pred.astype(np.float32) - zero_point)

    return y_pred, (end - start)


# =====================================================
# Keras inference
# =====================================================
def run_keras(model_path, x_input):
    from keras import config
    from tensorflow.keras.models import load_model

    config.enable_unsafe_deserialization()
    import models  # required if custom layers exist

    model = load_model(model_path)

    start = time.time()
    y_pred = model.predict(x_input, verbose=0)
    end = time.time()

    return y_pred, (end - start)


# =====================================================
# Unified inference wrapper
# =====================================================
def run_inference(model_path, x_input):
    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".tflite":
        print("Using TFLite model")
        return run_tflite(model_path, x_input)

    elif ext in [".keras", ".h5"]:
        print("Using Keras model")
        return run_keras(model_path, x_input)

    else:
        raise ValueError(f"Unsupported model format: {ext}")


# =====================================================
# Evaluate one trial
# =====================================================
def evaluate_trial(model_path,subj,LOSO, trial_index=None):
    X_test, y_test = load_test_data(subj,LOSO)

    x, y_true, idx = select_trial(X_test, y_test, trial_index)

    y_pred_prob, inference_time = run_inference(model_path, x)

    y_pred_class = np.argmax(y_pred_prob, axis=-1)[0]
    true_class = np.argmax(y_true, axis=-1)[0]

    success = y_pred_class == true_class

    print("\n=== Results ===")
    print(f"Trial index: {idx}")
    print(f"Prediction correct: {success}")
    print(f"Predicted: {y_pred_class} ({classes_labels[y_pred_class]})")
    print(f"True: {true_class} ({classes_labels[true_class]})")
    print(f"Inference time: {inference_time * 1000:.3f} ms")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    MODEL_PATH = "models/subject-dependent/subject_1_org.keras"
    subject=1
    LOSO=False #Point to subject-dependent or subject-indepent (if true)
    # MODEL_PATH = "full_model.keras"

    evaluate_trial(MODEL_PATH,subject,LOSO)