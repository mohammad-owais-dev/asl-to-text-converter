import os
# This line MUST come before importing mediapipe or tensorflow
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import json


cap = cv2.VideoCapture(0)  # Initialize the webcam
# Initialize MediaPipe Holistic or Hands (matching your feature engineering pipeline)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Setup sequence buffer configuration
max_len = 30  # Adjust to match your CFG.max_len if different
sequence_buffer = []

print("Starting webcam stream. Press 'q' to exit.")
class CFG:
    seed = 42
    max_len = 64
    batch_size = 32
    epoch = 150
    dim = 192

ROWS_PER_FRAME = 543
NUM_CLASSES = 250
PAD = -100.

NOSE = [1, 2, 98, 327]
LNOSE = [98]
RNOSE = [327]
LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]
REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]
LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
CHANNELS = 6 * len(POINT_LANDMARKS)

# 4. Data Pipeline
def interp1d_(x, target_len, method='random'):
    target_len = tf.maximum(1, target_len)
    if method == 'random':
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bilinear')
        elif tf.random.uniform(()) < 0.5:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bicubic')
        else:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'nearest')
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), method)
    return x
def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))
class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=CFG.max_len, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        x = inputs[None,...] if tf.rank(inputs) == 3 else inputs
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        x = (x - mean) / std
        if self.max_len is not None:
            x = x[:,:self.max_len]
        length = tf.shape(x)[1]
        x = x[...,:2]
        dx = tf.cond(tf.shape(x)[1]>1, lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]), lambda:tf.zeros_like(x))
        dx2 = tf.cond(tf.shape(x)[1]>2, lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]), lambda:tf.zeros_like(x))
        x = tf.concat([
            tf.reshape(x, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)
        return tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)
    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        return inputs * tf.nn.sigmoid(tf.squeeze(self.conv(tf.expand_dims(nn, -1)), -1))[:,None,:]

class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate)
    def build(self, input_shape):
        super().build(input_shape)
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA, trainable=False)
    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < self.start_step, lambda:inputs, lambda:self.dropout(inputs, training=training))
        if training: self._train_counter.assign_add(1)
        return x

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0))
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(kernel_size, strides=1, dilation_rate=dilation_rate, padding='valid', use_bias=False)
        self.supports_masking = True
    def call(self, inputs): return self.dw_conv(self.causal_pad(inputs))

def Conv1DBlock(channel_size, kernel_size, drop_rate=0.2):
    def apply(inputs):
        skip = inputs
        x = tf.keras.layers.Dense(tf.keras.backend.int_shape(inputs)[-1] * 2, use_bias=True, activation='swish')(inputs)
        x = CausalDWConv1D(kernel_size)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = ECA()(x)
        x = tf.keras.layers.Dense(channel_size, use_bias=True)(x)
        if drop_rate > 0: x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        return tf.keras.layers.add([x, skip]) if tf.keras.backend.int_shape(inputs)[-1] == channel_size else x
    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.num_heads, self.scale = dim, num_heads, dim ** -0.5
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True
    def call(self, inputs, mask=None):
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(self.qkv(inputs)))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        attn = tf.keras.layers.Softmax(axis=-1)(tf.matmul(q, k, transpose_b=True) * self.scale, mask=mask[:, None, None, :] if mask is not None else None)
        return self.proj(tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(self.drop1(attn) @ v)))

def TransformerBlock(dim=256, expand=2):
    def apply(inputs):
        x = tf.keras.layers.Add()([inputs, tf.keras.layers.Dropout(0.2, noise_shape=(None,1,1))(MultiHeadSelfAttention(dim=dim, num_heads=4, dropout=0.2)(tf.keras.layers.BatchNormalization(momentum=0.95)(inputs)))])
        return tf.keras.layers.Add()([x, tf.keras.layers.Dropout(0.2, noise_shape=(None,1,1))(tf.keras.layers.Dense(dim, use_bias=False)(tf.keras.layers.Dense(dim*expand, use_bias=False, activation='swish')(tf.keras.layers.BatchNormalization(momentum=0.95)(x))))])
    return apply


def get_model(max_len=CFG.max_len, dropout_step=0, dim=CFG.dim):
    inp = tf.keras.Input((max_len, CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD, input_shape=(max_len, CHANNELS))(inp)
    x = tf.keras.layers.BatchNormalization(momentum=0.95)(tf.keras.layers.Dense(dim, use_bias=False)(x))

    for _ in range(3): x = Conv1DBlock(dim, 17)(x)
    x = TransformerBlock(dim)(x)
    for _ in range(3): x = Conv1DBlock(dim, 17)(x)
    x = TransformerBlock(dim)(x)

    x = LateDropout(0.8, start_step=dropout_step)(
        tf.keras.layers.GlobalAveragePooling1D()(tf.keras.layers.Dense(dim * 2, activation=None)(x)))
    return tf.keras.Model(inp, tf.keras.layers.Dense(NUM_CLASSES)(x))

# ==========================================
# INITIALIZATION BLOCK
# ==========================================
# Dynamically find the project root (assumes this script is in src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("Loading Label Map...")
json_path = os.path.join(project_root, 'data', 'sign_to_prediction_index_map.json')
with open(json_path, 'r') as f:
    label_map = json.load(f)
inverse_map = {v: k for k, v in label_map.items()}

print("Loading Model Architecture and Weights...")
model = get_model()
weights_path = os.path.join(project_root, 'models', 'v3_fold0_best.h5')
model.load_weights(weights_path)

print("Initializing Preprocessor...")
preprocessor = Preprocess(max_len=CFG.max_len)
print("Initialization Complete! Starting video feed...")


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip frame horizontally for a natural selfie-view
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # 1. Extract and format the 543 landmarks to match the Kaggle dataset structure
        frame_landmarks = np.zeros((543, 3))

        # Face (0 - 467)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                frame_landmarks[i] = [lm.x, lm.y, lm.z]

        # Left Hand (468 - 488)[cite: 1]
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                frame_landmarks[468 + i] = [lm.x, lm.y, lm.z]

        # Pose (489 - 521)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[489 + i] = [lm.x, lm.y, lm.z]

        # Right Hand (522 - 542)[cite: 1]
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                frame_landmarks[522 + i] = [lm.x, lm.y, lm.z]

        # Replace uncaptured landmarks (0.0) with NaN to match your tf_nan_mean logic
        frame_landmarks[frame_landmarks == 0.0] = np.nan

        # 2. Buffer Management
        sequence_buffer.append(frame_landmarks)

        # Keep buffer length exactly at CFG.max_len (64)[cite: 1]
        if len(sequence_buffer) > 64:
            sequence_buffer.pop(0)
        # 3. Predict when buffer is full
        if len(sequence_buffer) == 64:
            # Convert buffer to tensor shape (1, 64, 543, 3)
            input_tensor = tf.convert_to_tensor([sequence_buffer], dtype=tf.float32)

            # Pass through your notebook's Preprocess layer to get the (1, 64, 708) shape
            processed_features = preprocessor(input_tensor)

            # Predict
            predictions = model.predict(processed_features, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_sign = inverse_map.get(predicted_index, "Unknown")

            # Display prediction on screen
            cv2.putText(frame, f"Sign: {predicted_sign}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        else:
            # Display awaiting message ONLY when buffer is NOT full
            cv2.putText(frame, f"Filling Buffer: {len(sequence_buffer)}/64", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('ASL Real-Time Recognition', frame)

        # Break loop gracefully on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
# print("Available signs to test:", list(inverse_map.values()))