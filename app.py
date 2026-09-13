import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
from collections import deque
import av
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


# ==========================================
# 1. CONSTANTS & CONFIGURATION
# ==========================================
class CFG:
    seed = 42
    max_len = 64
    batch_size = 32
    epoch = 150
    dim = 192


NUM_CLASSES = 250
PAD = -100.

NOSE = [1, 2, 98, 327]
LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
CHANNELS = 6 * len(POINT_LANDMARKS)


# ==========================================
# 2. CUSTOM TENSORFLOW LAYERS & MODEL
# ==========================================
def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis,
                         keepdims=keepdims) / tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)


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
        x = inputs[None, ...] if tf.rank(inputs) == 3 else inputs
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        x = (x - mean) / std
        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]
        dx = tf.cond(tf.shape(x)[1] > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
                     lambda: tf.zeros_like(x))
        dx2 = tf.cond(tf.shape(x)[1] > 2, lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
                      lambda: tf.zeros_like(x))
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
        return inputs * tf.nn.sigmoid(tf.squeeze(self.conv(tf.expand_dims(nn, -1)), -1))[:, None, :]


class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        super().build(input_shape)
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                          trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < self.start_step, lambda: inputs,
                    lambda: self.dropout(inputs, training=training))
        if training: self._train_counter.assign_add(1)
        return x


class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0))
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(kernel_size, strides=1, dilation_rate=dilation_rate,
                                                       padding='valid', use_bias=False)
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
        if drop_rate > 0: x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
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
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(self.qkv(inputs)))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        attn = tf.keras.layers.Softmax(axis=-1)(tf.matmul(q, k, transpose_b=True) * self.scale,
                                                mask=mask[:, None, None, :] if mask is not None else None)
        return self.proj(
            tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(self.drop1(attn) @ v)))


def TransformerBlock(dim=256, expand=2):
    def apply(inputs):
        x = tf.keras.layers.Add()([inputs, tf.keras.layers.Dropout(0.2, noise_shape=(None, 1, 1))(
            MultiHeadSelfAttention(dim=dim, num_heads=4, dropout=0.2)(
                tf.keras.layers.BatchNormalization(momentum=0.95)(inputs)))])
        return tf.keras.layers.Add()([x, tf.keras.layers.Dropout(0.2, noise_shape=(None, 1, 1))(
            tf.keras.layers.Dense(dim, use_bias=False)(
                tf.keras.layers.Dense(dim * expand, use_bias=False, activation='swish')(
                    tf.keras.layers.BatchNormalization(momentum=0.95)(x))))])

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
# 3. MODEL & PIPELINE CACHING
# ==========================================
@st.cache_resource
def load_prediction_pipeline():
    with open('data/sign_to_prediction_index_map.json', 'r') as f:
        label_map = json.load(f)
    inverse_map = {v: k for k, v in label_map.items()}

    model = get_model()
    model.load_weights('models/v3_fold0_best.h5')
    preprocessor = Preprocess(max_len=CFG.max_len)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    return model, inverse_map, preprocessor, holistic


model, inverse_map, preprocessor, holistic = load_prediction_pipeline()

# ==========================================
# 4. STREAMLIT APP UI & WEBRTC HANDLER
# ==========================================
st.title("Real-Time ASL Recognition (Live Demo)")
st.write("Perform American Sign Language gestures in front of your camera to see real-time predictions.")

# Session state to hold sequence data and current prediction
if "sequence_buffer" not in st.session_state:
    st.session_state.sequence_buffer = deque(maxlen=CFG.max_len)
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = "Waiting for camera..."

prediction_placeholder = st.empty()


class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="rgb24")
        results = holistic.process(image)

        frame_landmarks = np.zeros((543, 3))
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                frame_landmarks[i] = [lm.x, lm.y, lm.z]
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                frame_landmarks[468 + i] = [lm.x, lm.y, lm.z]
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[489 + i] = [lm.x, lm.y, lm.z]
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                frame_landmarks[522 + i] = [lm.x, lm.y, lm.z]

        frame_landmarks[frame_landmarks == 0.0] = np.nan
        st.session_state.sequence_buffer.append(frame_landmarks)

        if len(st.session_state.sequence_buffer) == CFG.max_len:
            input_tensor = tf.convert_to_tensor([list(st.session_state.sequence_buffer)], dtype=tf.float32)
            processed_features = preprocessor(input_tensor)
            predictions = model.predict(processed_features, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_sign = inverse_map.get(predicted_index, "Unknown")
            st.session_state.current_prediction = f"Sign: {predicted_sign}"
        else:
            st.session_state.current_prediction = f"Filling Buffer: {len(st.session_state.sequence_buffer)}/64"

        return av.VideoFrame.from_ndarray(image, format="rgb24")


webrtc_streamer(
    key="asl-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

prediction_placeholder.markdown(f"### Prediction: **{st.session_state.current_prediction}**")