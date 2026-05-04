"""
╔══════════════════════════════════════════════════════════════════╗
║   FER Publisher — Multi-Backend (PyTorch / ONNX / TensorRT)     ║
║                                                                  ║
║   Auto-detects the best available backend:                       ║
║     1. TensorRT .engine  (fastest — Jetson optimized FP16)       ║
║     2. ONNX .onnx        (fast    — CUDA/CPU via onnxruntime)    ║
║     3. PyTorch .pth      (default — CUDA/CPU via torch)          ║
║                                                                  ║
║   Usage:                                                         ║
║     python3 pub.py                    ← auto-detect backend      ║
║     python3 pub.py --backend onnx     ← force ONNX               ║
║     python3 pub.py --backend pytorch  ← force PyTorch             ║
║     python3 pub.py --backend tensorrt ← force TensorRT            ║
║     python3 pub.py --headless         ← tanpa GUI (production)    ║
║                                                                  ║
║   Model files yang dicari (sesuaikan di Config):                  ║
║     .engine → fer_resnet34_v1.2_fp16.engine                      ║
║     .onnx   → fer_resnet34_v1.2.onnx                             ║
║     .pth    → models/fer_model_v1.2_fusion_colab.pth             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import argparse
import platform
import paho.mqtt.client as mqtt
from collections import deque

# ==================== CONFIGURATION ====================
class Config:
    # ── Model Paths (dicek berurutan: engine → onnx → pth) ──
    TENSORRT_PATH = 'fer_resnet34_v1.2_fp16.engine'
    ONNX_PATH     = 'fer_resnet34_v1.2.onnx'
    PTH_PATH      = 'models/fer_model_v1.2_fusion_colab.pth'

    # Haarcascade
    CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(CASCADE_PATH):
        CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    # Model Settings
    NUM_CLASSES = 5
    EMOTION_LABELS = ['Upset', 'Shocked', 'Happy', 'Sad', 'Neutral']
    INPUT_SIZE = 112

    # Smoothing & Logic
    WINDOW_SIZE = 15
    CONFIDENCE_THRESHOLD = 0.50

    # MQTT Settings
    MQTT_BROKER = "localhost"
    MQTT_PORT = 1883
    MQTT_TOPIC = "robot/fer_emotion"

    # Mapping FER emotion labels → FaceRenderer expression states
    EMOTION_TO_EXPRESSION = {
        "Happy":   "happier",
        "Neutral": "idle",
        "Sad":     "sad",
        "Shocked": "shock",
        "Upset":   "cry",
    }

    # Camera settings (640×480 untuk Jetson, hemat resource)
    CAMERA_WIDTH  = 640
    CAMERA_HEIGHT = 480


# ==================== SOFTMAX HELPER ====================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# ==================== TEMPORAL AVERAGING ====================
class TemporalAveraging:
    def __init__(self, window_size=15, confidence_threshold=0.5):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)

    def add_prediction(self, probabilities):
        self.buffer.append(probabilities)

    def get_averaged_emotion(self):
        if len(self.buffer) < 2:
            return "Collecting...", 0.0
        avg_probs = np.mean(self.buffer, axis=0)
        idx  = np.argmax(avg_probs)
        conf = float(np.max(avg_probs))
        if conf >= self.confidence_threshold:
            return Config.EMOTION_LABELS[idx], conf
        return "UNCERTAIN", conf

    def reset(self):
        self.buffer.clear()


# ==================== MODEL BACKENDS ====================

class PyTorchBackend:
    """Backend menggunakan PyTorch (.pth) — untuk laptop NVIDIA / development."""

    def __init__(self, model_path):
        import torch
        import torch.nn as nn
        from torchvision import transforms, models

        self.torch = torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build model architecture
        model = models.resnet34(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, Config.NUM_CLASSES)
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        model.to(self.device)
        self.model = model

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"  ✓ PyTorch backend loaded on {self.device}")

    def predict(self, face_bgr):
        """Return probability array [num_classes]."""
        roi_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(input_tensor)
            probs = self.torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        return probs


class ONNXBackend:
    """Backend menggunakan ONNX Runtime (.onnx) — untuk Jetson (CUDA EP)."""

    def __init__(self, model_path):
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Prioritas: CUDA → CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        active_provider = self.session.get_providers()[0]
        print(f"  ✓ ONNX backend loaded ({active_provider})")

    def predict(self, face_bgr):
        """Return probability array [num_classes]."""
        img = cv2.resize(face_bgr, (Config.INPUT_SIZE, Config.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)          # HWC → CHW
        img  = np.expand_dims(img, axis=0)      # add batch dim

        logits = self.session.run([self.output_name], {self.input_name: img})[0]
        probs  = softmax(logits)[0]
        return probs


class TensorRTBackend:
    """Backend menggunakan TensorRT engine (.engine) — tercepat di Jetson."""

    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — auto-init CUDA context

        self.cuda = cuda
        self.trt  = trt

        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs  = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size  = int(np.prod(shape)) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name, 'mem': device_mem,
                    'shape': shape, 'dtype': dtype
                })
            else:
                self.outputs.append({
                    'name': name, 'mem': device_mem,
                    'shape': shape, 'dtype': dtype
                })

        print(f"  ✓ TensorRT backend loaded (FP16 engine)")

    def predict(self, face_bgr):
        """Return probability array [num_classes]."""
        img = cv2.resize(face_bgr, (Config.INPUT_SIZE, Config.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)
        img  = np.expand_dims(img, axis=0).astype(np.float32)
        img  = np.ascontiguousarray(img)

        # Copy input to GPU
        self.cuda.memcpy_htod_async(self.inputs[0]['mem'], img, self.stream)

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['mem']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['mem']))

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output from GPU
        output = np.empty(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
        self.cuda.memcpy_dtoh_async(output, self.outputs[0]['mem'], self.stream)
        self.stream.synchronize()

        probs = softmax(output)[0]
        return probs


# ==================== BACKEND AUTO-DETECTION ====================

def detect_backend(force=None):
    """
    Auto-detect backend terbaik berdasarkan file yang tersedia.
    Prioritas: TensorRT → ONNX → PyTorch
    """
    if force:
        if force == 'tensorrt':
            if not os.path.exists(Config.TENSORRT_PATH):
                print(f"  ✗ TensorRT engine tidak ditemukan: {Config.TENSORRT_PATH}")
                sys.exit(1)
            return TensorRTBackend(Config.TENSORRT_PATH)
        elif force == 'onnx':
            if not os.path.exists(Config.ONNX_PATH):
                print(f"  ✗ ONNX model tidak ditemukan: {Config.ONNX_PATH}")
                sys.exit(1)
            return ONNXBackend(Config.ONNX_PATH)
        elif force == 'pytorch':
            if not os.path.exists(Config.PTH_PATH):
                print(f"  ✗ PyTorch model tidak ditemukan: {Config.PTH_PATH}")
                sys.exit(1)
            return PyTorchBackend(Config.PTH_PATH)

    # Auto-detect: coba dari yang tercepat
    if os.path.exists(Config.TENSORRT_PATH):
        try:
            return TensorRTBackend(Config.TENSORRT_PATH)
        except Exception as e:
            print(f"  ⚠ TensorRT gagal ({e}), fallback ke ONNX...")

    if os.path.exists(Config.ONNX_PATH):
        try:
            return ONNXBackend(Config.ONNX_PATH)
        except Exception as e:
            print(f"  ⚠ ONNX gagal ({e}), fallback ke PyTorch...")

    if os.path.exists(Config.PTH_PATH):
        return PyTorchBackend(Config.PTH_PATH)

    print("  ✗ Tidak ada model ditemukan! Letakkan salah satu file berikut:")
    print(f"    .engine  → {Config.TENSORRT_PATH}")
    print(f"    .onnx    → {Config.ONNX_PATH}")
    print(f"    .pth     → {Config.PTH_PATH}")
    sys.exit(1)


# ==================== MAIN SYSTEM ====================

class FERPublisher:
    def __init__(self, backend, headless=False):
        self.backend  = backend
        self.headless = headless

        # MQTT
        self._setup_mqtt()

        # Face detection
        self.face_cascade = cv2.CascadeClassifier(Config.CASCADE_PATH)
        self.temporal_avg = TemporalAveraging(Config.WINDOW_SIZE,
                                              Config.CONFIDENCE_THRESHOLD)
        self.no_face_counter = 0
        self.prev_frame_time = 0
        self.fps = 0

    def _setup_mqtt(self):
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.connect(Config.MQTT_BROKER, Config.MQTT_PORT, 60)
            self.client.loop_start()
            print("  ✓ MQTT Connected")
        except Exception as e:
            print(f"  ⚠ MQTT Error: {e}")

    def publish_emotion(self, emotion, confidence):
        if emotion in ("UNCERTAIN", "Collecting...", "Analyzing..."):
            return
        expression = Config.EMOTION_TO_EXPRESSION.get(emotion, "idle")
        payload = {
            "timestamp":  time.time(),
            "emotion":    emotion,
            "expression": expression,
            "confidence": round(float(confidence), 2)
        }
        try:
            self.client.publish(Config.MQTT_TOPIC, json.dumps(payload))
        except Exception:
            pass

    def _open_camera(self):
        """Open camera — gunakan GStreamer di Jetson (aarch64) untuk HW-accelerated."""
        if platform.machine() == 'aarch64':
            # Jetson: GStreamer hardware-accelerated pipeline
            gst = (
                f'v4l2src device=/dev/video0 ! '
                f'video/x-raw, width={Config.CAMERA_WIDTH}, height={Config.CAMERA_HEIGHT}, '
                f'framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
            )
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"  ✓ Camera opened (GStreamer {Config.CAMERA_WIDTH}×{Config.CAMERA_HEIGHT})")
                return cap
            print("  ⚠ GStreamer gagal, fallback ke V4L2...")

        # Fallback: standard V4L2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        if cap.isOpened():
            print(f"  ✓ Camera opened (V4L2 {Config.CAMERA_WIDTH}×{Config.CAMERA_HEIGHT})")
        return cap

    def run(self):
        cap = self._open_camera()
        if not cap.isOpened():
            print("  ✗ Gagal membuka webcam!")
            return

        print("  ✓ Siap. Tekan 'q' untuk keluar.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS
            curr = time.time()
            dt = curr - self.prev_frame_time
            self.fps = 1.0 / dt if dt > 0 else 0
            self.prev_frame_time = curr

            frame_flip = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            if len(faces) > 0:
                self.no_face_counter = 0
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face_roi = frame_flip[y:y+h, x:x+w]

                try:
                    probs = self.backend.predict(face_roi)

                    self.temporal_avg.add_prediction(probs)
                    emo, conf = self.temporal_avg.get_averaged_emotion()

                    self.publish_emotion(emo, conf)

                    if not self.headless:
                        # Draw bounding box + label
                        color = (0, 255, 0)
                        cv2.rectangle(frame_flip, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame_flip, f"{emo} ({conf:.2f})", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except Exception as e:
                    print(f"  [Error inference] {e}")
            else:
                self.no_face_counter += 1
                if self.no_face_counter > 10:
                    self.temporal_avg.reset()

            if not self.headless:
                cv2.putText(frame_flip, f"FPS: {self.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('FER Publisher', frame_flip)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Headless: print FPS periodically
                if int(curr) % 5 == 0 and int(curr) != getattr(self, '_last_log', 0):
                    self._last_log = int(curr)
                    emo, conf = self.temporal_avg.get_averaged_emotion()
                    print(f"  [FPS: {self.fps:.1f}] {emo} ({conf:.2f})")

        cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        self.client.loop_stop()
        self.client.disconnect()
        print("  FER Publisher selesai.")


# ==================== ENTRY POINT ====================

def main():
    parser = argparse.ArgumentParser(
        description="FER Publisher — Multi-Backend (PyTorch / ONNX / TensorRT)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Contoh penggunaan:
  python3 pub.py                          # auto-detect backend terbaik
  python3 pub.py --backend onnx           # paksa ONNX
  python3 pub.py --backend pytorch        # paksa PyTorch (.pth)
  python3 pub.py --backend tensorrt       # paksa TensorRT (.engine)
  python3 pub.py --headless               # tanpa GUI (Jetson production)
  python3 pub.py --backend onnx --headless

Model files yang dicari (di folder yang sama dengan pub.py):
  TensorRT : fer_resnet34_v1.2_fp16.engine   ← tercepat (Jetson)
  ONNX     : fer_resnet34_v1.2.onnx          ← cepat (Jetson CUDA EP)
  PyTorch  : models/fer_model_v1.2_fusion_colab.pth  ← default (laptop)

Cara membuat file .onnx dan .engine:
  1. python3 converter.py                               # .pth → .onnx
  2. /usr/src/tensorrt/bin/trtexec \\
       --onnx=fer_resnet34_v1.2.onnx \\
       --saveEngine=fer_resnet34_v1.2_fp16.engine \\
       --fp16 --workspace=1024                          # .onnx → .engine
        """
    )
    parser.add_argument('--backend', choices=['pytorch', 'onnx', 'tensorrt'],
                        default=None, help='Force backend (default: auto-detect)')
    parser.add_argument('--headless', action='store_true',
                        help='Tanpa GUI window (untuk Jetson production)')
    args = parser.parse_args()

    print("=" * 55)
    print("  FER Publisher — Multi-Backend")
    print("=" * 55)

    # Detect & load backend
    print(f"\n  Mencari model...")
    backend = detect_backend(force=args.backend)

    # Run
    app = FERPublisher(backend, headless=args.headless)
    app.run()


if __name__ == "__main__":
    main()