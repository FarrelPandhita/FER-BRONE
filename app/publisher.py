"""
FER Publisher - Face Emotion Recognition
Mendeteksi emosi wajah dari webcam dan publish ke MQTT.

Jalankan: python app/publisher.py

Requirements:
- PyTorch
- OpenCV
- paho-mqtt
- torchvision
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt
from collections import deque

# ==================== CONFIGURATION ====================
class Config:
    # Paths (relative to FER-V2 root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fer_model_v1.2_fusion_colab.pth')
    
    # Haarcascade - Relative path from app folder
    CASCADE_PATH = '../haarcascades/haarcascade_frontalface_default.xml'

    # Model Settings
    NUM_CLASSES = 5
    EMOTION_LABELS = ['Upset', 'Shocked', 'Happy', 'Sad', 'Neutral']
    INPUT_SIZE = 112

    # Temporal Averaging
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

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== MODEL DEFINITION ====================
class EmotionRecognitionModel(nn.Module):
    """ResNet34 with custom head for 5-class emotion recognition"""
    
    def __init__(self, num_classes=5):
        super(EmotionRecognitionModel, self).__init__()
        self.backbone = models.resnet34(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ==================== TEMPORAL AVERAGING ====================
class TemporalAveraging:
    """Buffer predictions over multiple frames for stability"""
    
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
        idx = np.argmax(avg_probs)
        conf = np.max(avg_probs)
        
        if conf >= self.confidence_threshold:
            return Config.EMOTION_LABELS[idx], conf
        else:
            return "UNCERTAIN", conf

    def reset(self):
        self.buffer.clear()

# ==================== PUBLISHER SYSTEM ====================
class FERPublisher:
    """Main Face Emotion Recognition Publisher"""
    
    def __init__(self):
        print("=" * 50)
        print("🎯 FER Publisher (PyTorch + MQTT)")
        print(f"   Device: {Config.DEVICE}")
        print("=" * 50)

        # 1. Setup MQTT
        self.setup_mqtt()

        # 2. Load Model
        self.load_model()

        # 3. Setup Camera & Preprocessing
        self.setup_camera()

        # 4. Initialize Temporal Averaging
        self.temporal_avg = TemporalAveraging(
            Config.WINDOW_SIZE,
            Config.CONFIDENCE_THRESHOLD
        )
        
        # Tracking
        self.no_face_counter = 0
        self.prev_frame_time = 0
        self.fps = 0

    def setup_mqtt(self):
        """Initialize MQTT client"""
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_connect = self.on_mqtt_connect
            self.client.connect(Config.MQTT_BROKER, Config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f"🔗 MQTT: Connecting to {Config.MQTT_BROKER}:{Config.MQTT_PORT}")
        except Exception as e:
            print(f"⚠️ MQTT Error: {e}")
            self.client = None

    def on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"✅ MQTT Connected! Topic: {Config.MQTT_TOPIC}")
        else:
            print(f"❌ MQTT Connection failed: {rc}")

    def load_model(self):
        """Load PyTorch model"""
        self.model = EmotionRecognitionModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        
        if os.path.exists(Config.MODEL_PATH):
            print(f"📦 Loading model: {Config.MODEL_PATH}")
            checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print("✅ Model loaded successfully!")
        else:
            print(f"❌ Model not found: {Config.MODEL_PATH}")
            print("   Please ensure the model file exists.")
            exit(1)

    def setup_camera(self):
        """Setup camera and preprocessing"""
        self.face_cascade = cv2.CascadeClassifier(Config.CASCADE_PATH)
        
        if self.face_cascade.empty():
            print(f"❌ Haarcascade not found: {Config.CASCADE_PATH}")
            exit(1)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Camera preprocessing ready")

    def publish_emotion(self, emotion: str, confidence: float):
        """Publish emotion to MQTT"""
        if emotion in ["UNCERTAIN", "Collecting...", "Analyzing..."]:
            return
        
        if self.client is None:
            return
        
        expression = Config.EMOTION_TO_EXPRESSION.get(emotion, "idle")
        payload = {
            "timestamp": time.time(),
            "emotion": emotion,
            "expression": expression,
            "confidence": round(float(confidence), 2)
        }
        
        try:
            self.client.publish(Config.MQTT_TOPIC, json.dumps(payload))
        except Exception as e:
            print(f"⚠️ Publish error: {e}")

    def draw_ui(self, frame, x, y, w, h, emotion, confidence):
        """Draw bounding box and label on frame"""
        # Green box around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Emotion label
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame

    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        print("\n" + "=" * 50)
        print("🚀 FER Publisher Running!")
        print("   Press Q to quit")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate FPS
                current_time = time.time()
                if (current_time - self.prev_frame_time) > 0:
                    self.fps = 1 / (current_time - self.prev_frame_time)
                self.prev_frame_time = current_time

                # Flip horizontally (mirror)
                frame_flip = cv2.flip(frame, 1)
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(60, 60)
                )

                if len(faces) > 0:
                    self.no_face_counter = 0
                    
                    # Get largest face
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    face_roi = frame_flip[y:y + h, x:x + w]

                    try:
                        # Preprocess
                        roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        input_tensor = self.transform(roi_rgb).unsqueeze(0).to(Config.DEVICE)

                        # Inference
                        with torch.no_grad():
                            output = self.model(input_tensor)
                            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

                        # Temporal averaging
                        self.temporal_avg.add_prediction(probs)
                        emotion, confidence = self.temporal_avg.get_averaged_emotion()

                        # Publish
                        self.publish_emotion(emotion, confidence)
                        
                        # Draw UI
                        self.draw_ui(frame_flip, x, y, w, h, emotion, confidence)

                    except Exception as e:
                        print(f"⚠️ Processing error: {e}")
                else:
                    self.no_face_counter += 1
                    if self.no_face_counter > 10:
                        self.temporal_avg.reset()

                # Draw FPS
                cv2.putText(frame_flip, f"FPS: {self.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # MQTT status
                mqtt_status = "MQTT: Connected" if self.client else "MQTT: Offline"
                cv2.putText(frame_flip, mqtt_status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # Show frame
                cv2.imshow('FER Publisher', frame_flip)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
            print("👋 Publisher stopped")

# ==================== MAIN ====================
if __name__ == "__main__":
    app = FERPublisher()
    app.run()
