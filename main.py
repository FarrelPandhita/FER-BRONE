import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import os
import time
from collections import deque

# ==================== CONFIGURATION ====================
class Config:
    # Path Model (Pastikan file ini ada di folder models/)
    MODEL_PATH = 'models/fer_model_v1.2_fusion_colab.pth'
    
    # Haarcascade
    CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(CASCADE_PATH):
        # Fallback ke sistem jika file lokal tidak ada
        CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    # Settings Model
    NUM_CLASSES = 5 
    EMOTION_LABELS = ['Upset', 'Shocked', 'Happy', 'Sad', 'Neutral']
    INPUT_SIZE = 112
    
    # Settings UI & Logika
    WINDOW_SIZE = 30            # Buffer 30 frame (~1 detik pada 30FPS) untuk smoothing stabil
    CONFIDENCE_THRESHOLD = 0.50 
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== MODEL DEFINITION ====================
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5, architecture='resnet34'):
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


# ==================== UTILS (SMOOTHING) ====================
class TemporalAveraging:
    def __init__(self, window_size=15, confidence_threshold=0.5):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)
    
    def add_prediction(self, probabilities):
        self.buffer.append(probabilities)
    
    def get_averaged_emotion(self):
        # Butuh minimal data untuk mulai merata-rata
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


# ==================== MAIN LIVE SYSTEM ====================
class LiveSystem:
    def __init__(self):
        print(f"--- FER V2 SYSTEM (Dashboard UI) ---")
        print(f"Device: {Config.DEVICE}")
        
        # 1. Load Model
        self.model = EmotionRecognitionModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        
        if os.path.exists(Config.MODEL_PATH):
            print(f"Memuat model dari: {Config.MODEL_PATH}")
            checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            print("✅ Model siap!")
        else:
            print(f"❌ ERROR: Model tidak ditemukan di {Config.MODEL_PATH}")
            exit()

        # 2. Setup Camera
        self.face_cascade = cv2.CascadeClassifier(Config.CASCADE_PATH)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.temporal_avg = TemporalAveraging(Config.WINDOW_SIZE, Config.CONFIDENCE_THRESHOLD)
        self.no_face_counter = 0
        
        # Variabel FPS
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0

    def draw_static_ui(self, frame):
        """Menggambar elemen UI yang selalu ada (Header)"""
        height, width, _ = frame.shape
        
        # Header Background
        cv2.rectangle(frame, (width//2 - 120, 0), (width//2 + 120, 40), (20, 20, 20), -1)
        
        # Header Text
        cv2.putText(frame, "FER V2 ", (width//2 - 90, 28),  
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_dashboard(self, frame, x, y, w, h, instant_emo, instant_conf, smooth_emo, smooth_conf):
        # --- LOGIKA WARNA (UPDATED) ---
        # Green: Neutral, Happy
        # Red: Upset (Angry), Sad, Shocked
        # Orange: Uncertain, Collecting
        
        white = (255, 255, 255)
        orange = (0, 165, 255) # BGR
        green = (0, 255, 0)
        red = (0, 0, 255)
        
        if smooth_emo in ["Neutral", "Happy"]:
            status_color = green
        elif smooth_emo in ["Upset", "Sad", "Shocked"]:
            status_color = red
        else:
            status_color = orange

        # --- 1. Info Panel (Pojok Kiri Atas - Dashboard Style) ---
        overlay = frame.copy()
        panel_w, panel_h = 420, 160
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        start_x = 25
        start_y = 45
        spacing = 35

        # Baris 1: Instant
        cv2.putText(frame, f"Instant: {instant_emo} ({instant_conf:.2f})", 
                    (start_x, start_y), font, scale, white, 1, cv2.LINE_AA)

        # Baris 2: Smoothed
        cv2.putText(frame, f"Smoothed: {smooth_emo} ({smooth_conf:.2f})", 
                    (start_x, start_y + spacing), font, scale, status_color, 2, cv2.LINE_AA)

        # Baris 3: Buffer Capacity
        buffer_len = len(self.temporal_avg.buffer)
        buffer_pct = int((buffer_len / Config.WINDOW_SIZE) * 100)
        cv2.putText(frame, f"Buffer: {buffer_len}/{Config.WINDOW_SIZE} ({buffer_pct}%)", 
                    (start_x, start_y + spacing * 2), font, scale, white, 1, cv2.LINE_AA)

        # Baris 4: FPS Counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                    (start_x, start_y + spacing * 3), font, scale, white, 1, cv2.LINE_AA)

        # --- 2. Face Box & Label ---
        box_color = status_color
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        
        # Label di atas kepala
        label_text = f"{smooth_emo}"
        (tw, th), _ = cv2.getTextSize(label_text, font, 0.7, 2)
        
        # Background label
        cv2.rectangle(frame, (x, y - 35), (x + tw + 10, y), box_color, -1)
        # Text label (Hitam agar kontras dengan background warna)
        text_color = (0, 0, 0)
        cv2.putText(frame, label_text, (x + 5, y - 10), font, 0.7, text_color, 2)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Gagal membuka webcam.")
            return

        print("🎥 Webcam dimulai. Tekan 'q' untuk keluar.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- Hitung FPS ---
            self.new_frame_time = time.time()
            diff = self.new_frame_time - self.prev_frame_time
            self.fps = 1 / diff if diff > 0 else 0
            self.prev_frame_time = self.new_frame_time
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            # Gambar Header
            frame = self.draw_static_ui(frame)

            if len(faces) > 0:
                self.no_face_counter = 0
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    # 1. Konversi BGR ke RGB
                    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    
                    # 2. Inference
                    input_tensor = self.transform(face_roi_rgb).unsqueeze(0).to(Config.DEVICE)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                    
                    # 3. Dapatkan Data Instant
                    instant_idx = np.argmax(probs)
                    instant_emo = Config.EMOTION_LABELS[instant_idx]
                    instant_conf = np.max(probs)
                    
                    # 4. Update Buffer & Dapatkan Data Smoothed
                    self.temporal_avg.add_prediction(probs)
                    smooth_emo, smooth_conf = self.temporal_avg.get_averaged_emotion()
                    
                    # 5. Gambar Dashboard dengan Warna Dinamis
                    frame = self.draw_dashboard(frame, x, y, w, h, 
                                              instant_emo, instant_conf, 
                                              smooth_emo, smooth_conf)
                    
                except Exception as e:
                    print(f"Error: {e}")
            else:
                self.no_face_counter += 1
                if self.no_face_counter > 10: 
                    self.temporal_avg.reset()
                
                # Tetap tampilkan FPS saat idle
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (25, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.putText(frame, "Mencari Wajah...", (25, 115), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            cv2.imshow('FER V2 - Dashboard Mode', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LiveSystem()
    app.run()