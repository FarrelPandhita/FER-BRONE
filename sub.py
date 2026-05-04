import paho.mqtt.client as mqtt
import json
import time
import os
import sys
import subprocess
import threading

# ==================== CONFIGURATION ====================
class Config:
    MQTT_BROKER = "localhost" 
    MQTT_TOPIC = "robot/expression"
    
    # Mapping Emosi -> Script Python
    EMOTION_SCRIPTS = {
        "Happy":   "emotions/brone_happier.py",
        "Neutral": "emotions/brone_happy.py",  # Neutral & Idle state
        "Idle":    "emotions/brone_happy.py",  # Idle state
        "Upset":   "emotions/brone_sad.py",    # Upset state
        "Sad":     "emotions/brone_cry.py",    # Cry state
        "Shocked": "emotions/brone_shock.py"
    }

# ==================== PROCESS MANAGER ====================
class ScriptManager:
    def __init__(self):
        self.current_process = None
        self.current_script = None
        self.lock = threading.Lock()

    def run_script(self, script_name):
        with self.lock:
            # Jika script yang diminta sama dengan yang sedang jalan, abaikan
            if self.current_script == script_name and self.current_process and self.current_process.poll() is None:
                return

            # Matikan proses sebelumnya jika ada
            if self.current_process:
                print(f"🛑 Menghentikan: {self.current_script}")
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()

            # Jalankan script baru
            target_path = os.path.join(os.path.dirname(__file__), script_name)
            if os.path.exists(target_path):
                print(f"🚀 Menjalankan: {script_name}")
                # Menggunakan sys.executable untuk memastikan pakai python env yang sama
                self.current_process = subprocess.Popen([sys.executable, target_path])
                self.current_script = script_name
            else:
                print(f"❌ Script tidak ditemukan: {target_path}")

# ==================== MQTT CLIENT ====================
class Subscriber:
    def __init__(self):
        self.manager = ScriptManager()
        self.last_update_time = time.time()
        self.current_emotion = "Idle"
        
        # Jalankan Idle state pertama kali
        self.manager.run_script(Config.EMOTION_SCRIPTS["Idle"])

        # Setup MQTT
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        try:
            self.client.connect(Config.MQTT_BROKER, 1883, 60)
            self.client.loop_start()
            print(f"✅ Subscriber Ready. Listening to {Config.MQTT_TOPIC}")
        except Exception as e:
            print(f"❌ Gagal konek MQTT: {e}")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        client.subscribe(Config.MQTT_TOPIC)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            new_emotion = payload.get("emotion", "Neutral")
            
            self.last_update_time = time.time()
            
            # Jika emosi berubah, ganti script
            if new_emotion != self.current_emotion:
                if new_emotion in Config.EMOTION_SCRIPTS:
                    self.current_emotion = new_emotion
                    script = Config.EMOTION_SCRIPTS[new_emotion]
                    self.manager.run_script(script)
                    
        except Exception as e:
            print(f"Error: {e}")

    def run(self):
        try:
            while True:
                # Logic Timeout: Jika tidak ada data > 5 detik -> Kembali ke Idle
                if time.time() - self.last_update_time > 5.0:
                    if self.current_emotion != "Idle":
                        print("💤 Timeout: Masuk mode Idle")
                        self.current_emotion = "Idle"
                        self.manager.run_script(Config.EMOTION_SCRIPTS["Idle"])
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nMematikan subscriber...")
            if self.manager.current_process:
                self.manager.current_process.kill()
            self.client.loop_stop()

if __name__ == "__main__":
    app = Subscriber()
    app.run()