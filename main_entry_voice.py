"""
Voice-Triggered Entry Point - FER Robot System
===============================================
Menunggu trigger suara "Halo Brone" untuk memulai sistem.

Jalankan: python main_entry_voice.py

Requirements:
- SpeechRecognition
- PyAudio (untuk microphone)

Install:
  pip install SpeechRecognition pyaudio

Note: Untuk Jetson, mungkin perlu install portaudio:
  sudo apt-get install portaudio19-dev python3-pyaudio
"""
import subprocess
import sys
import os
import time

# ==================== CONFIGURATION ====================
class Config:
    # Trigger phrase (case insensitive)
    TRIGGER_PHRASE = "halo brone"
    ALTERNATIVE_TRIGGERS = ["hello brone", "hai brone", "hey brone", "hallo brone"]
    
    # Main entry script
    MAIN_ENTRY_SCRIPT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'main_entry_fer.py'
    )
    
    # Speech recognition settings
    ENERGY_THRESHOLD = 300  # Microphone sensitivity
    PAUSE_THRESHOLD = 0.8   # Silence duration to consider phrase complete
    LISTEN_TIMEOUT = None   # None = listen forever

# ==================== VOICE TRIGGER ====================
class VoiceTrigger:
    """Listen for voice trigger phrase"""
    
    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self.available = False
        
        self._setup_speech()
    
    def _setup_speech(self):
        """Initialize speech recognition"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = Config.ENERGY_THRESHOLD
            self.recognizer.pause_threshold = Config.PAUSE_THRESHOLD
            
            self.microphone = sr.Microphone()
            self.available = True
            print("✅ Speech recognition ready")
            
            # Calibrate for ambient noise
            print("🎤 Calibrating microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Microphone calibrated")
            
        except ImportError:
            print("❌ SpeechRecognition not installed!")
            print("   Install with: pip install SpeechRecognition pyaudio")
            self.available = False
        except Exception as e:
            print(f"❌ Microphone error: {e}")
            self.available = False
    
    def listen_for_trigger(self) -> bool:
        """Listen for trigger phrase"""
        if not self.available:
            return False
        
        import speech_recognition as sr
        
        try:
            with self.microphone as source:
                print("\n🎤 Listening for 'Halo Brone'...")
                audio = self.recognizer.listen(
                    source,
                    timeout=Config.LISTEN_TIMEOUT,
                    phrase_time_limit=5
                )
            
            # Try to recognize speech
            try:
                text = self.recognizer.recognize_google(audio, language="id-ID").lower()
                print(f"   Heard: \"{text}\"")
                
                # Check for trigger phrase
                if Config.TRIGGER_PHRASE in text:
                    return True
                
                # Check alternatives
                for alt in Config.ALTERNATIVE_TRIGGERS:
                    if alt in text:
                        return True
                
                return False
                
            except sr.UnknownValueError:
                # Speech not understood
                return False
            except sr.RequestError as e:
                print(f"⚠️ Speech API error: {e}")
                return False
                
        except Exception as e:
            print(f"⚠️ Listen error: {e}")
            return False
    
    def wait_for_trigger(self) -> bool:
        """Wait until trigger phrase is heard"""
        print("\n" + "=" * 50)
        print("👂 Waiting for voice trigger...")
        print(f"   Say: \"{Config.TRIGGER_PHRASE.title()}\"")
        print("=" * 50)
        
        while True:
            if self.listen_for_trigger():
                print("\n🎉 Trigger detected!")
                return True
            time.sleep(0.1)

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("🎤 BRONE - Voice Triggered Mode")
    print("=" * 60)
    print(f"Trigger phrase: \"{Config.TRIGGER_PHRASE.title()}\"")
    print("=" * 60)
    
    # Initialize voice trigger
    voice = VoiceTrigger()
    
    if not voice.available:
        print("\n⚠️ Voice recognition not available!")
        print("   Falling back to manual mode...")
        input("   Press Enter to start BRONE system...")
    else:
        # Wait for trigger
        voice.wait_for_trigger()
    
    # Launch main entry
    print("\n🚀 Launching BRONE system...")
    
    try:
        subprocess.run([sys.executable, Config.MAIN_ENTRY_SCRIPT])
    except KeyboardInterrupt:
        print("\n⛔ Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
