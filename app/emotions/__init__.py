# Emotion Modules Package
# Setiap modul berisi fungsi draw(screen) untuk menggambar wajah robot

from . import brone_happy
from . import brone_sad
from . import brone_shock
from . import brone_upset
from . import brone_neutral

EMOTION_MODULES = {
    "Happy": brone_happy,
    "Sad": brone_sad,
    "Shocked": brone_shock,
    "Upset": brone_upset,
    "Neutral": brone_happy,  # Use happy face for neutral
    "Idle": brone_happy,     # Use happy face for idle
}

def get_module(emotion_name: str):
    """Get emotion module by name, defaults to neutral if not found"""
    return EMOTION_MODULES.get(emotion_name, brone_neutral)
