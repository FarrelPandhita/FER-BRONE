"""
Brone Happy - Robot Face Emotion Module
Wajah senang dengan mata berbinar, blush on, dan mulut tersenyum lebar.
"""
import pygame
import math
import random
import time

# ==================== CONSTANTS ====================
WIDTH, HEIGHT = 800, 600

# Colors
BG_COLOR = (205, 215, 225)
BLACK = (0, 0, 0)
EYE_COLOR = (45, 40, 90)
HIGHLIGHT = (255, 255, 255)
MOUTH_DARK = (40, 40, 40)
TONGUE = (230, 130, 100)
BLUSH_COLOR = (255, 180, 200)
EYE_TOP = (80, 70, 150)      # Ungu
EYE_BOTTOM = (0, 0, 0)       # Hitam

# ==================== BLINK STATE (Global) ====================
_blink_state = "idle"
_blink_progress = 0.0
_blink_speed = 0.15
_last_blink_time = 0
_next_blink_interval = random.randint(2000, 5000)
_initialized = False

def trigger_blink():
    """Trigger a blink for transition effect"""
    global _blink_state, _blink_progress
    _blink_state = "closing"
    _blink_progress = 0.0

def _init_blink():
    """Initialize blink state on first call"""
    global _initialized, _last_blink_time
    if not _initialized:
        _last_blink_time = pygame.time.get_ticks()
        _initialized = True

# ==================== HELPER FUNCTIONS ====================

def _draw_star(surface, color, x, y, size):
    """Draw a sparkle star"""
    half = size // 2
    inner = size // 5
    points = [
        (x, y - half), (x + inner, y - inner),
        (x + half, y), (x + inner, y + inner),
        (x, y + half), (x - inner, y + inner),
        (x - half, y), (x - inner, y - inner)
    ]
    pygame.draw.polygon(surface, color, points)

def _draw_eye_gradient_with_sparkles(surface, rect):
    """Draw eye with gradient and sparkle effect"""
    # 1. Black outline
    pygame.draw.ellipse(surface, BLACK, rect.inflate(8, 8))

    # 2. Gradient
    gradient_tiny = pygame.Surface((1, 2))
    gradient_tiny.fill(EYE_TOP, (0, 0, 1, 1))
    gradient_tiny.fill(EYE_BOTTOM, (0, 1, 1, 1))
    gradient_surf = pygame.transform.smoothscale(gradient_tiny, (rect.width, rect.height))

    eye_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.ellipse(eye_surf, (255, 255, 255), (0, 0, rect.width, rect.height))
    eye_surf.blit(gradient_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    surface.blit(eye_surf, rect.topleft)

    # 3. Sparkles
    glint_x = rect.left + 35
    glint_y = rect.top + 45
    _draw_star(surface, HIGHLIGHT, glint_x, glint_y, 50)
    pygame.draw.circle(surface, HIGHLIGHT, (glint_x + 25, glint_y + 25), 5)
    pygame.draw.circle(surface, (150, 150, 255), (glint_x - 15, glint_y + 15), 3)

def _draw_blush(surface, x, y):
    """Draw pink blush on cheeks"""
    w, h = 70, 45
    blush_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.ellipse(blush_surf, (*BLUSH_COLOR, 120), (0, 0, w, h))
    surface.blit(blush_surf, (x - w // 2, y - h // 2))

def _draw_eyelid(surface, rect, progress):
    """Draw eyelid for blink effect"""
    if progress <= 0:
        return

    lid_height = rect.height * progress
    cover_rect = pygame.Rect(rect.left - 5, rect.top - 5, rect.width + 10, lid_height + 5)
    pygame.draw.rect(surface, BG_COLOR, cover_rect)

    line_y = rect.top + lid_height
    if line_y > rect.bottom:
        line_y = rect.bottom

    pygame.draw.line(surface, BLACK, (rect.left - 5, line_y), (rect.right + 5, line_y), 6)

def _update_blink():
    """Update blink animation state"""
    global _blink_state, _blink_progress, _last_blink_time, _next_blink_interval
    
    current_time = pygame.time.get_ticks()

    if _blink_state == "closing":
        _blink_progress += _blink_speed
        if _blink_progress >= 1.0:
            _blink_progress = 1.0
            _blink_state = "opening"

    elif _blink_state == "opening":
        _blink_progress -= _blink_speed
        if _blink_progress <= 0.0:
            _blink_progress = 0.0
            _blink_state = "idle"
            _last_blink_time = current_time
            _next_blink_interval = random.randint(2000, 6000)

    elif _blink_state == "idle":
        if current_time - _last_blink_time > _next_blink_interval:
            _blink_state = "closing"

# ==================== MAIN DRAW FUNCTION ====================

def draw(screen):
    """
    Main draw function - called by subscriber.py
    Draws happy robot face with blinking animation
    """
    _init_blink()
    _update_blink()

    screen.fill(BG_COLOR)

    center_x = WIDTH // 2
    eye_y = 220
    eye_width = 110
    eye_height = 150
    dist_from_center = 140

    left_eye_rect = pygame.Rect(center_x - dist_from_center - eye_width, eye_y, eye_width, eye_height)
    right_eye_rect = pygame.Rect(center_x + dist_from_center, eye_y, eye_width, eye_height)

    # 1. CABLES
    elbow_y = left_eye_rect.top - 50
    points_kiri = [(-20, 60), (left_eye_rect.centerx, elbow_y), (left_eye_rect.centerx, left_eye_rect.top)]
    pygame.draw.lines(screen, BLACK, False, points_kiri, 4)

    points_kanan = [(WIDTH + 20, 60), (right_eye_rect.centerx, elbow_y), (right_eye_rect.centerx, right_eye_rect.top)]
    pygame.draw.lines(screen, BLACK, False, points_kanan, 4)

    points_tengah = [
        (left_eye_rect.right - 10, left_eye_rect.centery),
        (center_x, left_eye_rect.centery + 40),
        (right_eye_rect.left + 10, right_eye_rect.centery)
    ]
    pygame.draw.lines(screen, BLACK, False, points_tengah, 4)

    # 2. BLUSH
    _draw_blush(screen, left_eye_rect.centerx - 20, left_eye_rect.bottom + 20)
    _draw_blush(screen, right_eye_rect.centerx + 20, right_eye_rect.bottom + 20)

    # 3. EYES
    _draw_eye_gradient_with_sparkles(screen, left_eye_rect)
    _draw_eye_gradient_with_sparkles(screen, right_eye_rect)

    # 4. EYELIDS (Blink)
    _draw_eyelid(screen, left_eye_rect, _blink_progress)
    _draw_eyelid(screen, right_eye_rect, _blink_progress)

    # 5. MOUTH (Wide Smile with Tongue)
    mouth_w = 240
    mouth_top_y = 400
    curve_top_sag = 25
    curve_bottom_depth = 130

    mouth_points = []
    steps = 60

    for i in range(steps + 1):
        t = i / steps
        px = (center_x - mouth_w // 2) + (t * mouth_w)
        py = mouth_top_y + (curve_top_sag * 4 * t * (1 - t))
        mouth_points.append((px, py))

    bottom_points = []
    a = mouth_w / 2
    b = curve_bottom_depth

    for i in range(steps + 1):
        t = i / steps
        px = (center_x - mouth_w // 2) + (t * mouth_w)
        dx = px - center_x
        inside_sqrt = max(0, 1 - (dx / a) ** 2)
        offset_y = b * math.sqrt(inside_sqrt)
        py = mouth_top_y + offset_y
        bottom_points.append((px, py))

    mouth_points.extend(reversed(bottom_points))

    pygame.draw.polygon(screen, MOUTH_DARK, mouth_points)

    # Tongue
    mouth_mask = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(mouth_mask, (255, 255, 255, 255), mouth_points)

    tongue_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tongue_rect = pygame.Rect(center_x - mouth_w // 2 + 10, mouth_top_y + 50, mouth_w - 20, 110)
    pygame.draw.ellipse(tongue_surf, TONGUE, tongue_rect)

    mouth_mask.blit(tongue_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    screen.blit(mouth_mask, (0, 0))

    pygame.draw.polygon(screen, BLACK, mouth_points, 8)
    pygame.draw.aalines(screen, BLACK, True, mouth_points)
