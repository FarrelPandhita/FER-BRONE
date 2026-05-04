"""
Brone Shock - Robot Face Emotion Module
Wajah kaget dengan mata bulat besar dan mulut "O".
"""
import pygame
import random

# ==================== CONSTANTS ====================
WIDTH, HEIGHT = 800, 600

# Colors
BG_COLOR = (205, 215, 225)
BLACK = (0, 0, 0)
EYE_TOP = (80, 70, 150)
EYE_BOTTOM = (0, 0, 0)
HIGHLIGHT = (240, 245, 255)
MOUTH_DARK = (40, 40, 40)
TONGUE = (230, 130, 100)

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
    global _initialized, _last_blink_time
    if not _initialized:
        _last_blink_time = pygame.time.get_ticks()
        _initialized = True

# ==================== HELPER FUNCTIONS ====================

def _draw_eye_gradient(surface, rect):
    """Draw eye with gradient"""
    pygame.draw.ellipse(surface, BLACK, rect.inflate(8, 8))

    gradient_tiny = pygame.Surface((1, 2))
    gradient_tiny.fill(EYE_TOP, (0, 0, 1, 1))
    gradient_tiny.fill(EYE_BOTTOM, (0, 1, 1, 1))
    gradient_surf = pygame.transform.smoothscale(gradient_tiny, (rect.width, rect.height))

    eye_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.ellipse(eye_surf, (255, 255, 255), (0, 0, rect.width, rect.height))
    eye_surf.blit(gradient_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    surface.blit(eye_surf, rect.topleft)

    # Highlight
    glint_x = rect.left + 35
    glint_y = rect.top + 40
    pygame.draw.circle(surface, HIGHLIGHT, (glint_x, glint_y), 22)
    pygame.draw.circle(surface, EYE_TOP, (glint_x + 10, glint_y + 10), 10)

    small_glint_x = glint_x - 5
    small_glint_y = glint_y + 45
    pygame.draw.circle(surface, HIGHLIGHT, (small_glint_x, small_glint_y), 6)

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
    Draws shocked robot face with wide eyes and O-shaped mouth
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

    # 2. EYES (Wide shocked eyes)
    _draw_eye_gradient(screen, left_eye_rect)
    _draw_eye_gradient(screen, right_eye_rect)

    # 3. EYELIDS (Blink)
    _draw_eyelid(screen, left_eye_rect, _blink_progress)
    _draw_eyelid(screen, right_eye_rect, _blink_progress)

    # 4. MOUTH (O-shape that follows blink)
    base_mouth_w = 160
    base_mouth_h = 130

    # Squash & Stretch: mouth shrinks when blinking
    current_mouth_h = base_mouth_h * (1.0 - _blink_progress)
    current_mouth_w = base_mouth_w + (_blink_progress * 20)

    if current_mouth_h < 4:
        current_mouth_h = 4

    mouth_rect = pygame.Rect(0, 0, current_mouth_w, current_mouth_h)
    mouth_rect.center = (center_x, 400 + base_mouth_h // 2)

    if current_mouth_h > 5:
        # A. Dark mouth base
        pygame.draw.ellipse(screen, MOUTH_DARK, mouth_rect)

        # B. Tongue (clipped)
        clip_rect = pygame.Rect(mouth_rect.left, mouth_rect.centery, mouth_rect.width, mouth_rect.height // 2)
        screen.set_clip(clip_rect)
        pygame.draw.ellipse(screen, TONGUE, mouth_rect)
        screen.set_clip(None)

        # C. Outline
        pygame.draw.ellipse(screen, BLACK, mouth_rect, 6)
    else:
        # When fully blinked, just draw a line
        pygame.draw.line(screen, BLACK,
                         (mouth_rect.left, mouth_rect.centery),
                         (mouth_rect.right, mouth_rect.centery), 6)
