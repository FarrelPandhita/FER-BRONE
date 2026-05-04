"""
Brone Sad - Robot Face Emotion Module
Wajah sedih dengan mata lesu dan mulut melengkung ke bawah.
"""
import pygame
import math
import random

# ==================== CONSTANTS ====================
WIDTH, HEIGHT = 800, 600

# Colors
BG_COLOR = (205, 215, 225)
BLACK = (0, 0, 0)
HIGHLIGHT = (240, 245, 255)
MOUTH_DARK = (40, 40, 40)
TONGUE = (230, 130, 100)
EYE_TOP = (80, 70, 150)
EYE_BOTTOM = (0, 0, 0)

# ==================== BLINK STATE (Global) ====================
_blink_state = "idle"
_blink_progress = 0.0
_blink_speed = 0.05  # Slower blink for sad expression
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

def _draw_sad_eyebrow(surface, eye_rect, is_left):
    """Draw droopy sad eyebrow above eye"""
    if is_left:
        start = (eye_rect.left - 10, eye_rect.top - 10)
        end = (eye_rect.right + 10, eye_rect.top - 30)
    else:
        start = (eye_rect.left - 10, eye_rect.top - 30)
        end = (eye_rect.right + 10, eye_rect.top - 10)
    
    pygame.draw.line(surface, BLACK, start, end, 8)

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
            _next_blink_interval = random.randint(3000, 7000)  # Slower blink interval

    elif _blink_state == "idle":
        if current_time - _last_blink_time > _next_blink_interval:
            _blink_state = "closing"

# ==================== MAIN DRAW FUNCTION ====================

def draw(screen):
    """
    Main draw function - called by subscriber.py
    Draws sad robot face with droopy eyes and frown
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
    pygame.draw.lines(screen, BLACK, False, [(-20, 60), (left_eye_rect.centerx, elbow_y), (left_eye_rect.centerx, left_eye_rect.top)], 4)
    pygame.draw.lines(screen, BLACK, False, [(WIDTH + 20, 60), (right_eye_rect.centerx, elbow_y), (right_eye_rect.centerx, right_eye_rect.top)], 4)
    pygame.draw.lines(screen, BLACK, False, [(left_eye_rect.right - 10, left_eye_rect.centery), (center_x, left_eye_rect.centery + 40), (right_eye_rect.left + 10, right_eye_rect.centery)], 4)

    # 2. SAD EYEBROWS
    _draw_sad_eyebrow(screen, left_eye_rect, is_left=True)
    _draw_sad_eyebrow(screen, right_eye_rect, is_left=False)

    # 3. EYES
    _draw_eye_gradient(screen, left_eye_rect)
    _draw_eye_gradient(screen, right_eye_rect)

    # 4. EYELIDS (Blink)
    _draw_eyelid(screen, left_eye_rect, _blink_progress)
    _draw_eyelid(screen, right_eye_rect, _blink_progress)

    # 5. MOUTH (Upside down curve - Frown)
    mouth_w = 250
    mouth_h = 110
    base_y = 520

    mouth_points = []
    steps = 100

    # Upper curve (rounded dome going UP)
    radius_x = mouth_w / 2
    radius_y = mouth_h

    for i in range(steps + 1):
        t = i / steps
        px = (center_x - mouth_w // 2) + (t * mouth_w)
        dx = px - center_x
        inside_sqrt = max(0, 1 - (dx / radius_x) ** 2)
        offset_y = radius_y * math.sqrt(inside_sqrt)
        py = base_y - offset_y
        mouth_points.append((px, py))

    # Bottom edge (slight curve inward)
    bottom_sag = 15
    for i in range(steps, -1, -1):
        t = i / steps
        px = (center_x - mouth_w // 2) + (t * mouth_w)
        py = base_y - (bottom_sag * math.sin(t * math.pi))
        mouth_points.append((px, py))

    # Dark mouth cavity
    pygame.draw.polygon(screen, MOUTH_DARK, mouth_points)

    # Tongue (masked)
    mouth_mask = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(mouth_mask, (255, 255, 255), mouth_points)

    tongue_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tongue_fill_height = 55
    tongue_rect = pygame.Rect(center_x - mouth_w // 2, base_y - tongue_fill_height, mouth_w, tongue_fill_height * 2)
    pygame.draw.ellipse(tongue_surf, TONGUE, tongue_rect)

    mouth_mask.blit(tongue_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    screen.blit(mouth_mask, (0, 0))

    # Outline
    pygame.draw.polygon(screen, BLACK, mouth_points, 8)
    pygame.draw.aalines(screen, BLACK, True, mouth_points)
