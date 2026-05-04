import pygame
import sys
import math
import random # 1. Import Random

# --- 1. Inisialisasi ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Face - Gradient Eyes + Natural Blink + Mouth Blink")

# --- 2. Warna ---
BG_COLOR    = (205, 215, 225) 
BLACK       = (0, 0, 0)

# Warna Gradasi Mata
EYE_TOP     = (80, 70, 150)   # Ungu
EYE_BOTTOM  = (0, 0, 0)       # Hitam  

HIGHLIGHT   = (240, 245, 255)
MOUTH_DARK  = (40, 40, 40)
TONGUE      = (230, 130, 100)

# --- 3. Fungsi Gambar ---

def draw_eye_gradient(surface, rect):
    """Menggambar mata dengan gradasi"""
    # 1. Gambar Outline Hitam
    pygame.draw.ellipse(surface, BLACK, rect.inflate(8, 8))

    # 2. Membuat Gradasi
    gradient_tiny = pygame.Surface((1, 2))
    gradient_tiny.fill(EYE_TOP, (0, 0, 1, 1))    
    gradient_tiny.fill(EYE_BOTTOM, (0, 1, 1, 1)) 
    gradient_surf = pygame.transform.smoothscale(gradient_tiny, (rect.width, rect.height))

    # 3. Masking
    eye_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.ellipse(eye_surf, (255, 255, 255), (0, 0, rect.width, rect.height))
    eye_surf.blit(gradient_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # 4. Tempel Mata
    surface.blit(eye_surf, rect.topleft)
    
    # 5. Highlight
    glint_x = rect.left + 35
    glint_y = rect.top + 40
    pygame.draw.circle(surface, HIGHLIGHT, (glint_x, glint_y), 22)
    pygame.draw.circle(surface, EYE_TOP, (glint_x + 10, glint_y + 10), 10)

    small_glint_x = glint_x - 5
    small_glint_y = glint_y + 45 
    pygame.draw.circle(surface, HIGHLIGHT, (small_glint_x, small_glint_y), 6)

def draw_eyelid(surface, rect, progress):
    """
    Fungsi menggambar kelopak mata untuk efek kedip.
    """
    if progress <= 0:
        return 

    lid_height = rect.height * progress
    
    cover_rect = pygame.Rect(rect.left - 5, rect.top - 5, rect.width + 10, lid_height + 5)
    pygame.draw.rect(surface, BG_COLOR, cover_rect)
    
    line_y = rect.top + lid_height
    if line_y > rect.bottom: line_y = rect.bottom
        
    pygame.draw.line(surface, BLACK, (rect.left - 5, line_y), (rect.right + 5, line_y), 6)

# --- 4. Loop Utama ---
running = True
clock = pygame.time.Clock()

# --- VARIABEL KEDIP (INTRO + NATURAL) ---
blink_state = "closing" # Intro: Langsung kedip
blink_progress = 0.0    
blink_speed = 0.15      # Kecepatan kedip

# Timer untuk kedip selanjutnya
last_blink_time = pygame.time.get_ticks()
next_blink_interval = random.randint(2000, 5000)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BG_COLOR)

    # --- LOGIKA UPDATE ANIMASI KEDIP ---
    current_time = pygame.time.get_ticks()

    if blink_state == "closing":
        blink_progress += blink_speed
        if blink_progress >= 1.0:
            blink_progress = 1.0
            blink_state = "opening"
            
    elif blink_state == "opening":
        blink_progress -= blink_speed
        if blink_progress <= 0.0:
            blink_progress = 0.0
            blink_state = "idle"
            # Reset Timer
            last_blink_time = current_time 
            next_blink_interval = random.randint(2000, 6000)

    elif blink_state == "idle":
        if current_time - last_blink_time > next_blink_interval:
            blink_state = "closing"

    # Koordinat Utama
    center_x = WIDTH // 2
    eye_y = 220
    eye_width = 110
    eye_height = 150
    dist_from_center = 140

    # Kotak Mata
    left_eye_rect = pygame.Rect(center_x - dist_from_center - eye_width, eye_y, eye_width, eye_height)
    right_eye_rect = pygame.Rect(center_x + dist_from_center, eye_y, eye_width, eye_height)

    # ==========================
    # BAGIAN 1: KABEL 
    # ==========================
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

    # ==========================
    # BAGIAN 2: MATA DENGAN GRADASI
    # ==========================
    draw_eye_gradient(screen, left_eye_rect)
    draw_eye_gradient(screen, right_eye_rect)

    # --- GAMBAR KELOPAK MATA (KEDIP) ---
    draw_eyelid(screen, left_eye_rect, blink_progress)
    draw_eyelid(screen, right_eye_rect, blink_progress)

    # ==========================================
    # BAGIAN 3: MULUT (IKUT BERKEDIP/GEPENG)
    # ==========================================
    
    # 1. Konfigurasi Awal
    base_mouth_w = 160
    base_mouth_h = 130
    
    # 2. Hitung Bentuk Mulut Berdasarkan Kedip
    # Semakin besar blink_progress (mata tertutup), semakin kecil mulut
    current_mouth_h = base_mouth_h * (1.0 - blink_progress)
    # Squash & Stretch: Mulut melebar sedikit saat gepeng
    current_mouth_w = base_mouth_w + (blink_progress * 20)
    
    if current_mouth_h < 4: current_mouth_h = 4

    # 3. Buat Rect Mulut Baru
    mouth_rect = pygame.Rect(0, 0, current_mouth_w, current_mouth_h)
    mouth_rect.center = (center_x, 400 + base_mouth_h//2) # Jaga posisi tengah

    # 4. Gambar Mulut
    if current_mouth_h > 5:
        # A. Dasar Mulut
        pygame.draw.ellipse(screen, MOUTH_DARK, mouth_rect)
        
        # B. Lidah (Clipping)
        clip_rect = pygame.Rect(mouth_rect.left, mouth_rect.centery, mouth_rect.width, mouth_rect.height // 2)
        screen.set_clip(clip_rect) 
        pygame.draw.ellipse(screen, TONGUE, mouth_rect)
        screen.set_clip(None)
        
        # C. Outline
        pygame.draw.ellipse(screen, BLACK, mouth_rect, 6)
    else:
        # Jika mulut sangat tipis (saat merem total), gambar garis saja
        pygame.draw.line(screen, BLACK, 
                         (mouth_rect.left, mouth_rect.centery), 
                         (mouth_rect.right, mouth_rect.centery), 6)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()