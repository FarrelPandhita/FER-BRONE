import pygame
import sys
import math
import random 

# --- 1. Inisialisasi ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Face - Final Fix + Natural Blink")

# --- 2. Warna ---
BG_COLOR    = (205, 215, 225) 
BLACK       = (0, 0, 0)
HIGHLIGHT   = (240, 245, 255)
MOUTH_DARK  = (40, 40, 40)
TONGUE      = (230, 130, 100)

# Variabel Warna Gradasi (Mata Biasa)
EYE_TOP     = (80, 70, 150)   # Ungu
EYE_BOTTOM  = (0, 0, 0)       # Hitam

# --- 3. Fungsi Gambar ---

def draw_eye_gradient(surface, rect):
    # 1. Outline Hitam
    pygame.draw.ellipse(surface, BLACK, rect.inflate(8, 8))

    # 2. GradasI
    gradient_tiny = pygame.Surface((1, 2))
    gradient_tiny.fill(EYE_TOP, (0, 0, 1, 1))    
    gradient_tiny.fill(EYE_BOTTOM, (0, 1, 1, 1)) 
    gradient_surf = pygame.transform.smoothscale(gradient_tiny, (rect.width, rect.height))

    # 3. Masking Oval
    eye_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.ellipse(eye_surf, (255, 255, 255), (0, 0, rect.width, rect.height))
    eye_surf.blit(gradient_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # 4. Tempel
    surface.blit(eye_surf, rect.topleft)
    
    # 5. Highlight (Kilatan)
    glint_x = rect.left + 35
    glint_y = rect.top + 40
    pygame.draw.circle(surface, HIGHLIGHT, (glint_x, glint_y), 22)
    pygame.draw.circle(surface, EYE_TOP, (glint_x + 10, glint_y + 10), 10)
    
    small_glint_x = glint_x - 5
    small_glint_y = glint_y + 45 
    pygame.draw.circle(surface, HIGHLIGHT, (small_glint_x, small_glint_y), 6)

def draw_eyelid(surface, rect, progress):
    """
    Fungsi menggambar kelopak mata untuk efek kedip
    """
    if progress <= 0:
        return 

    lid_height = rect.height * progress
    
    # Gambar kotak warna background menutupi mata
    # Inflate width sedikit (+10) agar menutupi garis outline mata juga
    cover_rect = pygame.Rect(rect.left - 5, rect.top - 5, rect.width + 10, lid_height + 5)
    pygame.draw.rect(surface, BG_COLOR, cover_rect)
    
    # Garis bulu mata (lipatan kelopak)
    line_y = rect.top + lid_height
    if line_y > rect.bottom: line_y = rect.bottom
        
    pygame.draw.line(surface, BLACK, (rect.left - 5, line_y), (rect.right + 5, line_y), 6)


# --- 4. Loop Utama ---
running = True
clock = pygame.time.Clock()

# --- VARIABEL TRANSISI KEDIP (INTRO + NATURAL) ---
blink_state = "closing" # Mulai dengan menutup (Intro)
blink_progress = 0.0    
blink_speed = 0.05      # KECEPATAN KEDIP 

# Timer untuk kedip berikutnya
last_blink_time = pygame.time.get_ticks()
next_blink_interval = random.randint(2000, 5000)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BG_COLOR)
    
    current_time = pygame.time.get_ticks()

    # --- LOGIKA KEDIP ---
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

    # Koordinat
    center_x = WIDTH // 2
    eye_y = 220
    eye_width = 110
    eye_height = 150
    dist_from_center = 140

    left_eye_rect = pygame.Rect(center_x - dist_from_center - eye_width, eye_y, eye_width, eye_height)
    right_eye_rect = pygame.Rect(center_x + dist_from_center, eye_y, eye_width, eye_height)

    # ==========================
    # BAGIAN 1: KABEL 
    # ==========================
    elbow_y = left_eye_rect.top - 50 
    pygame.draw.lines(screen, BLACK, False, [(-20, 60), (left_eye_rect.centerx, elbow_y), (left_eye_rect.centerx, left_eye_rect.top)], 4)
    pygame.draw.lines(screen, BLACK, False, [(WIDTH + 20, 60), (right_eye_rect.centerx, elbow_y), (right_eye_rect.centerx, right_eye_rect.top)], 4)
    pygame.draw.lines(screen, BLACK, False, [(left_eye_rect.right - 10, left_eye_rect.centery), (center_x, left_eye_rect.centery + 40), (right_eye_rect.left + 10, right_eye_rect.centery)], 4)

    # ==========================
    # BAGIAN 2: MATA (SIMPLE)
    # ==========================
    draw_eye_gradient(screen, left_eye_rect)
    draw_eye_gradient(screen, right_eye_rect)

    # ==========================
    # BAGIAN BARU: KELOPAK MATA (KEDIP)
    # ==========================
    draw_eyelid(screen, left_eye_rect, blink_progress)
    draw_eyelid(screen, right_eye_rect, blink_progress)

    # ==========================
    # BAGIAN 3: MULUT (ROUNDED - LIDAH TURUN)
    # ==========================
    
    mouth_w = 250        # Lebar mulut
    mouth_h = 110        # Tinggi kubah
    base_y = 520         # Garis dasar mulut

    mouth_points = []
    steps = 100

    # A. KUBAH ATAS (RUMUS ELIPS - ROUNDED)
    radius_x = mouth_w / 2
    radius_y = mouth_h

    for i in range(steps + 1):
        t = i / steps
        px = (center_x - mouth_w // 2) + (t * mouth_w)
        dx = px - center_x
        inside_sqrt = max(0, 1 - (dx / radius_x)**2)
        offset_y = radius_y * math.sqrt(inside_sqrt)
        py = base_y - offset_y
        mouth_points.append((px, py))

    # B. ALAS BAWAH (Sedikit melengkung ke dalam)
    bottom_sag = 15   # <--- INI DITAMBAHKAN
    for i in range(steps, -1, -1):
        t = i / steps
        px = (center_x - mouth_w // 2) + (t * mouth_w)
        py = base_y - (bottom_sag * math.sin(t * math.pi))
        mouth_points.append((px, py))

    # 1. Rongga Mulut
    pygame.draw.polygon(screen, MOUTH_DARK, mouth_points)

    # 2. Lidah (Masking Penuh)
    mouth_mask = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(mouth_mask, (255, 255, 255), mouth_points)
    
    tongue_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
   
    tongue_fill_height = 55 
    
    tongue_rect = pygame.Rect(center_x - mouth_w//2, base_y - tongue_fill_height, mouth_w, tongue_fill_height * 2)
    pygame.draw.ellipse(tongue_surf, TONGUE, tongue_rect)
    
    # Potong lidah
    mouth_mask.blit(tongue_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    screen.blit(mouth_mask, (0, 0))

    # 3. Outline
    pygame.draw.polygon(screen, BLACK, mouth_points, 8)
    pygame.draw.aalines(screen, BLACK, True, mouth_points)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()