import pygame
import sys
import math
import random # 1. Import modul random untuk waktu acak

# --- 1. Inisialisasi ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Face - Continuous Natural Blinking")

# --- 2. Warna ---
BG_COLOR    = (205, 215, 225) 
BLACK       = (0, 0, 0)
EYE_COLOR   = (45, 40, 90)
HIGHLIGHT   = (255, 255, 255) 
MOUTH_DARK  = (40, 40, 40)
TONGUE      = (230, 130, 100)

# Warna Baru
BLUSH_COLOR = (255, 180, 200)
EYE_TOP     = (80, 70, 150)
EYE_BOTTOM  = (0, 0, 0)

# --- 3. Fungsi Pendukung (TIDAK ADA YANG DIUBAH) ---

def draw_star(surface, color, x, y, size):
    half = size // 2
    inner = size // 5
    points = [
        (x, y - half), (x + inner, y - inner),
        (x + half, y), (x + inner, y + inner),
        (x, y + half), (x - inner, y + inner),
        (x - half, y), (x - inner, y + inner),
        (x - half, y), (x - inner, y - inner)
    ]
    pygame.draw.polygon(surface, color, points)

def draw_eye_gradient_with_sparkles(surface, rect):
    # 1. Gambar Dasar Mata
    pygame.draw.ellipse(surface, BLACK, rect.inflate(8, 8))

    # 2. Gradasi
    top = globals().get('EYE_TOP', (80, 70, 150))
    bottom = globals().get('EYE_BOTTOM', (0, 0, 0))
    
    gradient_tiny = pygame.Surface((1, 2))
    gradient_tiny.fill(top, (0, 0, 1, 1))    
    gradient_tiny.fill(bottom, (0, 1, 1, 1)) 
    gradient_surf = pygame.transform.smoothscale(gradient_tiny, (rect.width, rect.height))

    eye_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.ellipse(eye_surf, (255, 255, 255), (0, 0, rect.width, rect.height))
    eye_surf.blit(gradient_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    surface.blit(eye_surf, rect.topleft)
    
    # 3. Ornamen
    glint_x = rect.left + 35
    glint_y = rect.top + 45
    draw_star(surface, HIGHLIGHT, glint_x, glint_y, 50) 
    pygame.draw.circle(surface, HIGHLIGHT, (glint_x + 25, glint_y + 25), 5)
    pygame.draw.circle(surface, (150, 150, 255), (glint_x - 15, glint_y + 15), 3)

def draw_blush(surface, x, y):
    w, h = 70, 45
    blush_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.ellipse(blush_surf, (*BLUSH_COLOR, 120), (0, 0, w, h))
    surface.blit(blush_surf, (x - w//2, y - h//2))

def draw_eyelid(surface, rect, progress):
    """
    Fungsi untuk menggambar kelopak mata saat berkedip.
    progress: 0.0 (mata terbuka) sampai 1.0 (mata tertutup total)
    """
    if progress <= 0:
        return 

    # Hitung tinggi kelopak mata
    lid_height = rect.height * progress
    
    # Gambar kotak warna background menutupi mata
    cover_rect = pygame.Rect(rect.left - 5, rect.top - 5, rect.width + 10, lid_height + 5)
    pygame.draw.rect(surface, BG_COLOR, cover_rect)
    
    # Gambar garis hitam
    line_y = rect.top + lid_height
    if line_y > rect.bottom: line_y = rect.bottom
        
    pygame.draw.line(surface, BLACK, (rect.left - 5, line_y), (rect.right + 5, line_y), 6)


# --- 4. Loop Utama ---
running = True
clock = pygame.time.Clock()

# --- VARIABEL ANIMASI KEDIP ---
# Kita mulai dengan "closing" agar saat program jalan, dia langsung kedip sekali (Intro)
blink_state = "closing" 
blink_progress = 0.0    
blink_speed = 0.15      # Kecepatan kedip (0.15 = cepat natural)

# --- VARIABEL TIMER (Untuk Kedipan Seterusnya) ---
last_blink_time = pygame.time.get_ticks()
# Waktu tunggu acak pertama kali setelah intro selesai
next_blink_wait = random.randint(2000, 5000) 

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BG_COLOR)
    
    # Ambil waktu sekarang dalam milidetik
    current_time = pygame.time.get_ticks()

    # --- LOGIKA KEDIP (State Machine) ---
    
    # 1. Jika sedang menutup (Closing)
    if blink_state == "closing":
        blink_progress += blink_speed
        if blink_progress >= 1.0:
            blink_progress = 1.0
            blink_state = "opening" # Lanjut membuka
            
    # 2. Jika sedang membuka (Opening)
    elif blink_state == "opening":
        blink_progress -= blink_speed
        if blink_progress <= 0.0:
            blink_progress = 0.0
            blink_state = "idle"    # Selesai kedip, masuk mode diam (idle)
            
            # --- RESET TIMER ---
            # Setelah mata terbuka, catat waktunya dan tentukan kapan kedip lagi
            last_blink_time = current_time
            next_blink_wait = random.randint(2000, 6000) # Jeda acak 2-6 detik

    # 3. Jika sedang diam (Idle)
    elif blink_state == "idle":
        # Cek apakah waktu tunggu sudah habis?
        if current_time - last_blink_time > next_blink_wait:
            blink_state = "closing" # Mulai kedip lagi!

    # --- SETUP POSISI ---
    center_x = WIDTH // 2
    eye_y = 220
    eye_width = 110
    eye_height = 150
    dist_from_center = 140

    left_eye_rect = pygame.Rect(center_x - dist_from_center - eye_width, eye_y, eye_width, eye_height)
    right_eye_rect = pygame.Rect(center_x + dist_from_center, eye_y, eye_width, eye_height)

    # 1. KABEL
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

    # 2. BLUSH ON
    draw_blush(screen, left_eye_rect.centerx - 20, left_eye_rect.bottom + 20)
    draw_blush(screen, right_eye_rect.centerx + 20, right_eye_rect.bottom + 20)

    # 3. MATA
    draw_eye_gradient_with_sparkles(screen, left_eye_rect)
    draw_eye_gradient_with_sparkles(screen, right_eye_rect)

    # --- GAMBAR KELOPAK MATA (UNTUK EFEK KEDIP) ---
    # Digambar di atas mata agar menutupi
    draw_eyelid(screen, left_eye_rect, blink_progress)
    draw_eyelid(screen, right_eye_rect, blink_progress)

    # 4. MULUT
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
        inside_sqrt = max(0, 1 - (dx / a)**2) 
        offset_y = b * math.sqrt(inside_sqrt)
        py = mouth_top_y + offset_y
        bottom_points.append((px, py))
    
    mouth_points.extend(reversed(bottom_points))

    pygame.draw.polygon(screen, MOUTH_DARK, mouth_points)

    mouth_mask = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(mouth_mask, (255, 255, 255, 255), mouth_points)
    
    tongue_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tongue_rect = pygame.Rect(center_x - mouth_w//2 + 10, mouth_top_y + 50, mouth_w - 20, 110)
    pygame.draw.ellipse(tongue_surf, TONGUE, tongue_rect)
    
    mouth_mask.blit(tongue_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    screen.blit(mouth_mask, (0, 0))

    pygame.draw.polygon(screen, BLACK, mouth_points, 8)
    pygame.draw.aalines(screen, BLACK, True, mouth_points)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()