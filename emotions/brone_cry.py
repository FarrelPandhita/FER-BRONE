import pygame
import sys
import math

# --- 1. Inisialisasi ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Face - Purple Eyes (No Shadow Reflection)")

# --- 2. Warna ---
BG_COLOR    = (205, 215, 225) 
BLACK       = (0, 0, 0)
HIGHLIGHT   = (255, 255, 255)
MOUTH_DARK  = (40, 40, 40)
TONGUE      = (230, 130, 100) 

# Warna Air Mata & Gelombang (Cyan Terang)
TEAR_STREAM_COLOR = (170, 230, 255) 
EYE_WATER         = (130, 200, 255) # gelombang di dalam mata

# WARNA MATA DASAR
EYE_BASE_COLOR = (40, 30, 70)     

# --- 3. Variabel Animasi ---
time_counter = 0        

# --- 4. Fungsi Gambar ---

def draw_purple_eye_with_wave(surface, rect, time_val):
    
    
    # A. Dasar Mata (Ungu Gelap Penuh)
    pygame.draw.ellipse(surface, EYE_BASE_COLOR, rect)
    
    # B. LOGIKA GELOMBANG AIR (Wave)
    wave_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    
    water_points = []
    # Tinggi air (sekitar setengah mata)
    water_level = rect.height * 0.55 
    
    # Membuat titik-titik gelombang sinus
    for x in range(rect.width):
        wave_height = 4 * math.sin(0.15 * x + time_val) 
        water_points.append((x, water_level + wave_height))
    
    # Tutup poligon air ke bawah
    water_points.append((rect.width, rect.height))
    water_points.append((0, rect.height))
    
    # Gambar air biru pada surface sementara
    pygame.draw.polygon(wave_surf, EYE_WATER, water_points)
    
    # MASKING
    mask = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.ellipse(mask, (255, 255, 255), (0, 0, rect.width, rect.height))
    wave_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    # Tempelkan hasil air ke mata utama
    surface.blit(wave_surf, rect.topleft)

    

    # D. Outline Tebal Hitam
    pygame.draw.ellipse(surface, BLACK, rect, 6)

    # E. HIGHLIGHTS (Lingkaran Putih)
    # Highlight Besar
    big_glint_pos = (rect.left + 35, rect.top + 45)
    pygame.draw.circle(surface, HIGHLIGHT, big_glint_pos, 22)
    
    # Highlight Kecil
    small_glint_pos = (rect.left + 55, rect.top + 80)
    pygame.draw.circle(surface, HIGHLIGHT, small_glint_pos, 6)


def draw_cartoon_stream_slow(surface, start_x, start_y, time_val):
    """ 
    Fungsi Air Mata Deras (Stream) - DIPERLAMBAT
    """
    stream_points = []
    width_top = 40      
    width_bottom = 50   
    
    # Kiri (Turun)
    for y in range(start_y, HEIGHT):
        prog = (y - start_y) / (HEIGHT - start_y) 
        current_w = width_top + (width_bottom - width_top) * prog
        wiggle = math.sin(y * 0.05 + time_val) * 4
        x = start_x - (current_w / 2) + wiggle
        stream_points.append((x, y))
        
    # Kanan (Naik)
    for y in range(HEIGHT, start_y, -1):
        prog = (y - start_y) / (HEIGHT - start_y)
        current_w = width_top + (width_bottom - width_top) * prog
        wiggle = math.sin(y * 0.05 + time_val) * 4
        x = start_x + (current_w / 2) + wiggle
        stream_points.append((x, y))
        
    pygame.draw.polygon(surface, TEAR_STREAM_COLOR, stream_points)
    
    # Mahkota Air
    pygame.draw.circle(surface, TEAR_STREAM_COLOR, (start_x - 15, start_y + 5), 10)
    pygame.draw.circle(surface, TEAR_STREAM_COLOR, (start_x, start_y + 8), 12)
    pygame.draw.circle(surface, TEAR_STREAM_COLOR, (start_x + 15, start_y + 5), 10)

    # Kilauan Air Jatuh (DIPERLAMBAT)
    num_highlights = 3 
    for i in range(num_highlights):
        offset = i * 250 
        drop_speed = 25 
        drop_y = start_y + ((time_val * drop_speed + offset) % (HEIGHT - start_y + 100))
        
        if drop_y < HEIGHT:
            h_rect = pygame.Rect(start_x - 8, drop_y, 16, 35)
            pygame.draw.ellipse(surface, HIGHLIGHT, h_rect)


# --- 5. Loop Utama ---
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BG_COLOR)
    
    time_counter += 0.1 

    center_x = WIDTH // 2
    eye_y = 220
    eye_width = 110
    eye_height = 150
    dist_from_center = 140

    left_eye_rect = pygame.Rect(center_x - dist_from_center - eye_width, eye_y, eye_width, eye_height)
    right_eye_rect = pygame.Rect(center_x + dist_from_center, eye_y, eye_width, eye_height)

    # BAGIAN 1: KABEL 
    elbow_y = left_eye_rect.top - 50 
    pygame.draw.lines(screen, BLACK, False, [(-20, 60), (left_eye_rect.centerx, elbow_y), (left_eye_rect.centerx, left_eye_rect.top)], 4)
    pygame.draw.lines(screen, BLACK, False, [(WIDTH + 20, 60), (right_eye_rect.centerx, elbow_y), (right_eye_rect.centerx, right_eye_rect.top)], 4)
    pygame.draw.lines(screen, BLACK, False, [(left_eye_rect.right - 10, left_eye_rect.centery), (center_x, left_eye_rect.centery + 40), (right_eye_rect.left + 10, right_eye_rect.centery)], 4)

    # ==========================================
    # BAGIAN 2: ALIRAN AIR MATA (STREAM) - SLOW
    # ==========================================
    draw_cartoon_stream_slow(screen, left_eye_rect.centerx, left_eye_rect.bottom - 15, time_counter)
    draw_cartoon_stream_slow(screen, right_eye_rect.centerx, right_eye_rect.bottom - 15, time_counter)

    # ==========================================
    # BAGIAN 3: MATA UNGU + GELOMBANG AIR (TANPA BAYANGAN)
    # ==========================================
    draw_purple_eye_with_wave(screen, left_eye_rect, time_counter)
    draw_purple_eye_with_wave(screen, right_eye_rect, time_counter + 2)

    # ==========================================
    # BAGIAN 4: MULUT SEDIH
    # ==========================================
    mouth_w = 250        
    mouth_h = 110        
    base_y = 520         

    mouth_points = []
    steps = 100
    radius_x = mouth_w / 2
    radius_y = mouth_h

    # Kubah Atas
    for i in range(steps + 1):
        t = i / steps
        px = (center_x - mouth_w // 2) + (t * mouth_w)
        dx = px - center_x
        inside_sqrt = max(0, 1 - (dx / radius_x)**2)
        offset_y = radius_y * math.sqrt(inside_sqrt)
        py = base_y - offset_y
        mouth_points.append((px, py))

    # Alas Bawah
    bottom_sag = 15 
    for i in range(steps, -1, -1):
        t = i / steps
        px = (center_x - mouth_w // 2) + (t * mouth_w)
        py = base_y - (bottom_sag * math.sin(t * math.pi))
        mouth_points.append((px, py))

    pygame.draw.polygon(screen, MOUTH_DARK, mouth_points)

    mouth_mask = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(mouth_mask, (255, 255, 255), mouth_points)
    
    tongue_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tongue_fill_height = 55 
    tongue_rect = pygame.Rect(center_x - mouth_w//2, base_y - tongue_fill_height, mouth_w, tongue_fill_height * 2)
    pygame.draw.ellipse(tongue_surf, TONGUE, tongue_rect)
    
    mouth_mask.blit(tongue_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    screen.blit(mouth_mask, (0, 0))

    pygame.draw.polygon(screen, BLACK, mouth_points, 8)
    pygame.draw.aalines(screen, BLACK, True, mouth_points)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()