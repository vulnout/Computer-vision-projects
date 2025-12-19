"""
Fast 3D Earth with High-Quality Textures
Hand Gesture Control - HEIGHT-BASED ZOOM
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import threading
import os

class HandController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 5.0
        
        self.prev_palm = None
        self.prev_hand_height = None
        self.zoom_mode = False
        
        self.smoothing = 0.3
        self.zoom_smoothing = 0.2
        self.target_rot_x = 0.0
        self.target_rot_y = 0.0
        self.target_zoom = 5.0
        
        self.lock = threading.Lock()
        
    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def is_fist(self, hand):
        """Detect if hand is making a fist (all fingers curled)"""
        
        fingers_curled = 0
        
       
        if hand.landmark[8].y > hand.landmark[6].y:
            fingers_curled += 1
        
        if hand.landmark[12].y > hand.landmark[10].y:
            fingers_curled += 1
        
        if hand.landmark[16].y > hand.landmark[14].y:
            fingers_curled += 1
        
        if hand.landmark[20].y > hand.landmark[18].y:
            fingers_curled += 1
            
        return fingers_curled >= 3
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            
            self.mp_draw.draw_landmarks(
                frame, hand, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(0,200,0), thickness=2)
            )
            
            
            if self.is_fist(hand):
                self.zoom_mode = True
                
                
                palm = hand.landmark[9]
                current_height = palm.y
                
                if self.prev_hand_height is not None:
                    
                    delta = (self.prev_hand_height - current_height) * 15
                    with self.lock:
                        self.target_zoom -= delta
                        self.target_zoom = max(1.5, min(12.0, self.target_zoom))
                
                self.prev_hand_height = current_height
                self.prev_palm = None
                
                # feedback
                zoom_percent = int((12.0 - self.target_zoom) / 10.5 * 100)
                cv2.putText(frame, f'ZOOM {zoom_percent}%', (w//2-100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,200,0), 3)
                
                
                palm_pt = (int(palm.x * w), int(palm.y * h))
                cv2.circle(frame, palm_pt, 30, (255, 200, 0), 5)
                
                
                arrow_length = 60
                cv2.arrowedLine(frame, 
                               (palm_pt[0], palm_pt[1] + arrow_length//2),
                               (palm_pt[0], palm_pt[1] - arrow_length//2),
                               (0, 255, 255), 4, tipLength=0.3)
                
            else:
                # Open ha
                self.zoom_mode = False
                self.prev_hand_height = None
                
                palm = hand.landmark[9]
                
                if self.prev_palm:
                    dx = (palm.x - self.prev_palm[0]) * 300
                    dy = (palm.y - self.prev_palm[1]) * 300
                    
                    with self.lock:
                        self.target_rot_y += dx
                        self.target_rot_x += dy
                
                self.prev_palm = (palm.x, palm.y)
                
                cv2.putText(frame, 'ROTATE', (w//2-70, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        else:
            self.prev_palm = None
            self.prev_hand_height = None
            self.zoom_mode = False
        
        # Instructio
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-100), (w, h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, 'Open Hand: Rotate  |  Fist: Zoom', 
                   (10, h-65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, 'Move fist UP to zoom in, DOWN to zoom out', 
                   (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,255,150), 1)
        cv2.putText(frame, 'ESC to quit', 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        return frame
    
    def update(self):
        with self.lock:
            self.rotation_x += (self.target_rot_x - self.rotation_x) * self.smoothing
            self.rotation_y += (self.target_rot_y - self.rotation_y) * self.smoothing
            self.zoom += (self.target_zoom - self.zoom) * self.zoom_smoothing
            
            return self.rotation_x, self.rotation_y, self.zoom

class Earth3D:
    def __init__(self):
        self.controller = HandController()
        self.running = True
        self.cap = None
        
    def camera_thread(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        cv2.namedWindow('Hand Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Control', 480, 360)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = self.controller.process_frame(frame)
                cv2.imshow('Hand Control', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    self.running = False
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def create_earth_texture(self):
        """Create high-quality Earth texture"""
        print("🎨 Generating Earth texture...")
        
        w, h = 4096, 2048
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        print("   → Ocean...")
        for y in range(h):
            lat_factor = abs(y - h/2) / (h/2)
            ocean_blue = int(20 + 40 * (1 - lat_factor**0.5))
            ocean_green = int(40 + 60 * (1 - lat_factor**0.5))
            img[y, :] = [ocean_blue, ocean_green, ocean_blue*3]
        
        print("   → Continents...")
        land_base = [55, 115, 45]
        
        continents = [
            [(600,300), (900,280), (1040,400), (1160,560), (1200,700), (1100,840), (960,900), (800,880), (640,800), (560,640), (500,480), (540,360)],
            [(960,960), (1080,1000), (1160,1160), (1200,1320), (1160,1440), (1080,1480), (1000,1500), (920,1460), (880,1340), (840,1200), (880,1040)],
            [(1960,360), (2100,340), (2240,360), (2300,440), (2280,520), (2200,560), (2080,580), (2000,540), (1940,460)],
            [(2040,640), (2200,620), (2320,680), (2400,800), (2440,960), (2400,1120), (2320,1240), (2200,1320), (2080,1340), (2000,1280), (1960,1160), (1960,1040), (2000,920), (1980,760)],
            [(2400,280), (2700,300), (3000,360), (3240,440), (3400,560), (3440,700), (3360,840), (3200,900), (3000,920), (2800,880), (2600,800), (2500,700), (2440,560), (2400,440)],
            [(3100,1200), (3300,1220), (3440,1300), (3460,1400), (3400,1460), (3280,1480), (3160,1460), (3080,1400), (3060,1300)],
        ]
        
        for pts in continents:
            cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], land_base)
        
        print("   → Details...")
        for _ in range(800):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            if not np.allclose(img[y, x], [img[y, x][0], img[y, x][1], img[y, x][2]], atol=30):
                continue
            size = np.random.randint(10, 50)
            variation = np.random.randint(-25, 25, 3)
            color = np.clip(img[y, x].astype(int) + variation, 0, 255).astype(np.uint8)
            cv2.circle(img, (x, y), size, color.tolist(), -1)
        
        print("   → Ice caps...")
        cv2.ellipse(img, (w//2, 100), (1600, 120), 0, 0, 360, [245, 250, 255], -1)
        cv2.ellipse(img, (w//2, h-100), (1600, 120), 0, 0, 360, [245, 250, 255], -1)
        
        print("   → Clouds...")
        for _ in range(600):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(25, 90)
            overlay = img.copy()
            cv2.circle(overlay, (x, y), size, [235, 245, 255], -1)
            alpha = np.random.uniform(0.08, 0.22)
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        
        print("✓ Done\n")
        cv2.imwrite("earth_natural.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return img
    
    def load_texture(self, image):
        img_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_data = cv2.flip(img_data, 0)
        
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_data.shape[1], img_data.shape[0],
                     0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        
        return texture
    
    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)
        
        glLight(GL_LIGHT0, GL_POSITION, [10, 10, 10, 1])
        glLight(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.35, 1])
        glLight(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 0.98, 1])
        
        glClearColor(0.01, 0.01, 0.05, 1.0)
    
    def draw_stars(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glPointSize(2.0)
        glBegin(GL_POINTS)
        
        np.random.seed(42)
        for _ in range(3000):
            x = np.random.uniform(-40, 40)
            y = np.random.uniform(-40, 40)
            z = np.random.uniform(-40, -15)
            b = np.random.uniform(0.7, 1.0)
            glColor3f(b, b, b * 0.95)
            glVertex3f(x, y, z)
        
        glEnd()
        glEnable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
    
    def run(self):
        print("\n" + "="*70)
        print("      🌍 3D EARTH MODEL")
        print("="*70 + "\n")
        
        if os.path.exists("earth_natural.jpg"):
            print("✓ Loading cached texture\n")
            earth_img = cv2.imread("earth_natural.jpg")
        else:
            earth_img = self.create_earth_texture()
        
        pygame.init()
        display = (1400, 900)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption('🌍 Earth 3D - Hand Control')
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, display[0]/display[1], 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        self.init_gl()
        
        print("📦 Loading texture...")
        texture = self.load_texture(earth_img)
        print("✓ Ready\n")
        
        sphere = gluNewQuadric()
        gluQuadricTexture(sphere, GL_TRUE)
        gluQuadricNormals(sphere, GLU_SMOOTH)
        
        print("📹 Starting hand tracking...")
        camera = threading.Thread(target=self.camera_thread, daemon=True)
        camera.start()
        
        clock = pygame.time.Clock()
        
        print("\n" + "="*70)
        print("✅ READY!")
        print("="*70)
        print("\n🎮 CONTROLS:")
        print("  🖐️  Open hand → Rotate")
        print("  ✊ Fist + move UP/DOWN → Zoom")
        print("  ESC → Quit")
        print("="*70 + "\n")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            self.draw_stars()
            
            rot_x, rot_y, zoom = self.controller.update()
            
            glLoadIdentity()
            glTranslatef(0, 0, -zoom)
            glRotatef(rot_x * 0.5, 1, 0, 0)
            glRotatef(rot_y * 0.5, 0, 1, 0)
            
            glBindTexture(GL_TEXTURE_2D, texture)
            glColor3f(1, 1, 1)
            gluSphere(sphere, 2.0, 64, 64)
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    earth = Earth3D()
    
    try:
        earth.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        earth.running = False