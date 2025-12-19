import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import random
import math
from collections import deque
from PIL import Image, ImageTk
import mediapipe as mp

class Fruit:
    def __init__(self, x, y, fruit_type, canvas_width, difficulty_multiplier=1.0):
        self.x = x
        self.y = y
        self.fruit_type = fruit_type
        self.vx = random.uniform(-2, 2) * difficulty_multiplier
        self.vy = random.uniform(-15, -22) * (1 + difficulty_multiplier * 0.3)
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-8, 8) * (1 + difficulty_multiplier * 0.5)
        self.sliced = False
        self.slice_time = 0
        self.slice_angle = 0
        self.canvas_width = canvas_width
        self.pulse = 0
        
        self.fruits = {
            'apple': {'colors': ['#FF0800', '#D00000', '#8B0000'], 'size': 55, 'highlight': '#FF6B6B', 'glow': '#FF3333'},
            'orange': {'colors': ['#FF8C00', '#FF7000', '#CC5500'], 'size': 55, 'highlight': '#FFB347', 'glow': '#FFAA00'},
            'banana': {'colors': ['#FFE135', '#FFDB00', '#F4C430'], 'size': 65, 'highlight': '#FFEE88', 'glow': '#FFFF00'},
            'watermelon': {'colors': ['#00A651', '#008744', '#006633'], 'size': 70, 'highlight': '#5FD35F', 'glow': '#00FF66'},
            'kiwi': {'colors': ['#8DB600', '#7A9F35', '#6B8E23'], 'size': 50, 'highlight': '#B5E550', 'glow': '#AAFF00'},
            'strawberry': {'colors': ['#FC5A8D', '#F73859', '#C41E3A'], 'size': 50, 'highlight': '#FF8FAB', 'glow': '#FF66AA'},
        }
        
        self.props = self.fruits[fruit_type]
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.6
        self.rotation += self.rotation_speed
        self.pulse += 0.15
        
        if self.sliced:
            self.slice_time += 1
        
    def is_off_screen(self):
        return self.y > 600 or (self.sliced and self.slice_time > 30)

class Bomb:
    def __init__(self, x, y, canvas_width, difficulty_multiplier=1.0):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2) * difficulty_multiplier
        self.vy = random.uniform(-15, -22) * (1 + difficulty_multiplier * 0.3)
        self.rotation = 0
        self.sliced = False
        self.slice_time = 0
        self.canvas_width = canvas_width
        self.pulse = 0
        self.spark_timer = 0
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.6
        self.rotation += 5
        self.pulse += 0.2
        self.spark_timer += 1
        
        if self.sliced:
            self.slice_time += 1
    
    def is_off_screen(self):
        return self.y > 600 or (self.sliced and self.slice_time > 5)

class FruitNinjaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🥷 Fruit Ninja - Enhanced Edition")
        self.root.geometry("1000x750")
        
        self.root.configure(bg='#1A0F0A')
        
        # Game state
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.max_combo = 0
        self.game_started = False
        self.game_over = False
        self.fruits = []
        self.bombs = []
        self.trail = deque(maxlen=30)
        self.slash_effects = []
        self.particles = []
        self.background_particles = []
        self.level = 1
        self.difficulty_multiplier = 1.0
        
        # Enhanced visual effects
        self.screen_shake = 0
        self.flash_effect = 0
        self.combo_flash = 0
        self.score_popups = []
        
        # Initialize background particles
        for _ in range(50):
            self.background_particles.append({
                'x': random.randint(0, 500),
                'y': random.randint(0, 500),
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(0.3, 0.8),
                'size': random.uniform(1, 3),
                'color': random.choice(['#FFD700', '#FFA500', '#FF6347', '#FF69B4'])
            })
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.finger_pos = None
        self.prev_finger_pos = None
        self.finger_history = deque(maxlen=5)
        
        self.cap = None
        
        self.fruit_types = ['apple', 'orange', 'banana', 'watermelon', 'kiwi', 'strawberry']
        
        self.setup_ui()
        
        self.spawn_timer = 0
        self.bomb_spawn_timer = 0
        self.running = False
        self.combo_timer = 0
        
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#1A0F0A')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Left side - Webcam preview
        left_frame = tk.Frame(main_frame, bg='#2D1B13', relief=tk.RIDGE, borderwidth=4)
        left_frame.pack(side=tk.LEFT, padx=10)
        
        webcam_title = tk.Label(
            left_frame, 
            text="👁️ CAMERA VIEW", 
            font=("Impact", 16, "bold"), 
            bg='#2D1B13', 
            fg='#FFD700'
        )
        webcam_title.pack(pady=8)
        
        self.webcam_canvas = tk.Canvas(left_frame, width=320, height=240, bg='black', highlightthickness=0)
        self.webcam_canvas.pack(padx=8, pady=8)
        
        self.status_label = tk.Label(
            left_frame,
            text="🔴 Waiting for hand...",
            font=("Arial", 12, "bold"),
            bg='#2D1B13',
            fg='#FF5252'
        )
        self.status_label.pack(pady=5)
        
        # Difficulty indicator
        self.difficulty_label = tk.Label(
            left_frame,
            text="LEVEL 1",
            font=("Impact", 14, "bold"),
            bg='#2D1B13',
            fg='#00FF88'
        )
        self.difficulty_label.pack(pady=5)
        
        # Right side - Game area
        right_frame = tk.Frame(main_frame, bg='#1A0F0A')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Title with glow effect
        title_frame = tk.Frame(right_frame, bg='#1A0F0A')
        title_frame.pack(pady=5)
        
        title = tk.Label(
            title_frame, 
            text="⚔️ FRUIT", 
            font=("Impact", 46, "bold"),
            bg='#1A0F0A',
            fg='#FF4500'
        )
        title.pack(side=tk.LEFT)
        
        title2 = tk.Label(
            title_frame, 
            text=" NINJA ⚔️", 
            font=("Impact", 46, "bold"),
            bg='#1A0F0A',
            fg='#FFD700'
        )
        title2.pack(side=tk.LEFT)
        
        # Score display with effects
        score_frame = tk.Frame(right_frame, bg='#1A0F0A')
        score_frame.pack(pady=5)
        
        self.score_label = tk.Label(
            score_frame,
            text=f"SCORE: {self.score}",
            font=("Impact", 26, "bold"),
            bg='#1A0F0A',
            fg='#FFD700'
        )
        self.score_label.pack(side=tk.LEFT, padx=10)
        
        self.combo_label = tk.Label(
            score_frame,
            text="",
            font=("Impact", 24, "bold"),
            bg='#1A0F0A',
            fg='#FF1744'
        )
        self.combo_label.pack(side=tk.LEFT, padx=10)
        
        self.lives_label = tk.Label(
            score_frame,
            text="❤️" * self.lives,
            font=("Arial", 24),
            bg='#1A0F0A'
        )
        self.lives_label.pack(side=tk.LEFT, padx=10)
        
        # Game Canvas with gradient effect
        canvas_container = tk.Frame(right_frame, bg='#4A2511', relief=tk.RIDGE, borderwidth=5)
        canvas_container.pack(pady=10)
        
        self.canvas = tk.Canvas(
            canvas_container,
            width=500,
            height=500,
            bg='#0D0D0D',
            highlightthickness=0
        )
        self.canvas.pack(padx=3, pady=3)
        
        # Control buttons with gradient
        button_frame = tk.Frame(right_frame, bg='#1A0F0A')
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(
            button_frame,
            text="⚔️ START GAME ⚔️",
            font=("Impact", 18, "bold"),
            bg='#D32F2F',
            fg='white',
            command=self.start_game,
            padx=35,
            pady=15,
            relief=tk.RAISED,
            borderwidth=5,
            cursor="hand2",
            activebackground='#FF5252'
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        quit_button = tk.Button(
            button_frame,
            text="✖ EXIT",
            font=("Impact", 18, "bold"),
            bg='#212121',
            fg='white',
            command=self.cleanup,
            padx=35,
            pady=15,
            relief=tk.RAISED,
            borderwidth=5,
            cursor="hand2",
            activebackground='#424242'
        )
        quit_button.pack(side=tk.LEFT, padx=5)
        
        # Enhanced tips
        tips = tk.Label(
            right_frame,
            text="💡 Slice fruits, avoid bombs! Difficulty increases with score! 💡",
            font=("Arial", 11, "bold"),
            bg='#1A0F0A',
            fg='#FFB74D',
            justify=tk.CENTER
        )
        tips.pack(pady=5)
        
    def calculate_difficulty(self):
        """Progressive difficulty based on score"""
        if self.score < 50:
            self.level = 1
            self.difficulty_multiplier = 1.0
        elif self.score < 150:
            self.level = 2
            self.difficulty_multiplier = 1.3
        elif self.score < 300:
            self.level = 3
            self.difficulty_multiplier = 1.6
        elif self.score < 500:
            self.level = 4
            self.difficulty_multiplier = 2.0
        elif self.score < 750:
            self.level = 5
            self.difficulty_multiplier = 2.5
        else:
            self.level = 6
            self.difficulty_multiplier = 3.0
        
        self.difficulty_label.config(
            text=f"LEVEL {self.level}",
            fg=self.get_level_color()
        )
    
    def get_level_color(self):
        colors = ['#00FF88', '#00DDFF', '#FFD700', '#FF8800', '#FF4444', '#FF00FF']
        return colors[min(self.level - 1, 5)]
    
    def get_spawn_rate(self):
        """Faster spawning at higher levels"""
        base_rate = 40
        return max(15, int(base_rate - (self.level - 1) * 5))
    
    def get_bomb_spawn_rate(self):
        """increase the number of bombs at higher levels"""
        base_rate = 180
        return max(60, int(base_rate - (self.level - 1) * 20))
        
    def start_game(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access webcam!")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.lives = 3
        self.fruits = []
        self.bombs = []
        self.trail.clear()
        self.slash_effects = []
        self.particles = []
        self.score_popups = []
        self.game_over = False
        self.game_started = True
        self.running = True
        self.spawn_timer = 0
        self.bomb_spawn_timer = 0
        self.combo_timer = 0
        self.level = 1
        self.difficulty_multiplier = 1.0
        self.screen_shake = 0
        self.flash_effect = 0
        self.finger_history.clear()
        
        self.update_score()
        self.start_button.config(state=tk.DISABLED)
        self.game_loop()
        
    def detect_finger_simple(self, frame):
        """SIMPLIFIED detection - just get the finger tip!"""
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        finger_tip = None
        detection_method = "Searching"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=5),
                    self.mp_draw.DrawingSpec(color=(255, 215, 0), thickness=3)
                )
                
                index_tip = hand_landmarks.landmark[8]
                cx = int(index_tip.x * w)
                cy = int(index_tip.y * h)
                cx = max(0, min(cx, w - 1))
                cy = max(0, min(cy, h - 1))
                
                finger_tip = (cx, cy)
                detection_method = "MediaPipe"
                break
        
        if finger_tip:
            self.finger_history.append(finger_tip)
            
            if len(self.finger_history) >= 3:
                avg_x = sum(p[0] for p in self.finger_history) / len(self.finger_history)
                avg_y = sum(p[1] for p in self.finger_history) / len(self.finger_history)
                finger_tip = (int(avg_x), int(avg_y))
        
        if finger_tip:
            # Enhanced tracking indicator with glow
            cv2.circle(frame, finger_tip, 35, (255, 215, 0), 3)
            cv2.circle(frame, finger_tip, 28, (0, 255, 255), -1)
            cv2.circle(frame, finger_tip, 20, (255, 255, 255), -1)
            
            coord_text = f"({finger_tip[0]}, {finger_tip[1]})"
            cv2.putText(frame, coord_text, (finger_tip[0] - 60, finger_tip[1] - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        status_color = (0, 255, 255) if detection_method == "MediaPipe" else (255, 100, 100)
        cv2.putText(frame, detection_method, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        return frame, finger_tip, detection_method
    
    def check_collisions(self, x, y):
        if self.prev_finger_pos is None:
            return False
            
        prev_x, prev_y = self.prev_finger_pos
        speed = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        
        if speed < 3:
            return False
        
        sliced_any = False
        sliced_count = 0
        
        # Check bombs first
        for bomb in self.bombs:
            if bomb.sliced:
                continue
            
            dx = x - bomb.x
            dy = y - bomb.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 60:
                bomb.sliced = True
                self.explode_bomb(bomb.x, bomb.y)
                self.lives -= 1
                self.combo = 0
                self.screen_shake = 15
                self.flash_effect = 10
                self.update_score()
                
                if self.lives <= 0:
                    self.end_game()
                return True
        
        # Check fruits
        for fruit in self.fruits:
            if fruit.sliced:
                continue
            
            dx = x - fruit.x
            dy = y - fruit.y
            distance = math.sqrt(dx**2 + dy**2)
            
            prev_dx = prev_x - fruit.x
            prev_dy = prev_y - fruit.y
            prev_distance = math.sqrt(prev_dx**2 + prev_dy**2)
            
            size = fruit.props['size']
            hit_radius = size / 2 + 50
            
            if distance < hit_radius or prev_distance < hit_radius:
                fruit.sliced = True
                fruit.slice_angle = math.atan2(y - prev_y, x - prev_x)
                sliced_count += 1
                
                # Enhanced particles with colors
                for _ in range(15):
                    angle = random.uniform(0, 2 * math.pi)
                    speed_p = random.uniform(4, 12)
                    self.particles.append({
                        'x': fruit.x,
                        'y': fruit.y,
                        'vx': math.cos(angle) * speed_p,
                        'vy': math.sin(angle) * speed_p - 6,
                        'life': 25,
                        'color': fruit.props['colors'][random.randint(0, 2)],
                        'size': random.uniform(3, 8)
                    })
                
                # Juice splatter effect
                for _ in range(8):
                    angle = random.uniform(0, 2 * math.pi)
                    speed_p = random.uniform(2, 6)
                    self.particles.append({
                        'x': fruit.x,
                        'y': fruit.y,
                        'vx': math.cos(angle) * speed_p,
                        'vy': math.sin(angle) * speed_p - 3,
                        'life': 15,
                        'color': fruit.props['glow'],
                        'size': random.uniform(5, 12)
                    })
                
                self.slash_effects.append({
                    'x': fruit.x,
                    'y': fruit.y,
                    'time': 0,
                    'angle': fruit.slice_angle,
                    'color': fruit.props['glow']
                })
                
                sliced_any = True
        
        if sliced_any:
            self.combo += sliced_count
            points = int(10 * sliced_count * (1 + self.combo * 0.15))
            self.score += points
            self.combo_timer = 0
            self.combo_flash = 5
            self.max_combo = max(self.max_combo, self.combo)
            
            # Score popup
            self.score_popups.append({
                'x': x,
                'y': y,
                'points': f'+{points}',
                'life': 30,
                'vy': -3
            })
            
            self.calculate_difficulty()
            self.update_score()
        
        return sliced_any
    
    def explode_bomb(self, x, y):
        """Create explosion effect"""
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(5, 15)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 8,
                'life': 30,
                'color': random.choice(['#FF4500', '#FF6347', '#FFD700', '#FFA500']),
                'size': random.uniform(4, 10)
            })
    
    def spawn_fruit(self):
        fruit_type = random.choice(self.fruit_types)
        x = random.randint(80, 420)
        fruit = Fruit(x, 500, fruit_type, 500, self.difficulty_multiplier)
        self.fruits.append(fruit)
    
    def spawn_bomb(self):
        x = random.randint(100, 400)
        bomb = Bomb(x, 500, 500, self.difficulty_multiplier)
        self.bombs.append(bomb)
    
    def update_fruits(self):
        fruits_to_remove = []
        
        for fruit in self.fruits:
            fruit.update()
            
            if fruit.is_off_screen():
                if not fruit.sliced and fruit.y > 500:
                    self.lives -= 1
                    self.combo = 0
                    self.screen_shake = 8
                    self.update_score()
                    
                    if self.lives <= 0:
                        self.end_game()
                
                fruits_to_remove.append(fruit)
        
        for fruit in fruits_to_remove:
            self.fruits.remove(fruit)
        
        # Update bombs
        bombs_to_remove = []
        for bomb in self.bombs:
            bomb.update()
            if bomb.is_off_screen():
                bombs_to_remove.append(bomb)
        
        for bomb in bombs_to_remove:
            self.bombs.remove(bomb)
        
        # Update effects
        self.slash_effects = [s for s in self.slash_effects if s['time'] < 20]
        for slash in self.slash_effects:
            slash['time'] += 1
        
        # Update particles
        for particle in self.particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.5
            particle['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Update score popups
        for popup in self.score_popups:
            popup['y'] += popup['vy']
            popup['vy'] += 0.1
            popup['life'] -= 1
        self.score_popups = [p for p in self.score_popups if p['life'] > 0]
        
        # Update background particles
        for bp in self.background_particles:
            bp['x'] += bp['vx']
            bp['y'] += bp['vy']
            
            if bp['y'] > 500:
                bp['y'] = 0
                bp['x'] = random.randint(0, 500)
            if bp['x'] < 0:
                bp['x'] = 500
            elif bp['x'] > 500:
                bp['x'] = 0
        
        # Combo timer
        self.combo_timer += 1
        if self.combo_timer > 60:
            self.combo = 0
            self.update_score()
        
        # Effects decay
        if self.screen_shake > 0:
            self.screen_shake -= 1
        if self.flash_effect > 0:
            self.flash_effect -= 1
        if self.combo_flash > 0:
            self.combo_flash -= 1
    
    def update_score(self):
        self.score_label.config(text=f"SCORE: {int(self.score)}")
        
        if self.combo > 1:
            combo_text = f"✨ x{self.combo} COMBO! ✨"
            combo_color = '#FF1744' if self.combo_flash > 0 else '#FFD700'
            self.combo_label.config(text=combo_text, fg=combo_color)
        else:
            self.combo_label.config(text="")
        
        crosses = "💀" * (3 - self.lives)
        hearts = "❤️" * self.lives
        self.lives_label.config(text=crosses + hearts)
    
    def draw_game(self):
        self.canvas.delete("all")
        
        # Apply screen shake
        shake_x = random.randint(-self.screen_shake, self.screen_shake) if self.screen_shake > 0 else 0
        shake_y = random.randint(-self.screen_shake, self.screen_shake) if self.screen_shake > 0 else 0
        
        # Draw background particles
        for bp in self.background_particles:
            size = bp['size']
            self.canvas.create_oval(
                bp['x'] - size + shake_x, bp['y'] - size + shake_y,
                bp['x'] + size + shake_x, bp['y'] + size + shake_y,
                fill=bp['color'],
                outline='',
                stipple='gray50'
            )
        
        # Flash effect
        if self.flash_effect > 0:
            alpha = self.flash_effect / 10
            self.canvas.create_rectangle(
                0, 0, 500, 500,
                fill='red',
                stipple='gray25'
            )
        
        # Enhanced trail with gradient
        for i, (x, y) in enumerate(self.trail):
            progress = (i + 1) / len(self.trail)
            size = 5 + progress * 25
            
            # Outer glow
            glow_size = size * 1.5
            self.canvas.create_oval(
                x - glow_size + shake_x, y - glow_size + shake_y,
                x + glow_size + shake_x, y + glow_size + shake_y,
                fill='#FF8C00',
                outline='',
                stipple='gray25'
            )
            
            # Main trail
            self.canvas.create_oval(
                x - size + shake_x, y - size + shake_y,
                x + size + shake_x, y + size + shake_y,
                fill='#FF4500',
                outline=''
            )
            
            # Core
            core_size = size * 0.5
            self.canvas.create_oval(
                x - core_size + shake_x, y - core_size + shake_y,
                x + core_size + shake_x, y + core_size + shake_y,
                fill='#FFD700',
                outline=''
            )
        
        # Draw particles
        for particle in self.particles:
            alpha = particle['life'] / 30
            size = particle['size'] * alpha
            self.canvas.create_oval(
                particle['x'] - size + shake_x, particle['y'] - size + shake_y,
                particle['x'] + size + shake_x, particle['y'] + size + shake_y,
                fill=particle['color'],
                outline=''
            )
        
        # Enhanced slash effects
        for slash in self.slash_effects:
            progress = slash['time'] / 20
            opacity = 1 - progress
            size = 100 + slash['time'] * 10
            
            x, y = slash['x'], slash['y']
            angle = slash['angle']
            
            length = size
            x1 = x - math.cos(angle) * length / 2
            y1 = y - math.sin(angle) * length / 2
            x2 = x + math.cos(angle) * length / 2
            y2 = y + math.sin(angle) * length / 2
            
            # Outer glow
            if opacity > 0.3:
                self.canvas.create_line(
                    x1 + shake_x, y1 + shake_y, x2 + shake_x, y2 + shake_y,
                    fill=slash.get('color', '#FFD700'),
                    width=int(20 * opacity),
                    capstyle=tk.ROUND
                )
            
            # Main slash
            self.canvas.create_line(
                x1 + shake_x, y1 + shake_y, x2 + shake_x, y2 + shake_y,
                fill='white',
                width=int(8 * opacity),
                capstyle=tk.ROUND
            )
        
        # Draw score popups
        for popup in self.score_popups:
            alpha = popup['life'] / 30
            self.canvas.create_text(
                popup['x'] + shake_x, popup['y'] + shake_y,
                text=popup['points'],
                font=("Impact", int(20 * (1 + (1-alpha)))),
                fill='#FFD700',
                anchor=tk.CENTER
            )
        
        # Draw bombs with warning effect
        for bomb in self.bombs:
            if bomb.sliced:
                continue
            
            pulse = math.sin(bomb.pulse) * 0.3 + 1.0
            size = 40 * pulse
            
            # Warning glow
            glow_size = size * 1.8
            self.canvas.create_oval(
                bomb.x - glow_size + shake_x, bomb.y - glow_size + shake_y,
                bomb.x + glow_size + shake_x, bomb.y + glow_size + shake_y,
                fill='#FF0000',
                outline='',
                stipple='gray25'
            )
            
            # Bomb body
            self.canvas.create_oval(
                bomb.x - size + shake_x, bomb.y - size + shake_y,
                bomb.x + size + shake_x, bomb.y + size + shake_y,
                fill='#2C2C2C',
                outline='#FF0000',
                width=3
            )
            
            # Skull symbol
            self.canvas.create_text(
                bomb.x + shake_x, bomb.y + shake_y,
                text="💣",
                font=("Arial", int(size * 1.2)),
                anchor=tk.CENTER
            )
            
            # Sparks
            if bomb.spark_timer % 5 == 0:
                for _ in range(3):
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(size, size * 1.5)
                    sx = bomb.x + math.cos(angle) * dist
                    sy = bomb.y + math.sin(angle) * dist
                    self.canvas.create_oval(
                        sx - 3 + shake_x, sy - 3 + shake_y,
                        sx + 3 + shake_x, sy + 3 + shake_y,
                        fill='#FFA500',
                        outline=''
                    )
        
        # Draw fruits with enhanced effects
        for fruit in self.fruits:
            if fruit.sliced:
                offset = fruit.slice_time * 5
                angle_offset = fruit.slice_angle
                
                x_off = -math.cos(angle_offset) * offset
                y_off = -math.sin(angle_offset) * offset
                self.draw_fruit_half(fruit, x_off, y_off, 'left', shake_x, shake_y)
                
                x_off = math.cos(angle_offset) * offset
                y_off = math.sin(angle_offset) * offset
                self.draw_fruit_half(fruit, x_off, y_off, 'right', shake_x, shake_y)
            else:
                self.draw_whole_fruit(fruit, shake_x, shake_y)
    
    def draw_whole_fruit(self, fruit, shake_x=0, shake_y=0):
        size = fruit.props['size']
        colors = fruit.props['colors']
        pulse = math.sin(fruit.pulse) * 0.1 + 1.0
        current_size = size * pulse
        
        # Draw specific fruit type (no circular background)
        if fruit.fruit_type == 'apple':
            self.draw_apple(fruit, current_size, colors, shake_x, shake_y)
        elif fruit.fruit_type == 'orange':
            self.draw_orange(fruit, current_size, colors, shake_x, shake_y)
        elif fruit.fruit_type == 'watermelon':
            self.draw_watermelon(fruit, current_size, colors, shake_x, shake_y)
        elif fruit.fruit_type == 'kiwi':
            self.draw_kiwi(fruit, current_size, colors, shake_x, shake_y)
        elif fruit.fruit_type == 'strawberry':
            self.draw_strawberry(fruit, current_size, colors, shake_x, shake_y)
    
    def draw_apple(self, fruit, size, colors, shake_x, shake_y):
        x, y = fruit.x + shake_x, fruit.y + shake_y
        
        # Apple body
        self.canvas.create_oval(
            x - size/2, y - size/2.5,
            x + size/2, y + size/2,
            fill=colors[0], outline=colors[2], width=3
        )
        
        # Top indent
        self.canvas.create_oval(
            x - size/6, y - size/2.5,
            x + size/6, y - size/4,
            fill='#6B4423', outline=''
        )
        
        # Stem
        self.canvas.create_rectangle(
            x - size/20, y - size/1.8,
            x + size/20, y - size/3,
            fill='#5D4037', outline=''
        )
        
        # Leaf
        leaf_points = [
            x + size/20, y - size/2,
            x + size/4, y - size/2.3,
            x + size/5, y - size/3
        ]
        self.canvas.create_polygon(leaf_points, fill='#4CAF50', outline='#2E7D32', width=2)
        
        # Highlight
        self.canvas.create_oval(
            x - size/3, y - size/3,
            x - size/6, y - size/6,
            fill='#FF6B6B', outline=''
        )
    
    
    
    def draw_orange(self, fruit, size, colors, shake_x, shake_y):
        x, y = fruit.x + shake_x, fruit.y + shake_y
        
        # Orange body
        self.canvas.create_oval(
            x - size/2, y - size/2,
            x + size/2, y + size/2,
            fill=colors[0], outline=colors[2], width=3
        )
        
        # Texture dots for orange peel
        for i in range(15):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, size/3)
            dot_x = x + math.cos(angle) * dist
            dot_y = y + math.sin(angle) * dist
            dot_size = size/30
            self.canvas.create_oval(
                dot_x - dot_size, dot_y - dot_size,
                dot_x + dot_size, dot_y + dot_size,
                fill='#CC5500', outline=''
            )
        
        # Top stem area
        self.canvas.create_oval(
            x - size/8, y - size/2.2,
            x + size/8, y - size/3,
            fill='#8B4513', outline=''
        )
        
        # Highlight
        self.canvas.create_oval(
            x - size/3, y - size/3,
            x - size/6, y - size/6,
            fill='#FFB347', outline=''
        )
    
    def draw_watermelon(self, fruit, size, colors, shake_x, shake_y):
        x, y = fruit.x + shake_x, fruit.y + shake_y
        
        # Watermelon body (slightly oval)
        self.canvas.create_oval(
            x - size/2, y - size/2.3,
            x + size/2, y + size/2.3,
            fill=colors[0], outline='#004D00', width=4
        )
        
        # Dark green stripes
        for i in range(5):
            stripe_x = x - size/2.5 + i * size/5
            points = [
                stripe_x, y - size/2.3,
                stripe_x + size/15, y - size/2.3,
                stripe_x + size/10, y + size/2.3,
                stripe_x - size/20, y + size/2.3
            ]
            self.canvas.create_polygon(points, fill='#004D00', outline='', smooth=True)
        
        # Highlight
        self.canvas.create_oval(
            x - size/3, y - size/3,
            x - size/6, y - size/6,
            fill='#5FD35F', outline=''
        )
    
    def draw_kiwi(self, fruit, size, colors, shake_x, shake_y):
        x, y = fruit.x + shake_x, fruit.y + shake_y
        
        # Kiwi body (slightly oval)
        self.canvas.create_oval(
            x - size/2.2, y - size/2,
            x + size/2.2, y + size/2,
            fill=colors[0], outline=colors[2], width=3
        )
        
        # Fuzzy texture
        for i in range(20):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(size/3, size/2.3)
            fuzz_x = x + math.cos(angle) * dist
            fuzz_y = y + math.sin(angle) * dist
            self.canvas.create_line(
                fuzz_x, fuzz_y,
                fuzz_x + random.uniform(-3, 3), fuzz_y + random.uniform(-3, 3),
                fill='#5D4E37', width=2
            )
        
        # Highlight
        self.canvas.create_oval(
            x - size/4, y - size/4,
            x - size/8, y - size/8,
            fill='#B5E550', outline=''
        )
    
    def draw_strawberry(self, fruit, size, colors, shake_x, shake_y):
        x, y = fruit.x + shake_x, fruit.y + shake_y
        
        # Strawberry body (heart shape)
        points = [
            x, y + size/2,
            x - size/2, y,
            x - size/4, y - size/3,
            x, y - size/4,
            x + size/4, y - size/3,
            x + size/2, y,
        ]
        self.canvas.create_polygon(points, fill=colors[0], outline=colors[2], width=3, smooth=True)
        
        # Seeds
        for i in range(12):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, size/3)
            seed_x = x + math.cos(angle) * dist
            seed_y = y + math.sin(angle) * dist
            self.canvas.create_oval(
                seed_x - size/25, seed_y - size/25,
                seed_x + size/25, seed_y + size/25,
                fill='#FFD700', outline=''
            )
        
        # Leaves on top
        for i in range(3):
            leaf_angle = -math.pi/2 + (i - 1) * math.pi/4
            leaf_x = x + math.cos(leaf_angle) * size/6
            leaf_y = y - size/3
            leaf_points = [
                leaf_x, leaf_y,
                leaf_x + size/8, leaf_y - size/6,
                leaf_x, leaf_y - size/10
            ]
            self.canvas.create_polygon(leaf_points, fill='#4CAF50', outline='#2E7D32', width=2)
        
        # Highlight
        self.canvas.create_oval(
            x - size/4, y - size/8,
            x - size/8, y + size/8,
            fill='#FF8FAB', outline=''
        )
    
    def draw_fruit_half(self, fruit, x_off, y_off, half, shake_x=0, shake_y=0):
        size = fruit.props['size']
        colors = fruit.props['colors']
        
        x = fruit.x + x_off + shake_x
        y = fruit.y + y_off + shake_y
        
        rotation_offset = fruit.slice_time * 10
        
        if half == 'left':
            start_angle = 90 + rotation_offset
            self.canvas.create_arc(
                x - size/2, y - size/2,
                x + size/2, y + size/2,
                start=start_angle, extent=180,
                fill=colors[0],
                outline=colors[2],
                width=3
            )
        else:
            start_angle = 270 - rotation_offset
            self.canvas.create_arc(
                x - size/2, y - size/2,
                x + size/2, y + size/2,
                start=start_angle, extent=180,
                fill=colors[0],
                outline=colors[2],
                width=3
            )
    
    def game_loop(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            processed_frame, finger_tip, method = self.detect_finger_simple(frame)
            
            if method == "MediaPipe":
                self.status_label.config(text="🟢 Tracking Active!", fg='#4CAF50')
            else:
                self.status_label.config(text="🔴 No Hand Detected", fg='#FF5252')
            
            small_frame = cv2.resize(processed_frame, (320, 240))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            self.webcam_canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.webcam_canvas.image = img
            
            if finger_tip:
                x, y = finger_tip
                
                scale_x = 500 / frame.shape[1]
                scale_y = 500 / frame.shape[0]
                x = int(x * scale_x)
                y = int(y * scale_y)
                
                x = max(0, min(x, 500))
                y = max(0, min(y, 500))
                
                self.prev_finger_pos = self.finger_pos
                self.finger_pos = (x, y)
                
                self.trail.append((x, y))
                
                if self.prev_finger_pos:
                    self.check_collisions(x, y)
        
        # Dynamic spawn rates based on difficulty
        self.spawn_timer += 1
        if self.spawn_timer >= self.get_spawn_rate():
            self.spawn_fruit()
            if random.random() < 0.3 and self.level > 1:  # Multiple fruits
                self.spawn_fruit()
            self.spawn_timer = 0
        
        # Bomb spawning (increases with level)
        if self.level >= 2:
            self.bomb_spawn_timer += 1
            if self.bomb_spawn_timer >= self.get_bomb_spawn_rate():
                self.spawn_bomb()
                self.bomb_spawn_timer = 0
        
        self.update_fruits()
        self.draw_game()
        
        if self.game_started and not self.game_over:
            self.root.after(30, self.game_loop)
    
    def end_game(self):
        self.game_over = True
        self.game_started = False
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        
        if self.score > 1000:
            rank = "🏆 LEGENDARY NINJA"
        elif self.score > 750:
            rank = "⭐ MASTER NINJA"
        elif self.score > 500:
            rank = "💎 EXPERT NINJA"
        elif self.score > 300:
            rank = "⚔️ SKILLED NINJA"
        elif self.score > 150:
            rank = "💪 NINJA"
        else:
            rank = "🎯 BEGINNER"
        
        messagebox.showinfo(
            "⚔️ GAME OVER ⚔️",
            f"Final Score: {int(self.score)} points\n\n"
            f"Max Level: {self.level}\n"
            f"Max Combo: {self.max_combo}x\n\n"
            f"Rank: {rank}\n\n"
            f"Click START GAME to play again!"
        )
    
    def cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitNinjaApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
