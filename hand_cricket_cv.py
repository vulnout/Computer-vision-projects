import cv2
import mediapipe as mp
import random
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List

class GameMode(Enum):
    SINGLE_PLAYER = "Single Player"
    TWO_PLAYER = "Two Player"

class GamePhase(Enum):
    MENU = 0
    FIRST_INNINGS = 1
    INNINGS_BREAK = 2
    SECOND_INNINGS = 3
    MATCH_COMPLETE = 4

class GestureType(Enum):
    ZERO = (0, "Zero")
    ONE = (1, "One")
    TWO = (2, "Two")
    THREE = (3, "Three")
    FOUR = (4, "Four")
    FIVE = (5, "Five")
    SIX = (6, "Six")
    UNKNOWN = (-1, "Unknown")
    
    def __init__(self, value, display_name):
        self._value_ = value
        self.display_name = display_name

class Config:
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    FLIP_HORIZONTAL = True
    
    MAX_HANDS = 1
    DETECTION_CONFIDENCE = 0.75
    TRACKING_CONFIDENCE = 0.75
    
    DEFAULT_OVERS = 1
    MIN_OVERS = 1
    MAX_OVERS = 10
    BALLS_PER_OVER = 6
    
    ENABLE_ANIMATIONS = True
    BALL_ANIMATION_FRAMES = 15
    BALL_ANIMATION_DELAY_MS = 25
    WICKET_ANIMATION_FRAMES = 20
    
    COLOR_PRIMARY = (50, 200, 50)
    COLOR_SECONDARY = (200, 150, 50)
    COLOR_DANGER = (50, 50, 255)
    COLOR_INFO = (255, 200, 100)
    COLOR_BACKGROUND = (20, 20, 20)
    COLOR_PANEL = (40, 40, 40)
    COLOR_TEXT = (240, 240, 240)
    COLOR_ACCENT = (0, 180, 255)
    
    AI_RANDOM_SEED = None
    AI_SMART_MODE = True

@dataclass
class PlayerStats:
    name: str
    runs: int = 0
    balls_faced: int = 0
    is_out: bool = False
    boundaries: int = 0
    dots: int = 0
    
    def add_runs(self, runs: int):
        self.runs += runs
        self.balls_faced += 1
        if runs in [4, 6]:
            self.boundaries += 1
        elif runs == 0:
            self.dots += 1
    
    def get_strike_rate(self) -> float:
        if self.balls_faced == 0:
            return 0.0
        return (self.runs / self.balls_faced) * 100
    
    def reset(self):
        self.runs = 0
        self.balls_faced = 0
        self.is_out = False
        self.boundaries = 0
        self.dots = 0

@dataclass
class InningsData:
    batting_team: str
    bowling_team: str
    total_runs: int = 0
    total_balls: int = 0
    wickets: int = 0
    target: Optional[int] = None
    ball_by_ball: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.ball_by_ball is None:
            self.ball_by_ball = []
    
    def add_ball(self, batter_score: int, bowler_score: int):
        self.ball_by_ball.append((batter_score, bowler_score))
        self.total_balls += 1
        self.total_runs += batter_score
    
    def get_overs(self) -> str:
        overs = self.total_balls // 6
        balls = self.total_balls % 6
        return f"{overs}.{balls}"
    
    def get_run_rate(self) -> float:
        if self.total_balls == 0:
            return 0.0
        overs_decimal = self.total_balls / 6.0
        return self.total_runs / overs_decimal if overs_decimal > 0 else 0.0

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=Config.MAX_HANDS,
            min_detection_confidence=Config.DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.TRACKING_CONFIDENCE
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[GestureType], Optional[any]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            
            gesture = self._classify_gesture(hand_landmarks.landmark, handedness)
            return gesture, hand_landmarks
        
        return None, None
    
    def _classify_gesture(self, landmarks, handedness: str) -> GestureType:
        if not landmarks:
            return GestureType.UNKNOWN
        
        fingers_up = []
        
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        
        if handedness == "Right":
            thumb_up = thumb_tip.x < thumb_ip.x
        else:
            thumb_up = thumb_tip.x > thumb_ip.x
        
        fingers_up.append(thumb_up)
        
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            finger_up = landmarks[tip_idx].y < landmarks[pip_idx].y
            fingers_up.append(finger_up)
        
        total_fingers = sum(fingers_up)
        
        if fingers_up[0] and not any(fingers_up[1:]):
            return GestureType.SIX
        
        gesture_map = {
            0: GestureType.ZERO,
            1: GestureType.ONE,
            2: GestureType.TWO,
            3: GestureType.THREE,
            4: GestureType.FOUR,
            5: GestureType.FIVE
        }
        
        return gesture_map.get(total_fingers, GestureType.UNKNOWN)
    
    def draw_landmarks(self, frame: np.ndarray, landmarks):
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, 
                landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 100, 0), thickness=2)
            )

class SmartAI:
    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.player_pattern = []
        self.last_throws = []
        self.pattern_window = 5
        
    def get_throw(self, context: dict) -> int:
        phase = context.get("phase")
        runs_needed = context.get("runs_needed", 0)
        balls_left = context.get("balls_left", 30)
        
        if Config.AI_SMART_MODE and len(self.player_pattern) >= 3:
            return self._smart_throw(phase, runs_needed, balls_left)
        else:
            return self._random_throw(phase, runs_needed, balls_left)
    
    def _smart_throw(self, phase: GamePhase, runs_needed: int, balls_left: int) -> int:
        if phase == GamePhase.FIRST_INNINGS:
            if len(self.player_pattern) >= 2:
                recent = self.player_pattern[-5:]
                if recent:
                    from collections import Counter
                    most_common = Counter(recent).most_common(1)[0]
                    if most_common[1] >= 2:
                        if random.random() < 0.4:
                            return most_common[0]
            
            return random.randint(0, 6)
        
        else:
            required_rr = (runs_needed / balls_left) * 6 if balls_left > 0 else 0
            
            if runs_needed <= 0:
                return 0
            
            if balls_left <= 3 and runs_needed <= 6:
                return min(6, runs_needed)
            
            if required_rr > 10:
                return random.choice([4, 5, 6])
            elif required_rr > 6:
                return random.choice([2, 3, 4, 5])
            else:
                return random.choice([1, 2, 3, 4])
    
    def _random_throw(self, phase: GamePhase, runs_needed: int, balls_left: int) -> int:
        if phase == GamePhase.SECOND_INNINGS and runs_needed > 0:
            weights = [5, 15, 20, 20, 20, 15, 5]
            return random.choices(range(7), weights=weights)[0]
        return random.randint(0, 6)
    
    def update_player_pattern(self, throw: int):
        self.player_pattern.append(throw)
        if len(self.player_pattern) > 20:
            self.player_pattern.pop(0)

class CricketGame:
    def __init__(self, mode: GameMode, overs: int):
        self.mode = mode
        self.overs = overs
        self.max_balls = overs * Config.BALLS_PER_OVER
        
        self.phase = GamePhase.FIRST_INNINGS
        self.current_over_balls = []
        
        self.player1 = PlayerStats("Player 1" if mode == GameMode.TWO_PLAYER else "You")
        self.player2 = PlayerStats("Player 2" if mode == GameMode.TWO_PLAYER else "Computer")
        
        self.innings1 = InningsData(self.player1.name, self.player2.name)
        self.innings2 = InningsData(self.player2.name, self.player1.name)
        
        self.ai = SmartAI()
        
        self.waiting_for_player2 = False
        self.player1_throw = None
        
        self.last_result = ""
        self.status_message = "Press SPACE to start"
        
    def process_ball(self, player_gesture: GestureType) -> dict:
        if player_gesture == GestureType.UNKNOWN:
            return {"success": False, "message": "Invalid gesture detected"}
        
        player_value = player_gesture.value
        
        if self.mode == GameMode.SINGLE_PLAYER:
            return self._process_single_player(player_value)
        else:
            return self._process_two_player(player_value)
    
    def _process_single_player(self, player_value: int) -> dict:
        context = {
            "phase": self.phase,
            "runs_needed": self.innings1.total_runs + 1 - self.innings2.total_runs,
            "balls_left": self.max_balls - self.innings2.total_balls
        }
        ai_value = self.ai.get_throw(context)
        
        if self.phase == GamePhase.FIRST_INNINGS:
            self.ai.update_player_pattern(player_value)
            
            if player_value == ai_value:
                self.player1.is_out = True
                self.innings1.add_ball(0, ai_value)
                self.innings1.wickets = 1
                self.innings2.target = self.innings1.total_runs + 1
                self.phase = GamePhase.INNINGS_BREAK
                
                return {
                    "success": True,
                    "wicket": True,
                    "player_throw": player_value,
                    "opponent_throw": ai_value,
                    "message": f"OUT! You: {player_value}, Computer: {ai_value}",
                    "innings_complete": True
                }
            else:
                self.player1.add_runs(player_value)
                self.innings1.add_ball(player_value, ai_value)
                
                if self.innings1.total_balls >= self.max_balls:
                    self.innings2.target = self.innings1.total_runs + 1
                    self.phase = GamePhase.INNINGS_BREAK
                    innings_complete = True
                else:
                    innings_complete = False
                
                return {
                    "success": True,
                    "wicket": False,
                    "player_throw": player_value,
                    "opponent_throw": ai_value,
                    "runs": player_value,
                    "message": f"{player_value} run{'s' if player_value != 1 else ''}! You: {player_value}, Computer: {ai_value}",
                    "innings_complete": innings_complete
                }
        
        elif self.phase == GamePhase.SECOND_INNINGS:
            self.ai.update_player_pattern(player_value)
            
            if player_value == ai_value:
                self.player2.is_out = True
                self.innings2.add_ball(0, player_value)
                self.innings2.wickets = 1
                self.phase = GamePhase.MATCH_COMPLETE
                
                winner = self.player1.name if self.innings2.total_runs < self.innings2.target else self.player2.name
                
                return {
                    "success": True,
                    "wicket": True,
                    "player_throw": player_value,
                    "opponent_throw": ai_value,
                    "message": f"WICKET! You: {player_value}, Computer: {ai_value}",
                    "match_complete": True,
                    "winner": winner
                }
            else:
                self.player2.add_runs(ai_value)
                self.innings2.add_ball(ai_value, player_value)
                
                match_complete = False
                winner = None
                
                if self.innings2.total_runs >= self.innings2.target:
                    self.phase = GamePhase.MATCH_COMPLETE
                    match_complete = True
                    winner = self.player2.name
                elif self.innings2.total_balls >= self.max_balls:
                    self.phase = GamePhase.MATCH_COMPLETE
                    match_complete = True
                    winner = self.player1.name if self.innings2.total_runs < self.innings2.target else self.player2.name
                
                return {
                    "success": True,
                    "wicket": False,
                    "player_throw": player_value,
                    "opponent_throw": ai_value,
                    "runs": ai_value,
                    "message": f"Computer scores {ai_value}! You: {player_value}, Computer: {ai_value}",
                    "match_complete": match_complete,
                    "winner": winner
                }
        
        return {"success": False}
    
    def _process_two_player(self, player_value: int) -> dict:
        if not self.waiting_for_player2:
            self.player1_throw = player_value
            self.waiting_for_player2 = True
            return {
                "success": True,
                "waiting": True,
                "message": f"Player 1 showed {player_value}. Player 2's turn..."
            }
        else:
            player2_throw = player_value
            self.waiting_for_player2 = False
            
            if self.phase == GamePhase.FIRST_INNINGS:
                if self.player1_throw == player2_throw:
                    self.player1.is_out = True
                    self.innings1.add_ball(0, player2_throw)
                    self.innings1.wickets = 1
                    self.innings2.target = self.innings1.total_runs + 1
                    self.phase = GamePhase.INNINGS_BREAK
                    
                    return {
                        "success": True,
                        "wicket": True,
                        "player_throw": self.player1_throw,
                        "opponent_throw": player2_throw,
                        "message": f"OUT! P1: {self.player1_throw}, P2: {player2_throw}",
                        "innings_complete": True
                    }
                else:
                    self.player1.add_runs(self.player1_throw)
                    self.innings1.add_ball(self.player1_throw, player2_throw)
                    
                    innings_complete = self.innings1.total_balls >= self.max_balls
                    if innings_complete:
                        self.innings2.target = self.innings1.total_runs + 1
                        self.phase = GamePhase.INNINGS_BREAK
                    
                    return {
                        "success": True,
                        "wicket": False,
                        "player_throw": self.player1_throw,
                        "opponent_throw": player2_throw,
                        "runs": self.player1_throw,
                        "message": f"P1 scores {self.player1_throw}! P1: {self.player1_throw}, P2: {player2_throw}",
                        "innings_complete": innings_complete
                    }
            
            elif self.phase == GamePhase.SECOND_INNINGS:
                if self.player1_throw == player2_throw:
                    self.player2.is_out = True
                    self.innings2.add_ball(0, self.player1_throw)
                    self.innings2.wickets = 1
                    self.phase = GamePhase.MATCH_COMPLETE
                    
                    winner = self.player1.name if self.innings2.total_runs < self.innings2.target else self.player2.name
                    
                    return {
                        "success": True,
                        "wicket": True,
                        "player_throw": self.player1_throw,
                        "opponent_throw": player2_throw,
                        "message": f"WICKET! P1: {self.player1_throw}, P2: {player2_throw}",
                        "match_complete": True,
                        "winner": winner
                    }
                else:
                    self.player2.add_runs(player2_throw)
                    self.innings2.add_ball(player2_throw, self.player1_throw)
                    
                    match_complete = False
                    winner = None
                    
                    if self.innings2.total_runs >= self.innings2.target:
                        self.phase = GamePhase.MATCH_COMPLETE
                        match_complete = True
                        winner = self.player2.name
                    elif self.innings2.total_balls >= self.max_balls:
                        self.phase = GamePhase.MATCH_COMPLETE
                        match_complete = True
                        winner = self.player1.name if self.innings2.total_runs < self.innings2.target else self.player2.name
                    
                    return {
                        "success": True,
                        "wicket": False,
                        "player_throw": self.player1_throw,
                        "opponent_throw": player2_throw,
                        "runs": player2_throw,
                        "message": f"P2 scores {player2_throw}! P1: {self.player1_throw}, P2: {player2_throw}",
                        "match_complete": match_complete,
                        "winner": winner
                    }
    
    def start_second_innings(self):
        self.phase = GamePhase.SECOND_INNINGS
        self.status_message = f"Chase {self.innings2.target} to win!"
    
    def reset(self):
        self.phase = GamePhase.FIRST_INNINGS
        self.player1.reset()
        self.player2.reset()
        self.innings1 = InningsData(self.player1.name, self.player2.name)
        self.innings2 = InningsData(self.player2.name, self.player1.name)
        self.waiting_for_player2 = False
        self.player1_throw = None
        self.last_result = ""
        self.status_message = "Game reset - Press SPACE to play"

class UIRenderer:
    @staticmethod
    def draw_gradient_rect(frame, pt1, pt2, color1, color2):
        x1, y1 = pt1
        x2, y2 = pt2
        
        for i in range(y1, y2):
            alpha = (i - y1) / (y2 - y1)
            color = tuple(int(c1 * (1 - alpha) + c2 * alpha) for c1, c2 in zip(color1, color2))
            cv2.line(frame, (x1, i), (x2, i), color, 1)
    
    @staticmethod
    def draw_panel(frame, x, y, w, h, title=None, color=Config.COLOR_PANEL):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), Config.COLOR_ACCENT, 2)
        
        if title:
            cv2.rectangle(frame, (x, y), (x + w, y + 35), Config.COLOR_ACCENT, -1)
            cv2.putText(frame, title, (x + 10, y + 24), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    @staticmethod
    def draw_result_highlight(frame, message: str, y: int):
        h, w = frame.shape[:2]
        
        # Extract the "You: X, Computer: Y" part if it exists
        if "You:" in message or "P1:" in message or "P2:" in message:
            parts = message.split("!")
            if len(parts) >= 2:
                throw_info = parts[-1].strip()
                result_text = parts[0] + "!"
            else:
                throw_info = message
                result_text = ""
            
            # Create a highlighted panel
            panel_w = 400
            panel_h = 80
            panel_x = (w - panel_w) // 2
            panel_y = y
            
            # Draw panel with gradient effect
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                         Config.COLOR_ACCENT, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                         Config.COLOR_ACCENT, 3)
            
            # Draw the main result message
            if result_text:
                text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = panel_x + (panel_w - text_size[0]) // 2
                cv2.putText(frame, result_text, (text_x, panel_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_PRIMARY, 2)
            
            # Draw the throw info
            text_size = cv2.getTextSize(throw_info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = panel_x + (panel_w - text_size[0]) // 2
            text_y = panel_y + 60 if result_text else panel_y + 45
            cv2.putText(frame, throw_info, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    @staticmethod
    def draw_scoreboard(frame, game: CricketGame, x, y):
        w, h = 380, 280
        UIRenderer.draw_panel(frame, x, y, w, h, "SCOREBOARD")
        
        cy = y + 50
        
        innings1_text = f"{game.innings1.batting_team}: {game.innings1.total_runs}/{game.innings1.wickets}"
        cv2.putText(frame, innings1_text, (x + 15, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, Config.COLOR_PRIMARY, 2)
        
        overs1 = game.innings1.get_overs()
        rr1 = game.innings1.get_run_rate()
        cv2.putText(frame, f"Overs: {overs1}  RR: {rr1:.2f}", (x + 15, cy + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXT, 1)
        
        cy += 70
        
        innings2_text = f"{game.innings2.batting_team}: {game.innings2.total_runs}/{game.innings2.wickets}"
        cv2.putText(frame, innings2_text, (x + 15, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, Config.COLOR_SECONDARY, 2)
        
        overs2 = game.innings2.get_overs()
        rr2 = game.innings2.get_run_rate()
        cv2.putText(frame, f"Overs: {overs2}  RR: {rr2:.2f}", (x + 15, cy + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXT, 1)
        
        cy += 70
        
        if game.phase == GamePhase.SECOND_INNINGS:
            target = game.innings2.target
            needed = target - game.innings2.total_runs
            balls_left = game.max_balls - game.innings2.total_balls
            required_rr = (needed / balls_left * 6) if balls_left > 0 else 0
            
            cv2.putText(frame, f"Target: {target}", (x + 15, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_INFO, 2)
            cv2.putText(frame, f"Need {needed} from {balls_left} balls", (x + 15, cy + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXT, 1)
            cv2.putText(frame, f"Req RR: {required_rr:.2f}", (x + 15, cy + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_ACCENT, 1)
        elif game.innings2.target:
            cv2.putText(frame, f"Target: {game.innings2.target}", (x + 15, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_INFO, 2)
    
    @staticmethod
    def draw_player_stats(frame, player: PlayerStats, x, y, is_batting: bool):
        w, h = 280, 200
        title = f"{player.name} {'(Batting)' if is_batting else '(Bowling)'}"
        UIRenderer.draw_panel(frame, x, y, w, h, title)
        
        cy = y + 50
        
        cv2.putText(frame, f"Runs: {player.runs}", (x + 15, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_TEXT, 2)
        
        cv2.putText(frame, f"Balls: {player.balls_faced}", (x + 15, cy + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 1)
        
        sr = player.get_strike_rate()
        cv2.putText(frame, f"Strike Rate: {sr:.1f}", (x + 15, cy + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 1)
        
        cv2.putText(frame, f"Boundaries: {player.boundaries}", (x + 15, cy + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 1)
        
        cv2.putText(frame, f"Dots: {player.dots}", (x + 15, cy + 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 1)
    
    @staticmethod
    def draw_gesture_display(frame, gesture: GestureType, x, y, is_stable: bool):
        w, h = 280, 150
        UIRenderer.draw_panel(frame, x, y, w, h, "DETECTED")
        
        if gesture and gesture != GestureType.UNKNOWN:
            color = Config.COLOR_PRIMARY
            
            gesture_text = str(gesture.value)
            text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_DUPLEX, 3.0, 4)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + 90
            
            cv2.putText(frame, gesture_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 3.0, color, 4)
            
            name_text = gesture.display_name
            name_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            name_x = x + (w - name_size[0]) // 2
            cv2.putText(frame, name_text, (name_x, text_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 1)
            
            cv2.circle(frame, (x + w - 30, y + 70), 8, Config.COLOR_PRIMARY, -1)
            cv2.putText(frame, "READY", (x + w - 80, y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_PRIMARY, 1)
        else:
            no_hand_text = "Show Hand"
            text_size = cv2.getTextSize(no_hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            cv2.putText(frame, no_hand_text, (text_x, y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_DANGER, 2)
    
    @staticmethod
    def draw_status_bar(frame, message: str, y: int):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, y), (w, y + 50), Config.COLOR_BACKGROUND, -1)
        cv2.rectangle(frame, (0, y), (w, y + 50), Config.COLOR_ACCENT, 2)
        
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, message, (text_x, y + 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_TEXT, 2)
    
    @staticmethod
    def draw_controls(frame, x, y):
        controls = [
            "SPACE - Submit Throw",
            "R - Reset Match",
            "M - Change Mode",
            "ESC - Quit"
        ]
        
        cy = y
        for i, control in enumerate(controls):
            color = Config.COLOR_INFO if i % 2 == 0 else Config.COLOR_SECONDARY
            cv2.putText(frame, control, (x, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cy += 25
    
    @staticmethod
    def draw_match_result(frame, game: CricketGame):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        if game.phase == GamePhase.MATCH_COMPLETE:
            if game.innings2.total_runs >= game.innings2.target:
                winner = game.innings2.batting_team
                runs_diff = game.innings2.target - game.innings2.total_runs
                by_text = f"by reaching the target"
            else:
                winner = game.innings1.batting_team
                runs_diff = game.innings2.target - game.innings2.total_runs - 1
                by_text = f"by {runs_diff} runs"
            
            winner_text = f"{winner} WINS!"
            
            text_size = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_DUPLEX, 2.5, 5)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, winner_text, (text_x, h // 2 - 50),
                       cv2.FONT_HERSHEY_DUPLEX, 2.5, Config.COLOR_PRIMARY, 5)
            
            margin_size = cv2.getTextSize(by_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            margin_x = (w - margin_size[0]) // 2
            cv2.putText(frame, by_text, (margin_x, h // 2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, Config.COLOR_SECONDARY, 2)
            
            summary_y = h // 2 + 80
            summary = [
                f"{game.innings1.batting_team}: {game.innings1.total_runs}/{game.innings1.wickets} ({game.innings1.get_overs()} overs)",
                f"{game.innings2.batting_team}: {game.innings2.total_runs}/{game.innings2.wickets} ({game.innings2.get_overs()} overs)"
            ]
            
            for line in summary:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, line, (text_x, summary_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_TEXT, 2)
                summary_y += 40
            
            inst_text = "Press 'R' to play again"
            inst_size = cv2.getTextSize(inst_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            inst_x = (w - inst_size[0]) // 2
            cv2.putText(frame, inst_text, (inst_x, h - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, Config.COLOR_ACCENT, 2)
    
    @staticmethod
    def draw_innings_break(frame, game: CricketGame):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 50), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cy = h // 2 - 120
        
        title = "INNINGS BREAK"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.0, 4)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, title, (text_x, cy),
                   cv2.FONT_HERSHEY_DUPLEX, 2.0, Config.COLOR_INFO, 4)
        
        cy += 80
        
        summary = [
            f"{game.innings1.batting_team} scored:",
            f"{game.innings1.total_runs}/{game.innings1.wickets} in {game.innings1.get_overs()} overs",
            f"Run Rate: {game.innings1.get_run_rate():.2f}",
            "",
            f"{game.innings2.batting_team} needs {game.innings2.target} to win",
        ]
        
        for line in summary:
            if line:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                text_x = (w - text_size[0]) // 2
                font_weight = 3 if "needs" in line else 2
                color = Config.COLOR_PRIMARY if "needs" in line else Config.COLOR_TEXT
                cv2.putText(frame, line, (text_x, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, font_weight)
            cy += 45
        
        cy += 20
        inst_text = "Press SPACE to continue"
        inst_size = cv2.getTextSize(inst_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        inst_x = (w - inst_size[0]) // 2
        
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(frame, inst_text, (inst_x, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_ACCENT, 2)

class Animator:
    @staticmethod
    def animate_ball_delivery(frame, start_pos, end_pos, is_wicket=False) -> bool:
        if not Config.ENABLE_ANIMATIONS:
            return False
        
        frames = Config.BALL_ANIMATION_FRAMES
        
        for i in range(frames):
            t = i / (frames - 1)
            t_eased = t * t * (3 - 2 * t)
            
            x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * t_eased)
            arc_height = 80 * np.sin(t * np.pi)
            y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * t_eased - arc_height)
            
            ball_radius = int(12 + 6 * t)
            
            temp = frame.copy()
            
            shadow_y = end_pos[1] + 20
            shadow_radius = int(ball_radius * 0.6)
            cv2.ellipse(temp, (x, shadow_y), (shadow_radius, shadow_radius // 2), 
                       0, 0, 360, (50, 50, 50), -1)
            
            ball_color = Config.COLOR_DANGER if is_wicket else (0, 140, 255)
            cv2.circle(temp, (x, y), ball_radius, ball_color, -1)
            cv2.circle(temp, (x, y), ball_radius, (255, 255, 255), 2)
            
            cv2.circle(temp, (x - ball_radius // 3, y - ball_radius // 3), 
                      ball_radius // 4, (255, 255, 255), -1)
            
            cv2.imshow("Hand Cricket Pro", temp)
            
            key = cv2.waitKey(Config.BALL_ANIMATION_DELAY_MS)
            if key == 27:
                return True
        
        return False
    
    @staticmethod
    def animate_wicket(frame) -> bool:
        if not Config.ENABLE_ANIMATIONS:
            return False
        
        h, w = frame.shape[:2]
        
        for i in range(Config.WICKET_ANIMATION_FRAMES):
            temp = frame.copy()
            
            if i % 4 < 2:
                overlay = temp.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), Config.COLOR_DANGER, -1)
                cv2.addWeighted(overlay, 0.3, temp, 0.7, 0, temp)
            
            text = "WICKET!"
            scale = 3.0 + 0.5 * np.sin(i * 0.5)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 5)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.putText(temp, text, (text_x + 5, text_y + 5),
                       cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 5)
            cv2.putText(temp, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, scale, Config.COLOR_DANGER, 5)
            
            cv2.imshow("Hand Cricket Pro", temp)
            
            key = cv2.waitKey(50)
            if key == 27:
                return True
        
        return False
    
    @staticmethod
    def animate_boundary(frame, runs: int) -> bool:
        if not Config.ENABLE_ANIMATIONS or runs not in [4, 6]:
            return False
        
        h, w = frame.shape[:2]
        
        for i in range(15):
            temp = frame.copy()
            
            if runs == 6:
                for _ in range(3):
                    x = random.randint(w // 4, 3 * w // 4)
                    y = random.randint(h // 4, h // 2)
                    radius = random.randint(20, 50)
                    color = random.choice([
                        Config.COLOR_PRIMARY, 
                        Config.COLOR_SECONDARY, 
                        Config.COLOR_ACCENT
                    ])
                    cv2.circle(temp, (x, y), radius, color, 2)
            
            text = f"{runs}!" if runs == 4 else "SIX!"
            scale = 2.5 + 0.3 * np.sin(i * 0.5)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2 + 50
            
            color = Config.COLOR_SECONDARY if runs == 4 else Config.COLOR_PRIMARY
            
            cv2.putText(temp, text, (text_x + 3, text_y + 3),
                       cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 4)
            cv2.putText(temp, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, scale, color, 4)
            
            cv2.imshow("Hand Cricket Pro", temp)
            
            key = cv2.waitKey(40)
            if key == 27:
                return True
        
        return False

class HandCricketApp:
    def __init__(self):
        self.cap = None
        self.gesture_recognizer = GestureRecognizer()
        self.game = None
        self.mode = GameMode.SINGLE_PLAYER
        self.overs = Config.DEFAULT_OVERS
        self.running = False
        
        self.last_result_message = ""
        self.status_message = "Press SPACE to start match"
        
    def init_camera(self) -> bool:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Cannot open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def start_new_game(self):
        self.game = CricketGame(self.mode, self.overs)
        self.last_result_message = ""
        self.status_message = f"{'Player 1' if self.mode == GameMode.TWO_PLAYER else 'You'} batting - Press SPACE to throw"
    
    def process_throw(self, gesture: GestureType):
        if not self.game or gesture is None or gesture == GestureType.UNKNOWN:
            self.status_message = "Show clear hand gesture and press SPACE"
            return
        
        result = self.game.process_ball(gesture)
        
        if not result.get("success"):
            self.status_message = result.get("message", "Invalid throw")
            return
        
        if result.get("waiting"):
            self.status_message = result["message"]
            return
        
        self.last_result_message = result["message"]
        
        h = Config.CAMERA_HEIGHT
        w = Config.CAMERA_WIDTH
        start_pos = (w // 4, h // 4)
        end_pos = (3 * w // 4, 3 * h // 4)
        
        ret, frame = self.cap.read()
        if ret:
            if Config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            
            if result.get("wicket"):
                Animator.animate_ball_delivery(frame, start_pos, end_pos, is_wicket=True)
                Animator.animate_wicket(frame)
            else:
                Animator.animate_ball_delivery(frame, start_pos, end_pos)
                runs = result.get("runs", 0)
                if runs in [4, 6]:
                    Animator.animate_boundary(frame, runs)
        
        if result.get("innings_complete"):
            self.game.phase = GamePhase.INNINGS_BREAK
            self.status_message = "Press SPACE for 2nd innings"
        elif result.get("match_complete"):
            self.status_message = f"{result.get('winner', 'Unknown')} wins! Press R to restart"
        else:
            if self.mode == GameMode.TWO_PLAYER and not self.game.waiting_for_player2:
                self.status_message = "Player 1: Show hand and press SPACE"
            else:
                self.status_message = "Press SPACE for next ball"
    
    def render_frame(self, frame, gesture: GestureType, landmarks):
        h, w = frame.shape[:2]
        
        self.gesture_recognizer.draw_landmarks(frame, landmarks)
        
        if not self.game:
            self.draw_main_menu(frame)
        elif self.game.phase == GamePhase.INNINGS_BREAK:
            UIRenderer.draw_innings_break(frame, self.game)
        elif self.game.phase == GamePhase.MATCH_COMPLETE:
            UIRenderer.draw_scoreboard(frame, self.game, 20, 20)
            UIRenderer.draw_match_result(frame, self.game)
        else:
            UIRenderer.draw_scoreboard(frame, self.game, 20, 20)
            UIRenderer.draw_gesture_display(frame, gesture, w - 300, 20, True)
            
            if self.game.phase == GamePhase.FIRST_INNINGS:
                UIRenderer.draw_player_stats(frame, self.game.player1, 20, h - 220, True)
                UIRenderer.draw_player_stats(frame, self.game.player2, w - 300, h - 220, False)
            else:
                UIRenderer.draw_player_stats(frame, self.game.player2, 20, h - 220, True)
                UIRenderer.draw_player_stats(frame, self.game.player2, w - 300, h - 220, False)
            
            UIRenderer.draw_controls(frame, w // 2 - 100, h - 110)
        
        UIRenderer.draw_status_bar(frame, self.status_message, h - 50)
        
        if self.last_result_message and self.game and self.game.phase not in [GamePhase.INNINGS_BREAK, GamePhase.MATCH_COMPLETE]:
            UIRenderer.draw_result_highlight(frame, self.last_result_message, h - 150)
    
    def draw_main_menu(self, frame):
        h, w = frame.shape[:2]
        
        title = "HAND CRICKET PRO"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.5, 5)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, title, (text_x, 150),
                   cv2.FONT_HERSHEY_DUPLEX, 2.5, Config.COLOR_PRIMARY, 5)
        
        menu_y = 250
        options = [
            f"Mode: {self.mode.value}  (Press M)",
            f"Overs: {self.overs}  (Press O)",
            "",
            "Press SPACE to START",
            "",
            "How to Play:",
            "- Show 0-5 fingers for runs",
            "- Show only THUMB for SIX",
            "- Same number = WICKET"
        ]
        
        for i, option in enumerate(options):
            if option:
                text_size = cv2.getTextSize(option, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (w - text_size[0]) // 2
                
                if "START" in option:
                    color = Config.COLOR_ACCENT
                    weight = 3
                elif "Press" in option or "Mode" in option or "Overs" in option:
                    color = Config.COLOR_SECONDARY
                    weight = 2
                else:
                    color = Config.COLOR_TEXT
                    weight = 1
                
                cv2.putText(frame, option, (text_x, menu_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, weight)
            menu_y += 45
    
    def handle_key(self, key):
        if key == -1:
            return True
        
        if key == 27:
            return False
        
        if key == ord(' '):
            if not self.game:
                self.start_new_game()
            elif self.game.phase == GamePhase.INNINGS_BREAK:
                self.game.start_second_innings()
                self.status_message = "2nd innings started!"
            else:
                gesture, _ = self.gesture_recognizer.process_frame(self.current_frame)
                if gesture:
                    self.process_throw(gesture)
        
        elif key == ord('r') or key == ord('R'):
            if self.game:
                self.game.reset()
                self.last_result_message = ""
                self.status_message = "Match reset - Press SPACE"
        
        elif key == ord('m') or key == ord('M'):
            self.mode = GameMode.TWO_PLAYER if self.mode == GameMode.SINGLE_PLAYER else GameMode.SINGLE_PLAYER
            if self.game:
                self.start_new_game()
            self.status_message = f"Mode: {self.mode.value}"
        
        elif key == ord('o') or key == ord('O'):
            self.overs = (self.overs % Config.MAX_OVERS) + 1
            if self.overs < Config.MIN_OVERS:
                self.overs = Config.MIN_OVERS
            if self.game:
                self.start_new_game()
            self.status_message = f"Overs set to {self.overs}"
        
        return True
    
    def run(self):
        if not self.init_camera():
            return
        
        self.running = True
        print("=" * 60)
        print("HAND CRICKET PRO")
        print("=" * 60)
        print("Controls:")
        print("  SPACE - Submit throw / Start game")
        print("  M - Toggle game mode")
        print("  O - Change overs")
        print("  R - Reset match")
        print("  ESC - Quit")
        print("=" * 60)
        
        cv2.namedWindow("Hand Cricket Pro", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Cricket Pro", Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("ERROR: Cannot read frame")
                break
            
            if Config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            
            self.current_frame = frame.copy()
            
            gesture, landmarks = self.gesture_recognizer.process_frame(frame)
            
            self.render_frame(frame, gesture, landmarks)
            
            cv2.imshow("Hand Cricket Pro", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_key(key):
                break
        
        self.cleanup()
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Thanks for playing Hand Cricket Pro!")

def main():
    app = HandCricketApp()
    app.run()

if __name__ == "__main__":
    main()
