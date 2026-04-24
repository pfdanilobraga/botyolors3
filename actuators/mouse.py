import time
import random
import numpy as np
import pydirectinput
import pyautogui
from typing import Tuple

class MouseActuator:
    """
    Simulates human-like mouse movements using Bézier curves and parametric noise.
    """
    
    def __init__(self):
        # Disable pyautogui's fail-safe for more control, but keep it in mind
        pyautogui.FAILSAFE = True
        # Set pause between actions to 0 to manage it manually
        pyautogui.PAUSE = 0
        pydirectinput.PAUSE = 0

    def _calculate_bezier_curve(self, start: Tuple[int, int], end: Tuple[int, int], 
                                num_points: int) -> list:
        """
        Generates a Bézier curve path between start and end.
        """
        # Create control points for the Bézier curve
        # We add some randomness to the control points to simulate human jitter
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Offset for control points
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        offset = dist * 0.2
        
        cp1 = (
            mid_x + random.uniform(-offset, offset),
            mid_y + random.uniform(-offset, offset)
        )
        
        path = []
        for t in np.linspace(0, 1, num_points):
            # Quadratic Bézier curve formula: (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
            x = (1 - t)**2 * start[0] + 2 * (1 - t) * t * cp1[0] + t**2 * end[0]
            y = (1 - t)**2 * start[1] + 2 * (1 - t) * t * cp1[1] + t**2 * end[1]
            path.append((int(x), int(y)))
            
        return path

    def move_to(self, x: int, y: int, duration: float = None):
        """
        Moves the mouse to (x, y) coordinates with a human-like curve.
        """
        start_x, start_y = pyautogui.position()
        
        if duration is None:
            # Calculate duration based on distance
            dist = np.sqrt((x - start_x)**2 + (y - start_y)**2)
            duration = random.uniform(0.15, 0.4) + (dist / 1000) * 0.2
            
        num_points = max(10, int(duration * 60)) # Target 60fps movement
        path = self._calculate_bezier_curve((start_x, start_y), (x, y), num_points)
        
        time_per_point = duration / num_points
        
        for px, py in path:
            # Add micro-jitter
            jx = px + random.randint(-1, 1) if random.random() > 0.8 else px
            jy = py + random.randint(-1, 1) if random.random() > 0.8 else py
            
            pydirectinput.moveTo(jx, jy)
            time.sleep(time_per_point * random.uniform(0.8, 1.2))

    def click(self, button: str = 'left', delay_range: Tuple[float, float] = (0.05, 0.15)):
        """
        Simulates a human click with a small variable hold time.
        """
        pydirectinput.mouseDown(button=button)
        time.sleep(random.uniform(*delay_range))
        pydirectinput.mouseUp(button=button)

    def human_click_at(self, x: int, y: int, button: str = 'left'):
        """
        Moves to a target and clicks.
        """
        self.move_to(x, y)
        time.sleep(random.uniform(0.05, 0.2))
        self.click(button)

if __name__ == "__main__":
    # Test block
    print("Testing human-like mouse movement...")
    actuator = MouseActuator()
    actuator.move_to(500, 500)
    print("Test complete.")
