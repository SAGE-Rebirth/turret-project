import numpy as np
from .config import TURRET_LIMITS, PIXELS_PER_DEGREE

class TurretController:
    """
    Simulates a 2-axis gimbal/turret with PID control.
    Physical Model: Converts pixel error -> servo angle adjustments.
    """
    def __init__(self, kp=0.1, ki=0.01, kd=0.05):
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        
        # PID constants
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # State
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.integral_x = 0
        self.integral_y = 0

    def update(self, error_x, error_y):
        """
        Calculate new servo angles based on pixel error.
        error_x, error_y: Pixel difference from center.
        """
        # --- PID: X-axis (Pan) ---
        self.integral_x += error_x
        derivative_x = error_x - self.prev_error_x
        output_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * derivative_x)
        self.prev_error_x = error_x
        
        # --- PID: Y-axis (Tilt) ---
        self.integral_y += error_y
        derivative_y = error_y - self.prev_error_y
        output_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * derivative_y)
        self.prev_error_y = error_y
        
        # --- Physical Limiting ---
        # 1. Speed Limit (Simulate servo max speed)
        # Note: We scale output by 0.1 to convert "PID unit" to "Degrees" approximately
        delta_pan = np.clip(output_x * 0.1, -TURRET_LIMITS['MAX_PAN_SPEED'], TURRET_LIMITS['MAX_PAN_SPEED'])
        delta_tilt = np.clip(output_y * 0.1, -TURRET_LIMITS['MAX_TILT_SPEED'], TURRET_LIMITS['MAX_TILT_SPEED'])
        
        # 2. Angle Limit (Mechanical stops)
        self.pan_angle = np.clip(self.pan_angle + delta_pan, TURRET_LIMITS['PAN_RANGE'][0], TURRET_LIMITS['PAN_RANGE'][1])
        # Invert tilt because +Y pixel (down) usually means -Tilt (down) or vice versa depending on mount.
        # Here: Box is below center (+error) -> Camera needs to look down.
        # If looking down is negative angle:
        self.tilt_angle = np.clip(self.tilt_angle - delta_tilt, TURRET_LIMITS['TILT_RANGE'][0], TURRET_LIMITS['TILT_RANGE'][1])
        
        return self.pan_angle, self.tilt_angle

    def get_current_aim_point(self, center_x, center_y):
        """
        Convert current servo angles back to screen coordinates for visualization.
        """
        # Pan: +Angle (Right) -> +Pixels
        aim_x = center_x + (self.pan_angle * PIXELS_PER_DEGREE)
        
        # Tilt: +Angle (Up) -> -Pixels (Up on screen) 
        # Note: In update(), we defined +Y pixel error (target below center) = Need to look Down (-Tilt).
        # So -Tilt = Down. 
        # Screen Y: Down is +Pixels.
        # So -Tilt * Scale => Positive Pixel Offset.
        # Let's verify: 
        # Target at (Center, Center+100) [Down]. ErrorY = 100.
        # PID OutputY > 0. DeltaTilt > 0. 
        # Tilt = Tilt - DeltaTilt. Tilt becomes negative (Looking Down).
        # We want AimY to be (Center+100).
        # aim_y = Center - (Tilt * Scale).
        # if Tilt is -10 deg. AimY = Center - (-10 * 10) = Center + 100. Correct.
        aim_y = center_y - (self.tilt_angle * PIXELS_PER_DEGREE)
        
        return int(aim_x), int(aim_y)
