# --- CONFIGURATION ---

SAFE_ZONES = [
    # Format: (x_start_norm, y_start_norm, x_end_norm, y_end_norm)
    # These are normalized coordinates (0.0 to 1.0)
    (0.0, 0.0, 0.15, 1.0), # Left edge safety margin (e.g. "Doorway")
    (0.85, 0.0, 1.0, 1.0)  # Right edge safety margin
]

TURRET_LIMITS = {
    'MAX_PAN_SPEED': 8.0,   # Max degrees per frame (Increased for snappy tracking)
    'MAX_TILT_SPEED': 6.0,
    'PAN_RANGE': (-90, 90), # Mechanical limit
    'TILT_RANGE': (-45, 45)
}

AIM_MODES = {
    1: "HEAD (Forehead)",
    2: "UPPER_BODY",
    3: "NON_LETHAL"
}

# Calibration: Pixels per Degree
# Assuming approx 60 deg FOV for 1920 width => ~32 px/deg
PIXELS_PER_DEGREE = 32.0
