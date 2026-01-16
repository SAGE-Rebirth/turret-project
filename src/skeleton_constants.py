# COCO Keypoint Index
# 0: Nose
# 1: Left Eye
# 2: Right Eye
# 3: Left Ear
# 4: Right Ear
# 5: Left Shoulder
# 6: Right Shoulder
# 7: Left Elbow
# 8: Right Elbow
# 9: Left Wrist
# 10: Right Wrist
# 11: Left Hip
# 12: Right Hip
# 13: Left Knee
# 14: Right Knee
# 15: Left Ankle
# 16: Right Ankle

SKELETON_CONNECTIONS = [
    (5, 7), (7, 9), # Left Arm
    (6, 8), (8, 10), # Right Arm
    (5, 6), (5, 11), (6, 12), (11, 12), # Torso
    (11, 13), (13, 15), # Left Leg
    (12, 14), (14, 16)  # Right Leg
]
