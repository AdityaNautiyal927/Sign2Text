# utils.py
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def extract_landmarks_from_results(results):
    """
    Input: results from mp_hands.Hands.process(rgb_image)
    Output: flattened list of 63 floats (21 landmarks x 3 coords) normalized relative to wrist,
            or None if no hand detected.
    """
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    # Get raw normalized coords
    lms = [(lm.x, lm.y, lm.z) for lm in hand.landmark]  # 21 tuples
    # Convert to numpy and make relative to wrist (index 0)
    arr = np.array(lms, dtype=np.float32)
    base = arr[0]  # wrist
    rel = arr - base  # translation invariance
    # Optionally normalize scale by max absolute coordinate
    max_val = np.max(np.abs(rel))
    if max_val > 0:
        rel = rel / max_val
    return rel.flatten().tolist()  # length 63
