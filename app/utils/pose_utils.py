import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Named constants instead of magic numbers
LANDMARK = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_HIP": 23,      "R_HIP": 24,
    "L_KNEE": 25,     "R_KNEE": 26,
    "L_ANKLE": 27,    "R_ANKLE": 28,
}

def calculate_angle(a, b, c) -> float:
    """Returns angle at point b formed by a-b-c in degrees."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / norm_product
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_features(kp: np.ndarray) -> Optional[list]:
    """
    Extract biomechanical features from 33 MediaPipe landmarks.
    Returns None if keypoints are invalid.
    """
    try:
        required = list(LANDMARK.values())
        if kp.shape[0] < max(required) + 1:
            logger.error("Insufficient landmarks: got %d", kp.shape[0])
            return None

        L_SH, R_SH     = kp[11], kp[12]
        L_HIP, R_HIP   = kp[23], kp[24]
        L_KNEE, R_KNEE = kp[25], kp[26]
        L_ANKLE, R_ANKLE = kp[27], kp[28]

        left_knee_angle   = calculate_angle(L_HIP, L_KNEE, L_ANKLE)
        right_knee_angle  = calculate_angle(R_HIP, R_KNEE, R_ANKLE)
        left_hip_angle    = calculate_angle(L_SH, L_HIP, L_KNEE)
        right_hip_angle   = calculate_angle(R_SH, R_HIP, R_KNEE)

        # Ankle dorsiflexion: vertical reference below ankle
        left_ankle_angle  = calculate_angle(L_KNEE, L_ANKLE, [L_ANKLE[0], L_ANKLE[1] + 50])
        right_ankle_angle = calculate_angle(R_KNEE, R_ANKLE, [R_ANKLE[0], R_ANKLE[1] + 50])

        mid_sh  = [(L_SH[0]  + R_SH[0])  / 2, (L_SH[1]  + R_SH[1])  / 2]
        mid_hip = [(L_HIP[0] + R_HIP[0]) / 2, (L_HIP[1] + R_HIP[1]) / 2]
        # Spine angle relative to vertical
        spine_angle = calculate_angle(mid_sh, mid_hip, [mid_hip[0], mid_hip[1] + 100])

        torso_lean         = abs(left_hip_angle - right_hip_angle)
        left_knee_lateral  = abs(L_KNEE[0] - L_ANKLE[0])
        right_knee_lateral = abs(R_KNEE[0] - R_ANKLE[0])
        symmetry_score     = abs(left_knee_angle - right_knee_angle)
        hip_depth          = abs(
            ((L_HIP[1] + R_HIP[1]) / 2) - ((L_KNEE[1] + R_KNEE[1]) / 2)
        )

        return [
            left_knee_angle, right_knee_angle,
            left_hip_angle,  right_hip_angle,
            left_ankle_angle, right_ankle_angle,
            spine_angle, torso_lean,
            left_knee_lateral, right_knee_lateral,
            symmetry_score, hip_depth,
        ]

    except (IndexError, ValueError) as e:
        logger.exception("Feature extraction failed: %s", e)
        return None