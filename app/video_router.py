import logging
import os
import pickle
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Feature columns (must match squat_model.pkl training) ─────────────────────
FEATURE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "spine_angle",
    "torso_lean",
    "left_knee_lateral",
    "right_knee_lateral",
    "symmetry_score",
    "hip_depth",
]

# ── Singletons ─────────────────────────────────────────────────────────────────
_squat_model = None
_pose        = None


def get_model():
    global _squat_model
    if _squat_model is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(BASE_DIR, "app", "models", "squat_model.pkl")
        if not os.path.exists(path):
            raise RuntimeError(f"Model not found at {path}")
        with open(path, "rb") as f:
            _squat_model = pickle.load(f)
        logger.info("Squat model loaded.")
    return _squat_model


def get_pose():
    global _pose
    if _pose is None:
        # static_image_mode=True because we process individual frames from video
        _pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        logger.info("MediaPipe Pose initialized for video.")
    return _pose


# ── Angle helper ───────────────────────────────────────────────────────────────
def _angle(a, b, c) -> float:
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc  = a - b, c - b
    denom   = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))


# ── Extract all 12 features from a single frame's landmarks ───────────────────
def _extract(lms, h, w) -> dict | None:
    kp = np.array([[lms[i].x * w, lms[i].y * h] for i in range(33)])

    required = [11, 12, 23, 24, 25, 26, 27, 28]
    # Skip frame if any key landmark is not visible
    if any(lms[i].visibility < 0.4 for i in required):
        return None

    L_SH, R_SH     = kp[11], kp[12]
    L_HIP, R_HIP   = kp[23], kp[24]
    L_KNEE, R_KNEE = kp[25], kp[26]
    L_ANKLE, R_ANKLE = kp[27], kp[28]

    left_knee_angle   = _angle(L_HIP,  L_KNEE,  L_ANKLE)
    right_knee_angle  = _angle(R_HIP,  R_KNEE,  R_ANKLE)
    left_hip_angle    = _angle(L_SH,   L_HIP,   L_KNEE)
    right_hip_angle   = _angle(R_SH,   R_HIP,   R_KNEE)
    left_ankle_angle  = _angle(L_KNEE, L_ANKLE, [L_ANKLE[0], L_ANKLE[1] + 50])
    right_ankle_angle = _angle(R_KNEE, R_ANKLE, [R_ANKLE[0], R_ANKLE[1] + 50])

    mid_sh  = (L_SH  + R_SH)  / 2
    mid_hip = (L_HIP + R_HIP) / 2
    spine_angle = _angle(mid_sh, mid_hip, [mid_hip[0], mid_hip[1] + 100])

    torso_lean         = abs(left_hip_angle  - right_hip_angle)
    left_knee_lateral  = abs(L_KNEE[0]  - L_ANKLE[0])
    right_knee_lateral = abs(R_KNEE[0]  - R_ANKLE[0])
    symmetry_score     = abs(left_knee_angle - right_knee_angle)
    hip_depth          = abs(mid_hip[1] - ((L_KNEE[1] + R_KNEE[1]) / 2))

    return {
        "left_knee_angle":   left_knee_angle,
        "right_knee_angle":  right_knee_angle,
        "left_hip_angle":    left_hip_angle,
        "right_hip_angle":   right_hip_angle,
        "left_ankle_angle":  left_ankle_angle,
        "right_ankle_angle": right_ankle_angle,
        "spine_angle":       spine_angle,
        "torso_lean":        torso_lean,
        "left_knee_lateral": left_knee_lateral,
        "right_knee_lateral":right_knee_lateral,
        "symmetry_score":    symmetry_score,
        "hip_depth":         hip_depth,
    }


# ── Scoring & corrections ──────────────────────────────────────────────────────
def _compute_score_and_corrections(avg: dict, correct_pct: float) -> tuple[int, list[str]]:
    """
    Returns (score 0–100, list of correction strings).
    Score weights:
      40% — ML correct-form frame ratio
      20% — squat depth (knee angle)
      20% — spine uprightness
      10% — symmetry
      10% — knee tracking (no cave)
    """
    score = 0
    corrections = []

    # 1. ML form ratio (40 pts)
    score += int(correct_pct * 40)

    # 2. Squat depth (20 pts) — ideal knee angle 70–100°
    avg_knee = (avg["left_knee_angle"] + avg["right_knee_angle"]) / 2
    if 70 <= avg_knee <= 100:
        score += 20
    elif avg_knee > 100:
        depth_score = max(0, 20 - int((avg_knee - 100) / 3))
        score += depth_score
        if avg_knee > 120:
            corrections.append(
                f"Squat deeper — your average knee angle was {avg_knee:.0f}°, aim for 70–100°"
            )
        else:
            corrections.append(
                f"Try to go slightly deeper — knee angle {avg_knee:.0f}° (target 70–100°)"
            )
    else:
        depth_score = max(0, 20 - int((70 - avg_knee) / 2))
        score += depth_score
        corrections.append(
            f"Squat is too deep ({avg_knee:.0f}°) — risk of knee strain, aim for 70–100°"
        )

    # 3. Spine uprightness (20 pts) — ideal spine_angle 80–95°
    sp = avg["spine_angle"]
    if 80 <= sp <= 95:
        score += 20
    elif sp < 80:
        spine_score = max(0, 20 - int((80 - sp) / 2))
        score += spine_score
        corrections.append(
            f"Keep your chest more upright — spine angle was {sp:.0f}° (ideal 80–95°). "
            "Brace your core and lift your chest throughout the squat."
        )
    else:
        score += 15  # slightly past vertical is fine

    # 4. Symmetry (10 pts) — symmetry_score < 10° is good
    sym = avg["symmetry_score"]
    if sym < 10:
        score += 10
    elif sym < 20:
        score += 5
        corrections.append(
            f"Mild left/right imbalance detected ({sym:.0f}° difference). "
            "Focus on equal weight distribution on both feet."
        )
    else:
        corrections.append(
            f"Significant left/right imbalance ({sym:.0f}°). "
            "One side is compensating — consider single-leg strengthening exercises."
        )

    # 5. Knee tracking (10 pts) — lateral drift < 25px is good
    avg_lateral = (avg["left_knee_lateral"] + avg["right_knee_lateral"]) / 2
    if avg_lateral < 25:
        score += 10
    elif avg_lateral < 45:
        score += 5
        corrections.append(
            "Slight knee cave detected — push your knees out in line with your toes."
        )
    else:
        corrections.append(
            f"Knee cave (valgus collapse) detected — knees are drifting inward significantly. "
            "Strengthen glutes and focus on knee-out cue during descent."
        )

    # Clamp
    score = max(0, min(100, score))

    # Grade
    if score >= 85:
        grade = "Excellent"
    elif score >= 70:
        grade = "Good"
    elif score >= 50:
        grade = "Needs Work"
    else:
        grade = "Poor"

    if not corrections:
        corrections.append("Great form! Keep it up — maintain consistent depth and tempo.")

    return score, grade, corrections


# ── Rep counter from angle stream ──────────────────────────────────────────────
def _count_reps(angles: list[float]) -> int:
    reps  = 0
    state = "UP"
    for a in angles:
        if a < 90 and state == "UP":
            state = "DOWN"
        elif a > 155 and state == "DOWN":
            reps += 1
            state = "UP"
    return reps


# ── Route ──────────────────────────────────────────────────────────────────────
@router.post("")
async def analyze_video(file: UploadFile = File(...)):
    # Write upload to temp file (OpenCV needs a path)
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps            = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_skip     = max(1, int(fps / 8))  # analyse ~8 frames/sec

        frame_features = []   # one dict per analysed frame
        knee_angles    = []   # for rep counting
        frame_idx      = 0
        pose           = get_pose()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            feats = _extract(results.pose_landmarks.landmark, h, w)
            if feats is None:
                continue

            frame_features.append(feats)
            avg_knee = (feats["left_knee_angle"] + feats["right_knee_angle"]) / 2
            knee_angles.append(avg_knee)

        cap.release()

    finally:
        os.unlink(tmp_path)

    if len(frame_features) == 0:
        return {
            "score": 0,
            "grade": "No Data",
            "posture": "UNKNOWN",
            "reps_detected": 0,
            "frames_analysed": 0,
            "corrections": ["No squat pose detected in the video. Make sure your full body is visible."],
            "stats": {},
        }

    # ── Run ML classifier on each frame ───────────────────────────────────────
    model  = get_model()
    df_all = pd.DataFrame(frame_features, columns=FEATURE_COLUMNS)
    preds  = model.predict(df_all)

    correct_count = int((preds == 1).sum())
    correct_pct   = correct_count / len(preds)
    overall_form  = "CORRECT" if correct_pct >= 0.6 else "INCORRECT"

    # ── Average features across all frames ────────────────────────────────────
    avg_features = {col: float(df_all[col].mean()) for col in FEATURE_COLUMNS}

    # ── Score + corrections ────────────────────────────────────────────────────
    score, grade, corrections = _compute_score_and_corrections(avg_features, correct_pct)

    # ── Rep count ─────────────────────────────────────────────────────────────
    reps = _count_reps(knee_angles)

    return {
        "score":            score,
        "grade":            grade,
        "posture":          overall_form,
        "reps_detected":    reps,
        "frames_analysed":  len(frame_features),
        "correct_frames_pct": round(correct_pct * 100, 1),
        "corrections":      corrections,
        "stats": {
            "avg_knee_angle":    round(avg_features["left_knee_angle"], 1),
            "avg_spine_angle":   round(avg_features["spine_angle"],     1),
            "avg_symmetry":      round(avg_features["symmetry_score"],  1),
            "avg_knee_lateral":  round((avg_features["left_knee_lateral"] +
                                        avg_features["right_knee_lateral"]) / 2, 1),
            "min_knee_angle":    round(float(df_all[["left_knee_angle","right_knee_angle"]].min().min()), 1),
            "max_knee_angle":    round(float(df_all[["left_knee_angle","right_knee_angle"]].max().max()), 1),
        },
    }
