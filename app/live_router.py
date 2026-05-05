import logging
import pickle
 
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
 
from app.utils.pose_utils import calculate_angle, extract_features
from app.utils.squat_logic import SquatCounter
 
logger = logging.getLogger(__name__)
router = APIRouter()
 
FEATURE_COLUMNS = [
    "left_knee_angle",  "right_knee_angle",
    "left_hip_angle",   "right_hip_angle",
    "left_ankle_angle", "right_ankle_angle",
    "spine_angle",      "torso_lean",
    "left_knee_lateral","right_knee_lateral",
    "symmetry_score",   "hip_depth",
]
 
 
class AnalysisResponse(BaseModel):
    rep_count:  int
    form:       str
    knee_angle: int
    feedback:   list[str]
    landmarks:  list  # normalized [x, y] 0.0–1.0
 
 
_squat_model = None
_pose        = None
_counter     = SquatCounter()
 
 
def get_model():
    global _squat_model
    if _squat_model is None:
        import os
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(BASE_DIR, "app", "models", "squat_model.pkl"), "rb") as f:
            _squat_model = pickle.load(f)
        logger.info("Squat model loaded.")
    return _squat_model
 
 
def get_pose():
    global _pose
    if _pose is None:
        _pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _pose
 
 
def fix_orientation(frame: np.ndarray) -> np.ndarray:
    """
    Android back camera sends frames rotated 90° clockwise in portrait mode.
    Rotate 90° counter-clockwise to make the person upright for MediaPipe.
    """
    h, w = frame.shape[:2]
    if w > h:
        # Landscape frame from portrait phone → rotate to portrait
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame
 
 
def process_frame(frame: np.ndarray) -> dict:
    frame = fix_orientation(frame)
 
    # Resize to portrait — keeps correct aspect ratio for skeleton coords
    frame = cv2.resize(frame, (320, 568))
 
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = get_pose().process(rgb)
 
    if not results.pose_landmarks:
        return AnalysisResponse(
            rep_count=_counter.rep_count,
            form="NO_PERSON",
            knee_angle=0,
            feedback=["Stand in frame — ensure full body visible"],
            landmarks=[],
        ).dict()
 
    lms = results.pose_landmarks.landmark
 
    # ✅ Send normalized coords directly — no pixel conversion needed
    # MediaPipe returns x,y as 0.0–1.0 of the processed frame dimensions
    normalized = [[lm.x, lm.y] for lm in lms]
 
    # Pixel coords for angle calculation only
    h, w = frame.shape[:2]
    kp = np.array([[lm.x * w, lm.y * h] for lm in lms])
 
    # Use both sides — pick better visibility side for rep counting
    left_knee  = calculate_angle(kp[23], kp[25], kp[27])
    right_knee = calculate_angle(kp[24], kp[26], kp[28])
 
    left_vis  = (lms[23].visibility + lms[25].visibility + lms[27].visibility) / 3
    right_vis = (lms[24].visibility + lms[26].visibility + lms[28].visibility) / 3
    knee_angle = left_knee if left_vis >= right_vis else right_knee
 
    reps = _counter.update(knee_angle)
 
    features = extract_features(kp)
    if features is None:
        return AnalysisResponse(
            rep_count=reps,
            form="INCORRECT",
            knee_angle=int(knee_angle),
            feedback=["Ensure full body is visible"],
            landmarks=normalized,
        ).dict()
 
    pred    = get_model().predict(
        pd.DataFrame([features], columns=FEATURE_COLUMNS))[0]
    posture = "CORRECT" if pred == 1 else "INCORRECT"
 
    feedback = []
    if posture == "INCORRECT":
        if knee_angle > 120:
            feedback.append("Lower your hips")
        elif knee_angle < 60:
            feedback.append("Too deep — reduce depth")
        feedback.append("Keep chest upright")
 
    return AnalysisResponse(
        rep_count=reps,
        form=posture,
        knee_angle=int(knee_angle),
        feedback=feedback,
        landmarks=normalized,
    ).dict()
 
 
@router.post("", response_model=AnalysisResponse)
async def live_analysis(file: UploadFile = File(...)):
    contents = await file.read()
    npimg    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
 
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")
 
    try:
        return process_frame(frame)
    except RuntimeError as e:
        logger.exception("Processing error")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@router.post("/reset")
async def reset_counter():
    _counter.reset()
    return {"message": "Counter reset", "rep_count": 0}
 