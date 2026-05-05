import logging
 
logger = logging.getLogger(__name__)
 
 
class SquatCounter:
    DOWN_THRESHOLD = 90    # knee angle below this = squat down
    UP_THRESHOLD   = 155   # knee angle above this = standing up
    MIN_FRAMES     = 2     # must hold state for N frames to count (debounce)
 
    def __init__(self):    # ✅ double underscores — was _init_ (broken)
        self.rep_count      = 0
        self.state          = "UP"
        self._frames_in_state = 0
 
    def update(self, knee_angle: float) -> int:
        if not isinstance(knee_angle, (int, float)):
            return self.rep_count
 
        if knee_angle < self.DOWN_THRESHOLD and self.state == "UP":
            self._frames_in_state += 1
            if self._frames_in_state >= self.MIN_FRAMES:
                self.state = "DOWN"
                self._frames_in_state = 0
                logger.debug("DOWN (angle=%.1f)", knee_angle)
 
        elif knee_angle > self.UP_THRESHOLD and self.state == "DOWN":
            self._frames_in_state += 1
            if self._frames_in_state >= self.MIN_FRAMES:
                self.rep_count += 1
                self.state = "UP"
                self._frames_in_state = 0
                logger.info("Rep counted: %d", self.rep_count)
        else:
            self._frames_in_state = 0
 
        return self.rep_count
 
    def reset(self):
        self.rep_count        = 0
        self.state            = "UP"
        self._frames_in_state = 0