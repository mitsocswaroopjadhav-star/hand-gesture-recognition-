"""
Hand Gesture Media Control - Detector Module
Real-time hand landmark detection using MediaPipe Tasks + OpenCV.
Detects one hand, extracts all 21 landmarks, draws them manually,
returns coordinates as a Python list, displays FPS, exits on Q.
"""

import os
import urllib.request
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe hand has 21 landmarks. The following 20 edges define the
# skeleton connections (matching the original HAND_CONNECTIONS).
_HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Drawing colours in BGR format
_LANDMARK_COLOR = (0, 255, 0)       # Green circles for landmarks
_CONNECTION_COLOR = (0, 255, 0)     # Green lines for connections
_LANDMARK_RADIUS = 4                # Radius of each landmark dot
_CONNECTION_THICKNESS = 2           # Thickness of each connection line

# Official MediaPipe Hand Landmarker model URL
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_MODEL_PATH = "hand_landmarker.task"


def _ensure_model_exists():
    """Download the hand landmarker model if it doesn't exist locally."""
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading MediaPipe Hand Landmarker model to {_MODEL_PATH}...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print(f"Please manually download from: {_MODEL_URL}")
            print(f"And save as {_MODEL_PATH} in the working directory.")
            raise


class HandDetector:
    """
    Encapsulates MediaPipe Tasks HandLandmarker detection and OpenCV
    manual landmark visualisation.
    Single-hand mode — optimised for media-control gestures.
    """

    def __init__(self):
        """
        Initialise MediaPipe Tasks HandLandmarker with real-time webcam
        parameters.

        num_hands=1              -> only one hand needed for gesture control
        min_hand_detection_confidence=0.7 -> 70% threshold to declare a hand present
        min_hand_tracking_confidence=0.5   -> 50% threshold to keep tracking a hand
        """
        # Ensure model file exists (download if needed)
        _ensure_model_exists()

        # Build HandLandmarkerOptions for single-hand real-time detection
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=_MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Create the detector instance
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        # FPS tracking — prev_time holds the timestamp of the last processed frame
        self.prev_time = 0.0
        self.fps = 0.0

    def detect_hand(self, frame):
        """
        Run MediaPipe Tasks HandLandmarker detection on a BGR frame.

        Args:
            frame: numpy BGR image array from OpenCV VideoCapture.read()

        Returns:
            HandLandmarkerResult containing hand_landmarks,
            or None if no hand is found.
        """
        # MediaPipe Tasks expects RGB input; OpenCV provides BGR — convert once
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the frame in a MediaPipe Image object (use mp.Image, mp.ImageFormat)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run detection
        result = self.hand_landmarker.detect(mp_image)

        return result

    def get_landmarks(self, result, frame_shape):
        """
        Convert normalized MediaPipe landmarks to pixel-space (x, y) tuples.

        Args:
            result: HandLandmarkerResult from detect_hand()
            frame_shape: tuple (height, width, channels) from frame.shape

        Returns:
            List of 21 (x, y) integer tuples — pixel coordinates of each landmark,
            or None if no hand was detected in result.hand_landmarks.
        """
        # Guard: no hands detected or empty result
        if result is None or not result.hand_landmarks:
            return None

        # We configured num_hands=1, so index 0 is the only hand
        hand_landmarks = result.hand_landmarks[0]

        # Unpack frame dimensions for coordinate denormalization
        # MediaPipe returns x, y in [0.0, 1.0] relative to frame size
        height = frame_shape[0]
        width = frame_shape[1]

        # Iterate over all 21 landmarks, convert normalized coords to pixel coords
        landmarks = []
        for lm in hand_landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            landmarks.append((x, y))

        return landmarks  # List of 21 (int, int) tuples

    def draw_landmarks(self, frame, result):
        """
        Draw the 21 hand landmarks + 20 connecting edges onto the frame
        using only OpenCV (cv2.circle and cv2.line).

        Args:
            frame: BGR image to annotate (modified in-place)
            result: HandLandmarkerResult — may contain zero or more detected hands

        Returns:
            The same frame object, now with landmarks drawn on it
        """
        # Only draw when at least one hand is present
        if result is None or not result.hand_landmarks:
            return frame

        for hand_landmarks in result.hand_landmarks:
            height, width = frame.shape[:2]

            # Convert all normalized landmarks to pixel coordinates once
            pixel_landmarks = []
            for lm in hand_landmarks:
                px = int(lm.x * width)
                py = int(lm.y * height)
                pixel_landmarks.append((px, py))

            # Draw connections (lines) between paired landmarks
            for idx_a, idx_b in _HAND_CONNECTIONS:
                pt_a = pixel_landmarks[idx_a]
                pt_b = pixel_landmarks[idx_b]
                cv2.line(frame, pt_a, pt_b, _CONNECTION_COLOR, _CONNECTION_THICKNESS)

            # Draw landmark points (filled circles)
            for (px, py) in pixel_landmarks:
                cv2.circle(frame, (px, py), _LANDMARK_RADIUS, _LANDMARK_COLOR, -1)

        return frame  # Return for method chaining (frame is modified in-place)

    def calculate_fps(self):
        """
        Compute instantaneous FPS from the time delta between consecutive calls.

        Returns:
            float: FPS value (0.0 on the first call, when prev_time has not been set yet)
        """
        current_time = time.time()

        # First-call guard: prev_time is 0.0 on initialisation
        if self.prev_time == 0.0:
            self.prev_time = current_time
            return 0.0

        time_diff = current_time - self.prev_time
        if time_diff > 0.0:
            self.fps = 1.0 / time_diff

        self.prev_time = current_time

        return self.fps

    def display_fps(self, frame):
        """
        Render a white FPS counter in the top-left corner of the frame.

        Args:
            frame: BGR image to annotate (modified in-place)

        Returns:
            The same frame object with FPS text overlaid
        """
        fps = self.calculate_fps()

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame


def main():
    """
    Entry point: opens the webcam, runs real-time hand-keypoint detection,
    draws landmarks + FPS on each frame, and exits when the user presses Q.
    """
    # --- Webcam setup ---
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    time.sleep(2)

    if not cap.isOpened():
        print("Error: Could not open webcam (index 0)")
        print("Check camera permissions in System Settings > Privacy > Camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- Detector setup ---
    detector = HandDetector()

    print("=" * 45)
    print("  Hand Gesture Media Control — Detector")
    print("=" * 45)
    print("  Webcam opened: 640x480")
    print("  Detecting 1 hand with 21 landmarks")
    print("  Press 'Q' to quit")
    print("=" * 45)

    # --- Main processing loop ---
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot receive frame from webcam")
            break

        # Detect hand landmarks
        result = detector.detect_hand(frame)

        # If at least one hand is present, extract and draw its landmarks
        if result and result.hand_landmarks:
            landmarks = detector.get_landmarks(result, frame.shape)
            frame = detector.draw_landmarks(frame, result)

            # Landmarks extracted but no per-frame console spam

        # Add the FPS overlay
        frame = detector.display_fps(frame)

        # Show the annotated frame
        cv2.imshow("Hand Gesture Detection", frame)

        # Exit on Q key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("Q pressed — shutting down.")
            break

    # --- Cleanup ---
    cap.release()
    detector.hand_landmarker.close()
    cv2.destroyAllWindows()

    print("Detector stopped.")


if __name__ == "__main__":
    main()