from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from config import VisionConfig


@dataclass
class HandObservation:
    normalized_landmarks: np.ndarray
    pixel_landmarks: np.ndarray
    center_xy: tuple[float, float]
    bbox_norm: tuple[float, float, float, float]
    confidence: float
    handedness: str
    palm_width_norm: float
    wrist_middle_norm: float
    bbox_diag_norm: float
    depth_metric: float
    pinch_metric: float
    finger_curl_metric: float
    wrist_tilt_metric: float


# ============================================================================
# Geometry helpers
# ============================================================================
def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _finger_curl(points: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> float:
    pip_angle = _angle_degrees(points[mcp], points[pip], points[dip])
    dip_angle = _angle_degrees(points[pip], points[dip], points[tip])
    pip_curl = np.clip((180.0 - pip_angle) / 110.0, 0.0, 1.0)
    dip_curl = np.clip((180.0 - dip_angle) / 100.0, 0.0, 1.0)
    return float(0.6 * pip_curl + 0.4 * dip_curl)


def _finger_closure(points, palm_center, mcp, pip, dip, tip) -> float:
    joint_curl = _finger_curl(points, mcp, pip, dip, tip)
    chain_length = float(
        np.linalg.norm(points[pip] - points[mcp])
        + np.linalg.norm(points[dip] - points[pip])
        + np.linalg.norm(points[tip] - points[dip])
    )
    chain_length = max(chain_length, 1e-6)
    tip_to_base_ratio = float(np.linalg.norm(points[tip] - points[mcp]) / chain_length)
    tip_to_palm_ratio = float(np.linalg.norm(points[tip] - palm_center) / chain_length)
    base_distance_curl = float(np.clip((0.92 - tip_to_base_ratio) / 0.52, 0.0, 1.0))
    palm_distance_curl = float(np.clip((1.55 - tip_to_palm_ratio) / 0.95, 0.0, 1.0))
    return float(0.5 * joint_curl + 0.3 * base_distance_curl + 0.2 * palm_distance_curl)


def _roll_metric(points: np.ndarray) -> float:
    """In-plane hand roll (twist) angle in radians. 0 = hand pointing straight up.
    Uses the hand axis wrist(0) -> mean of finger MCPs. MediaPipe Y grows downward,
    so -v[1] is the upward direction."""
    palm_base_center = points[[5, 9, 17]].mean(axis=0)
    v = palm_base_center - points[0]
    return float(np.arctan2(v[0], -v[1]))


class HandTracker:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            model_complexity=int(getattr(config, "model_complexity", 1)),
        )

    def close(self) -> None:
        self._hands.close()

    def process(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        """Return ALL detected hands (filtered), sorted by confidence. Two-hand
        mode consumes the list; direct mode takes best_of()."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        observations: list[HandObservation] = []

        for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            handedness_label = "unknown"
            handedness_score = 1.0
            if result.multi_handedness and index < len(result.multi_handedness):
                handedness = result.multi_handedness[index].classification[0]
                handedness_label = handedness.label.lower()
                handedness_score = float(handedness.score)

            requested = self.config.handedness.lower()
            if requested in {"left", "right"} and handedness_label != requested:
                continue

            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )

            palm_width = float(np.linalg.norm(landmarks[5, :2] - landmarks[17, :2]))
            if palm_width < self.config.min_palm_width_norm:
                continue

            wrist_middle = float(np.linalg.norm(landmarks[0, :2] - landmarks[9, :2]))
            x_min = float(np.min(landmarks[:, 0]))
            y_min = float(np.min(landmarks[:, 1]))
            x_max = float(np.max(landmarks[:, 0]))
            y_max = float(np.max(landmarks[:, 1]))
            bbox_diag = float(np.hypot(x_max - x_min, y_max - y_min))

            # Palm center on the wrist->middle-MCP axis: it lies on the twist axis,
            # so hand X/Y stay stable when the wrist rolls (keeps yaw/height clean).
            palm_center = 0.5 * (landmarks[0] + landmarks[9])

            pinch_metric = float(
                np.linalg.norm(landmarks[4, :2] - landmarks[8, :2]) / max(palm_width, 1e-6)
            )

            # --- Depth proxy (Project A style): apparent hand size + slight z. ---
            # Simple and predictable; it IS perturbed by roll/grip, which is exactly
            # why the control layer freezes depth during roll and grip (see main.py).
            mean_palm_depth = float(max(0.0, -landmarks[[0, 5, 9, 13, 17], 2].mean()))
            depth_metric = 0.65 * palm_width + 0.25 * wrist_middle + 0.10 * mean_palm_depth

            finger_closures = [
                _finger_closure(landmarks, palm_center, 1, 2, 3, 4),
                _finger_closure(landmarks, palm_center, 5, 6, 7, 8),
                _finger_closure(landmarks, palm_center, 9, 10, 11, 12),
                _finger_closure(landmarks, palm_center, 13, 14, 15, 16),
                _finger_closure(landmarks, palm_center, 17, 18, 19, 20),
            ]
            finger_curl_metric = float(
                0.7 * np.mean(finger_closures) + 0.3 * np.min(finger_closures)
            )

            wrist_tilt_metric = _roll_metric(landmarks)

            pixel_landmarks = np.empty((landmarks.shape[0], 2), dtype=np.int32)
            pixel_landmarks[:, 0] = np.clip(landmarks[:, 0] * frame_w, 0, frame_w - 1).astype(np.int32)
            pixel_landmarks[:, 1] = np.clip(landmarks[:, 1] * frame_h, 0, frame_h - 1).astype(np.int32)

            observation = HandObservation(
                normalized_landmarks=landmarks,
                pixel_landmarks=pixel_landmarks,
                center_xy=(float(palm_center[0]), float(palm_center[1])),
                bbox_norm=(x_min, y_min, x_max, y_max),
                confidence=handedness_score,
                handedness=handedness_label,
                palm_width_norm=palm_width,
                wrist_middle_norm=wrist_middle,
                bbox_diag_norm=bbox_diag,
                depth_metric=depth_metric,
                pinch_metric=pinch_metric,
                finger_curl_metric=finger_curl_metric,
                wrist_tilt_metric=wrist_tilt_metric,
            )
            observations.append(observation)

        observations.sort(key=lambda o: o.confidence, reverse=True)
        return observations


def best_of(observations: list[HandObservation]) -> HandObservation | None:
    """Highest-confidence hand (legacy single-hand behaviour)."""
    return observations[0] if observations else None
