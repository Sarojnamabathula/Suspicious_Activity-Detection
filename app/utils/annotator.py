"""
SentinelAI — Frame Annotator.

Draws detection bounding boxes and a semi-transparent heads-up display
(HUD) onto each video frame using OpenCV.  The HUD shows the current
severity, risk score, and active violation reasons.
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from app.api.schemas import (
    BoundingBox,
    DecisionOutput,
    FrameDetections,
    Severity,
)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# BGR colour tuples
_GREEN: tuple[int, int, int] = (0, 200, 0)
_RED: tuple[int, int, int] = (0, 0, 220)
_BLUE: tuple[int, int, int] = (220, 150, 0)
_ORANGE: tuple[int, int, int] = (0, 140, 255)
_DARK_RED: tuple[int, int, int] = (0, 0, 139)
_YELLOW: tuple[int, int, int] = (0, 220, 220)
_WHITE: tuple[int, int, int] = (255, 255, 255)
_BLACK: tuple[int, int, int] = (0, 0, 0)
_PANEL_BG: tuple[int, int, int] = (30, 30, 30)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_LABEL: float = 0.5
_FONT_SCALE_HUD_TITLE: float = 0.55
_FONT_SCALE_HUD_BODY: float = 0.45
_FONT_SCALE_TIMESTAMP: float = 0.45
_FONT_THICKNESS: int = 1
_BOX_THICKNESS: int = 2
_HUD_PADDING: int = 12
_HUD_LINE_HEIGHT: int = 22
_HUD_WIDTH: int = 240
_HUD_ALPHA: float = 0.70


class FrameAnnotator:
    """Renders detection overlays and a severity HUD onto video frames.

    All drawing uses OpenCV primitives with anti-aliased text
    (``cv2.LINE_AA``) for a clean, professional look.
    """

    # Severity → BGR colour mapping
    SEVERITY_COLORS: dict[Severity, tuple[int, int, int]] = {
        Severity.SAFE: _GREEN,
        Severity.LOW: _YELLOW,
        Severity.MEDIUM: _ORANGE,
        Severity.HIGH: _RED,
        Severity.CRITICAL: _DARK_RED,
    }

    # ── public API ───────────────────────────────────────────────────

    def annotate(
        self,
        frame: np.ndarray,
        decision: DecisionOutput,
        detections: FrameDetections,
    ) -> np.ndarray:
        """Return a copy of *frame* with detection boxes, HUD, and timestamp.

        Parameters
        ----------
        frame:
            BGR image array (H×W×3, ``uint8``).
        decision:
            Current :class:`DecisionOutput` from the decision engine.
        detections:
            Per-frame :class:`FrameDetections` aggregating all detector
            outputs.

        Returns
        -------
        np.ndarray
            Annotated copy of the input frame.
        """
        canvas: np.ndarray = frame.copy()

        # 1. Face bounding box — green when present, red when missing
        if detections.face.bounding_box is not None:
            face_colour = _GREEN if detections.face.face_present else _RED
            face_label = "Face" if detections.face.face_present else "Face (missing)"
            self._draw_box(canvas, detections.face.bounding_box, face_colour, face_label)
        elif not detections.face.face_present:
            # No bounding box at all — place a small warning in the top-left
            cv2.putText(
                canvas,
                "NO FACE DETECTED",
                (10, 30),
                _FONT,
                _FONT_SCALE_HUD_TITLE,
                _RED,
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )

        # 2. Person boxes — blue
        for person_box in detections.objects.person_boxes:
            self._draw_box(canvas, person_box, _BLUE, "Person")

        # 3. Phone box — orange
        if detections.objects.phone_box is not None:
            self._draw_box(canvas, detections.objects.phone_box, _ORANGE, "Phone")

        # 4. Suspicious object boxes — red
        for sus_box in detections.objects.suspicious_objects:
            label = sus_box.label if sus_box.label else "Suspicious"
            self._draw_box(canvas, sus_box, _RED, label)

        # 5. HUD overlay (top-right)
        self._draw_hud(canvas, decision)

        # 6. Timestamp bar (bottom)
        self._draw_timestamp(canvas)

        return canvas

    # ── private helpers ──────────────────────────────────────────────

    def _draw_box(
        self,
        frame: np.ndarray,
        box: BoundingBox,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw a labelled bounding box on *frame*.

        Parameters
        ----------
        frame:
            Target image array (modified in-place).
        box:
            Axis-aligned bounding box coordinates.
        color:
            BGR colour for the rectangle and label background.
        label:
            Short text rendered above the top-left corner.
        """
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, _BOX_THICKNESS)

        # Label background
        text = f"{label} {box.confidence:.0%}" if box.confidence > 0.0 else label
        (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE_LABEL, _FONT_THICKNESS)
        label_y = max(box.y1 - 6, th + 4)
        cv2.rectangle(
            frame,
            (box.x1, label_y - th - 4),
            (box.x1 + tw + 6, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (box.x1 + 3, label_y - 2),
            _FONT,
            _FONT_SCALE_LABEL,
            _WHITE,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    def _draw_hud(self, frame: np.ndarray, decision: DecisionOutput) -> None:
        """Render a semi-transparent HUD panel in the top-right corner.

        Shows severity badge, risk score, and active violation reasons.

        Parameters
        ----------
        frame:
            Target image array (modified in-place).
        decision:
            Current decision engine output.
        """
        h, w = frame.shape[:2]

        # Calculate panel height based on content
        reason_lines: list[str] = decision.reasons if decision.reasons else ["None"]
        # Wrap long reason text
        wrapped_reasons: list[str] = []
        max_chars = 28
        for reason in reason_lines:
            while len(reason) > max_chars:
                split_idx = reason.rfind(" ", 0, max_chars)
                if split_idx == -1:
                    split_idx = max_chars
                wrapped_reasons.append(reason[:split_idx])
                reason = reason[split_idx:].lstrip()
            wrapped_reasons.append(reason)

        # Header (2 lines: severity, risk) + separator + reasons
        num_lines = 2 + 1 + len(wrapped_reasons)  # severity, risk, divider, reasons
        panel_h = _HUD_PADDING * 2 + num_lines * _HUD_LINE_HEIGHT
        panel_w = _HUD_WIDTH

        x1 = w - panel_w - 10
        y1 = 10
        x2 = w - 10
        y2 = y1 + panel_h

        # Clamp to frame bounds
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _PANEL_BG, cv2.FILLED)
        cv2.addWeighted(overlay, _HUD_ALPHA, frame, 1.0 - _HUD_ALPHA, 0, frame)

        # Border
        sev_color = self.SEVERITY_COLORS.get(decision.severity, _WHITE)
        cv2.rectangle(frame, (x1, y1), (x2, y2), sev_color, 2)

        # --- Content rendering ---
        tx = x1 + _HUD_PADDING
        ty = y1 + _HUD_PADDING + _HUD_LINE_HEIGHT

        # Line 1: Severity
        cv2.putText(
            frame,
            f"Severity: {decision.severity.value}",
            (tx, ty),
            _FONT,
            _FONT_SCALE_HUD_TITLE,
            sev_color,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
        ty += _HUD_LINE_HEIGHT

        # Line 2: Risk score
        cv2.putText(
            frame,
            f"Risk Score: {decision.risk_score}/100",
            (tx, ty),
            _FONT,
            _FONT_SCALE_HUD_BODY,
            _WHITE,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
        ty += _HUD_LINE_HEIGHT

        # Divider line
        cv2.line(frame, (tx, ty - 10), (x2 - _HUD_PADDING, ty - 10), sev_color, 1)

        # Reason lines
        for line in wrapped_reasons:
            cv2.putText(
                frame,
                line,
                (tx, ty),
                _FONT,
                _FONT_SCALE_HUD_BODY,
                _WHITE,
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )
            ty += _HUD_LINE_HEIGHT

    @staticmethod
    def _draw_timestamp(frame: np.ndarray) -> None:
        """Render a UTC timestamp bar at the bottom of the frame.

        Parameters
        ----------
        frame:
            Target image array (modified in-place).
        """
        h, w = frame.shape[:2]
        ts_text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        (tw, th), baseline = cv2.getTextSize(
            ts_text, _FONT, _FONT_SCALE_TIMESTAMP, _FONT_THICKNESS,
        )

        bar_h = th + baseline + 12
        bar_y1 = h - bar_h

        # Semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y1), (w, h), _BLACK, cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        text_x = w - tw - 10
        text_y = h - baseline - 6
        cv2.putText(
            frame,
            ts_text,
            (text_x, text_y),
            _FONT,
            _FONT_SCALE_TIMESTAMP,
            _WHITE,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
