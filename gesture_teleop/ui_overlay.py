from __future__ import annotations

"""Glassmorphic teleop UI.

Renders the camera feed plus a modern overlay:
  - resizable window (UI is composed at the actual window resolution, so it
    stays crisp at any size)
  - full-window green border glow while a hand is actively tracked
  - frosted-glass (blur + tint + rounded corners) status panel and bottom
    telemetry strip with smooth, read-only animated bars
  - anti-aliased, sub-pixel hand skeleton with a soft glow
  - SF Pro text if font files are dropped into gesture_teleop/fonts/,
    automatic fallback to Segoe UI, then to OpenCV Hershey.
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np

from config import AppConfig
from vision import HandObservation

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_OK = True
except ImportError:  # pragma: no cover - Pillow is in requirements.txt
    _PIL_OK = False

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# Palette (BGR)
ACCENT = (120, 210, 0)          # green accent
ACCENT_SOFT = (110, 190, 60)
AMBER = (60, 190, 255)
RED = (70, 70, 235)
TEXT_MAIN = (245, 245, 245)
TEXT_DIM = (170, 170, 170)
BAR_TRACK = (52, 50, 46)
SKELETON_GLOW = (40, 120, 255)
SKELETON_LINE = (80, 190, 255)
SKELETON_DOT = (255, 230, 120)

_SHIFT = 4  # sub-pixel bits for AA drawing
_S = 1 << _SHIFT


def _find_marker_path() -> Path | None:
    """The neutral-point marker image (transparent PNG) dropped into the
    gesture_teleop folder. Filename-agnostic so odd names/spaces still work."""
    d = Path(__file__).resolve().parent
    pngs = sorted(d.glob("*.png"))
    for p in pngs:
        n = p.name.lower()
        if any(k in n for k in ("logo", "agnisys", "crosshair", "marker")):
            return p
    return pngs[0] if pngs else None


def _find_font_paths() -> tuple[Path | None, Path | None]:
    """(regular, semibold). Prefers SF Pro files from gesture_teleop/fonts/,
    falls back to Segoe UI from the Windows font directory."""
    fonts_dir = Path(__file__).resolve().parent / "fonts"
    regular: Path | None = None
    semibold: Path | None = None
    if fonts_dir.is_dir():
        files = sorted(fonts_dir.glob("*.otf")) + sorted(fonts_dir.glob("*.ttf"))
        for f in files:
            name = f.name.lower()
            if any(k in name for k in ("semibold", "medium", "bold")):
                if semibold is None:
                    semibold = f
            elif regular is None:
                regular = f
        if regular is None and files:
            regular = files[0]
    if regular is None:
        windows_fonts = Path("C:/Windows/Fonts")
        if (windows_fonts / "segoeui.ttf").exists():
            regular = windows_fonts / "segoeui.ttf"
        if (windows_fonts / "seguisb.ttf").exists() and semibold is None:
            semibold = windows_fonts / "seguisb.ttf"
    return regular, semibold or regular


class _FontBank:
    def __init__(self) -> None:
        self._regular_path, self._semibold_path = _find_font_paths()
        self._cache: dict[tuple[int, bool], "ImageFont.FreeTypeFont"] = {}
        self._announce()

    def _announce(self) -> None:
        """Tell the user which font is in use, and flag an unusable .dmg drop."""
        import sys
        fonts_dir = Path(__file__).resolve().parent / "fonts"
        usable = self._regular_path is not None and "fonts" in str(self._regular_path).replace("\\", "/").lower()
        if not usable and fonts_dir.is_dir() and list(fonts_dir.glob("*.dmg")):
            print(
                "[ui] fonts/ contains a .dmg (macOS installer) which cannot be used on "
                "Windows. Extract the SF-Pro *.otf/*.ttf files into gesture_teleop/fonts/ "
                "to enable SF Pro; using the system font for now.",
                file=sys.stderr,
            )

    @property
    def usable(self) -> bool:
        return _PIL_OK and self._regular_path is not None

    def get(self, size: int, bold: bool = False):
        if not self.usable:
            return None
        key = (size, bold)
        font = self._cache.get(key)
        if font is None:
            path = self._semibold_path if bold else self._regular_path
            font = ImageFont.truetype(str(path), size)
            self._cache[key] = font
        return font

    def measure(self, text: str, size: int, bold: bool = False) -> float | None:
        font = self.get(size, bold)
        return None if font is None else float(font.getlength(text))


def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    r = max(1, min(radius, w // 2, h // 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _rounded_outline(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                     color: tuple, radius: int, thickness: int = 1) -> None:
    r = max(1, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    lt = cv2.LINE_AA
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, lt)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, lt)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, lt)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, lt)
    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, thickness, lt)
    cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, thickness, lt)
    cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, thickness, lt)
    cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, thickness, lt)


class TeleopUI:
    def __init__(self, window_name: str, initial_size: tuple[int, int] = (1280, 720)) -> None:
        self.window = window_name
        # WINDOW_GUI_NORMAL disables the Qt "enhanced" GUI (the toolbar seen on
        # Linux builds). That toolbar makes getWindowImageRect report a stale/
        # smaller size, so the canvas was composed too small and the window
        # padded it with grey borders. The plain GUI fills the client area and
        # reports the real size, matching the Windows (Win32) behaviour.
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_GUI_NORMAL"):
            flags |= cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(window_name, flags)
        cv2.resizeWindow(window_name, *initial_size)
        self._fonts = _FontBank()
        self._bar_state: dict[str, float] = {}
        self._glow_ramps: dict[tuple[int, int], np.ndarray] = {}
        self._texts: list[tuple[str, int, int, int, tuple, bool]] = []

        # Neutral-point marker image (transparent PNG); cache resized versions.
        self._marker: tuple[np.ndarray, np.ndarray] | None = None
        self._marker_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        marker_path = _find_marker_path()
        if marker_path is not None:
            img = cv2.imread(str(marker_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 4:
                    self._marker = (img[:, :, :3].copy(), img[:, :, 3].copy())
                else:
                    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    self._marker = (bgr, np.full(bgr.shape[:2], 255, np.uint8))

    # ------------------------------------------------------------------ text
    def _text(self, text: str, x: int, y: int, size: int,
              color: tuple = TEXT_MAIN, bold: bool = False) -> None:
        self._texts.append((text, x, y, size, color, bold))

    def _flush_texts(self, canvas: np.ndarray) -> np.ndarray:
        if not self._texts:
            return canvas
        if self._fonts.usable:
            pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for text, x, y, size, color, bold in self._texts:
                font = self._fonts.get(size, bold)
                draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
            canvas = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        else:  # Hershey fallback
            for text, x, y, size, color, bold in self._texts:
                scale = size / 26.0
                cv2.putText(canvas, text, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 2 if bold else 1, cv2.LINE_AA)
        self._texts.clear()
        return canvas

    def _text_width(self, text: str, size: int, bold: bool = False) -> float:
        w = self._fonts.measure(text, size, bold) if self._fonts.usable else None
        if w is None:
            scale = size / 26.0
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 if bold else 1)
            return float(tw)
        return w

    # --------------------------------------------------------------- logo mark
    def _draw_marker(self, canvas: np.ndarray, cx: float, cy: float, target_h: float) -> None:
        """Alpha-composite the transparent logo PNG centered at (cx, cy),
        scaled to `target_h` pixels tall (aspect preserved). Falls back to a
        small amber cross if the image is missing."""
        if self._marker is None:
            p = (int(round(cx)), int(round(cy)))
            cv2.drawMarker(canvas, p, AMBER, cv2.MARKER_CROSS, max(8, int(target_h)), 1, cv2.LINE_AA)
            return

        bgr, alpha = self._marker
        h0, w0 = bgr.shape[:2]
        th = max(8, int(round(target_h)))
        tw = max(8, int(round(w0 * th / h0)))
        cache = self._marker_cache.get((tw, th))
        if cache is None:
            rb = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
            ra = (cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_AREA)
                  .astype(np.float32) / 255.0)[..., None]
            cache = (rb, ra)
            self._marker_cache[(tw, th)] = cache
        rb, ra = cache

        x0, y0 = int(round(cx - tw / 2.0)), int(round(cy - th / 2.0))
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(canvas.shape[1], x0 + tw), min(canvas.shape[0], y0 + th)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sx0, sy0 = cx0 - x0, cy0 - y0
        fg = rb[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)]
        a = ra[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)]
        roi = canvas[cy0:cy1, cx0:cx1]
        roi[:] = (fg * a + roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)

    # ----------------------------------------------------------------- panels
    def _glass(self, canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               radius: int = 16) -> None:
        h, w = canvas.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        pw, ph = x1 - x0, y1 - y0
        if pw < 8 or ph < 8:
            return
        roi = canvas[y0:y1, x0:x1]
        # Liquid-glass: moderate blur + a light, translucent frost. Low tint
        # weight keeps the panel see-through (more of the background reads).
        small = cv2.resize(roi, (max(1, pw // 6), max(1, ph // 6)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3.2)
        blurred = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_LINEAR)
        tint = np.full_like(blurred, (42, 40, 38))
        glass = cv2.addWeighted(blurred, 0.72, tint, 0.08, 5)
        mask = _rounded_mask(pw, ph, radius)
        roi[:] = np.where(mask[..., None] > 0, glass, roi)
        _rounded_outline(canvas, x0, y0, x1 - 1, y1 - 1, (185, 182, 176), radius, 1)
        # subtle top highlight for the glass sheen
        cv2.line(canvas, (x0 + radius, y0 + 1), (x1 - radius, y0 + 1), (230, 228, 220), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ glow
    def _edge_ramp(self, depth: int, length: int) -> np.ndarray:
        key = (depth, length)
        ramp = self._glow_ramps.get(key)
        if ramp is None:
            fall = (1.0 - (np.arange(depth, dtype=np.float32) / max(depth - 1, 1))) ** 2.2
            ramp = np.repeat(fall[:, None], length, axis=1)[..., None]  # (d, len, 1)
            self._glow_ramps[key] = ramp
        return ramp

    def _border_glow(self, canvas: np.ndarray, intensity: float,
                     color: tuple = ACCENT) -> None:
        if intensity <= 0.0:
            return
        h, w = canvas.shape[:2]
        d = max(6, min(int(min(h, w) * 0.035), 42))
        col = np.array(color, dtype=np.float32)

        def blend(strip: np.ndarray, alpha: np.ndarray) -> None:
            a = alpha * intensity
            strip[:] = (strip.astype(np.float32) * (1.0 - a) + col * a).astype(np.uint8)

        blend(canvas[0:d, :], self._edge_ramp(d, w))                       # top
        blend(canvas[h - d:h, :], self._edge_ramp(d, w)[::-1])             # bottom
        blend(canvas[:, 0:d], self._edge_ramp(d, h).transpose(1, 0, 2))    # left
        blend(canvas[:, w - d:w], self._edge_ramp(d, h).transpose(1, 0, 2)[:, ::-1])  # right

    # ------------------------------------------------------------------ bars
    def _bar(self, canvas: np.ndarray, key: str, label: str, value_text: str,
             value01: float, x: int, y: int, w: int, h: int,
             color: tuple = ACCENT_SOFT, label_size: int = 12) -> None:
        value01 = float(np.clip(value01, 0.0, 1.0))
        shown = self._bar_state.get(key)
        shown = value01 if shown is None else shown + 0.30 * (value01 - shown)
        self._bar_state[key] = shown

        self._text(label, x, y - label_size - 6, label_size, TEXT_DIM)
        if value_text:
            self._text(value_text, x + w - 7 * len(value_text), y - label_size - 6,
                       label_size, TEXT_MAIN)
        radius = h // 2
        track = np.full((h, w, 3), BAR_TRACK, np.uint8)
        mask = _rounded_mask(w, h, radius)
        roi = canvas[y:y + h, x:x + w]
        if roi.shape[0] != h or roi.shape[1] != w:
            return
        roi[:] = np.where(mask[..., None] > 0, track, roi)
        fill_w = int(round(shown * w))
        if fill_w > 2:
            fill = np.full((h, fill_w, 3), color, np.uint8)
            fmask = _rounded_mask(fill_w, h, radius) if fill_w > h else _rounded_mask(fill_w, h, fill_w // 2)
            froi = canvas[y:y + h, x:x + fill_w]
            froi[:] = np.where(fmask[..., None] > 0, fill, froi)

    # -------------------------------------------------------------- skeleton
    def _skeleton(self, canvas: np.ndarray, observation: HandObservation,
                  fw: int, fh: int, scale: float, ox: int, oy: int,
                  line_color: tuple = SKELETON_LINE) -> None:
        lm = observation.normalized_landmarks
        pts = np.empty((lm.shape[0], 2), np.float32)
        pts[:, 0] = lm[:, 0] * fw * scale + ox
        pts[:, 1] = lm[:, 1] * fh * scale + oy

        # soft glow underlay, blended only inside the hand bounding box
        pad = 26
        x0 = int(max(0, pts[:, 0].min() - pad))
        y0 = int(max(0, pts[:, 1].min() - pad))
        x1 = int(min(canvas.shape[1], pts[:, 0].max() + pad))
        y1 = int(min(canvas.shape[0], pts[:, 1].max() + pad))
        if x1 - x0 > 4 and y1 - y0 > 4:
            roi = canvas[y0:y1, x0:x1]
            overlay = roi.copy()
            local = ((pts - (x0, y0)) * _S).astype(np.int32)
            for a, b in HAND_CONNECTIONS:
                cv2.line(overlay, tuple(local[a]), tuple(local[b]),
                         SKELETON_GLOW, 7, cv2.LINE_AA, shift=_SHIFT)
            cv2.addWeighted(overlay, 0.35, roi, 0.65, 0, dst=roi)

        # crisp sub-pixel lines + joints
        fixed = (pts * _S).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, tuple(fixed[a]), tuple(fixed[b]),
                     line_color, 2, cv2.LINE_AA, shift=_SHIFT)
        for p in fixed:
            cv2.circle(canvas, tuple(p), 3 * _S, SKELETON_DOT, -1, cv2.LINE_AA, shift=_SHIFT)
            cv2.circle(canvas, tuple(p), 3 * _S, (30, 30, 30), 1, cv2.LINE_AA, shift=_SHIFT)

    # ------------------------------------------------------------------ main
    def render(self, frame: np.ndarray, config: AppConfig,
               observation: HandObservation | None, runtime: dict) -> None:
        fh, fw = frame.shape[:2]

        try:
            _, _, ww, wh = cv2.getWindowImageRect(self.window)
        except cv2.error:
            ww, wh = fw, fh
        if ww <= 0 or wh <= 0:
            ww, wh = fw, fh
        ww, wh = max(480, ww), max(300, wh)

        # fill_window=True -> cover (crop overflow, no bars); else contain (letterbox).
        if getattr(config.camera, "fill_window", False):
            scale = max(ww / fw, wh / fh)
        else:
            scale = min(ww / fw, wh / fh)
        dw, dh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
        ox, oy = (ww - dw) // 2, (wh - dh) // 2

        canvas = np.full((wh, ww, 3), (14, 13, 12), np.uint8)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (dw, dh), interpolation=interp)
        # Clipped blit so cover mode (ox/oy < 0, dw/dh > window) can't overflow.
        dx0, dy0 = max(0, ox), max(0, oy)
        dx1, dy1 = min(ww, ox + dw), min(wh, oy + dh)
        canvas[dy0:dy1, dx0:dx1] = resized[dy0 - oy:dy1 - oy, dx0 - ox:dx1 - ox]

        ui = max(0.72, min(ww / 1280.0, 1.7))  # UI scale with window size

        hands2 = runtime.get("hands2")
        if hands2 is not None:
            # ------- two-hand scheme: divider, L/R anchors, skeletons, lines
            dx = int(ox + 0.5 * fw * scale)
            cv2.line(canvas, (dx, oy), (dx, oy + dh), (72, 70, 66), 1, cv2.LINE_AA)
            role_colors = {"R": ACCENT_SOFT, "L": AMBER}
            for entry in hands2:
                color = role_colors.get(entry.get("role"), SKELETON_LINE)
                axn, ayn = entry.get("anchor", (0.5, 0.5))
                axp = axn * fw * scale + ox
                ayp = ayn * fh * scale + oy
                self._draw_marker(canvas, axp, ayp, 26.0 * ui)
                self._text(entry.get("role", "?"), int(axp + 15 * ui), int(ayp - 22 * ui),
                           int(14 * ui), color, bold=True)
                obs = entry.get("obs")
                if obs is None:
                    continue
                self._skeleton(canvas, obs, fw, fh, scale, ox, oy, line_color=color)
                hx = obs.center_xy[0] * fw * scale + ox
                hy = obs.center_xy[1] * fh * scale + oy
                d = float(entry.get("deflection", 0.0))
                line_col = tuple(int(c * (0.35 + 0.65 * d)) for c in color)
                cv2.line(canvas, (int(axp), int(ayp)), (int(hx), int(hy)),
                         line_col, 2, cv2.LINE_AA)
                tag = entry["role"] + (" LOCK" if entry.get("locked") else "")
                wx, wy = obs.pixel_landmarks[0]
                self._text(tag, int(wx * scale + ox) - int(8 * ui),
                           int(wy * scale + oy) + int(10 * ui), int(12 * ui), color, True)
        else:
            # ------- single-hand (direct) scheme: one neutral marker + skeleton
            nx = config.vision.neutral_x * fw * scale + ox
            ny = config.vision.neutral_y * fh * scale + oy
            self._draw_marker(canvas, nx, ny, 30.0 * ui)
            if observation is not None:
                self._skeleton(canvas, observation, fw, fh, scale, ox, oy)

        # ---------------- status panel (top-left) ----------------
        pad = int(16 * ui)
        line_h = int(20 * ui)
        size_body = int(13 * ui)
        size_title = int(15 * ui)
        size_small = int(11 * ui)

        frozen = bool(runtime.get("frozen"))
        estop = bool(runtime.get("estop"))
        hand_active = bool(runtime.get("hand_active"))
        depth_hold = bool(runtime.get("depth_hold"))
        roll_active = bool(runtime.get("roll_active"))

        holds = [name for flag, name in ((depth_hold, "DEPTH"), (roll_active, "ROLL")) if flag]
        if hands2 is not None:
            for entry in hands2:
                if entry.get("role") == "R" and entry.get("locked"):
                    holds.append("R-LOCK")
                if entry.get("role") == "L" and entry.get("obs") is None:
                    holds.append("L-HOLD")

        # Gripper open/closed indicator (gripper_open is the actual openness:
        # 1.0 = open, 0.0 = closed, independent of the hand-curl invert setting).
        grip_open_frac = float(runtime.get("gripper_open", 0.0))
        if grip_open_frac >= 0.55:
            grip_state, grip_color = "OPEN", ACCENT
        elif grip_open_frac <= 0.45:
            grip_state, grip_color = "CLOSED", AMBER
        else:
            grip_state, grip_color = "PARTIAL", TEXT_MAIN

        state_bits = []
        if estop:
            state_bits.append("E-STOP")
        if frozen:
            state_bits.append("FROZEN")
        state_line = "  ".join(state_bits) if state_bits else "running"

        lines: list[tuple[str, int, tuple, bool]] = [
            (str(runtime.get("status", "idle")), size_title, TEXT_MAIN, True),
            (f"phase {runtime.get('phase', '-')}   mode {runtime.get('mode', '-')}   "
             f"map {runtime.get('control_mode', '-')}", size_body, TEXT_DIM, False),
            (f"tracking {'yes' if runtime.get('tracking_ok') else 'no'}   "
             f"state {state_line}", size_body,
             RED if (estop or frozen) else TEXT_DIM, False),
            (f"ramp {runtime.get('transition_rate', 0.0):.0f} deg/s   "
             f"speed {runtime.get('live_scale', 1.0):.2f}x   "
             f"fps {runtime.get('fps', 0.0):.0f}", size_body, TEXT_DIM, False),
            (f"gripper: {grip_state}   ({int(grip_open_frac * 100)}% open, "
             f"{runtime.get('gripper_deg', 0.0):.0f} deg)",
             size_body, grip_color, False),
            (f"holds {' + '.join(holds) if holds else '-'}   "
             f"link {runtime.get('transport_status', 'unknown')}", size_body,
             AMBER if holds else TEXT_DIM, False),
            ("Q quit   F freeze   X e-stop   H home   N neutral   "
             "up/down ramp   +/- speed", size_small, TEXT_DIM, False),
        ]

        # Size the panel to the widest line so nothing clips (fixes the keys
        # row overflowing) and it stays aligned at any window size.
        content_w = max(self._text_width(t, s, b) for t, s, _, b in lines)
        panel_w = min(int(content_w + pad * 2 + 6), ww - pad * 2)
        panel_h = pad * 2 + line_h * len(lines)
        px0, py0 = pad, pad
        self._glass(canvas, px0, py0, px0 + panel_w, py0 + panel_h, radius=int(14 * ui))
        ty = py0 + pad - int(3 * ui)
        for text, size, color, bold in lines:
            self._text(text, px0 + pad, ty, size, color, bold)
            ty += line_h

        # ---------------- bottom telemetry strip ----------------
        strip_h = int(116 * ui)
        sx0, sy1 = pad, wh - pad
        sx1, sy0 = ww - pad, wh - pad - strip_h
        if sx1 - sx0 > 320:
            self._glass(canvas, sx0, sy0, sx1, sy1, radius=int(14 * ui))
            inner = int(14 * ui)
            bar_h = max(6, int(8 * ui))
            row1_y = sy0 + inner + int(18 * ui)
            row2_y = row1_y + int(46 * ui)
            avail = (sx1 - sx0) - inner * 2

            # Normalize each bar over the joint's ACTUAL usable range so the
            # fill is meaningful across the configured p2 direction.
            lim = config.limits
            avg_lo = max(0.5 * (lim.p1_min + lim.p3_min), lim.p2_min)
            avg_hi = min(0.5 * (lim.p1_max + lim.p3_max), lim.p2_max)
            if config.mapping.p2_invert:
                p2_lo, p2_hi = 180.0 - avg_hi, 180.0 - avg_lo
            else:
                p2_lo, p2_hi = avg_lo, avg_hi
            joints = [
                ("base", "BASE", runtime.get("base_deg", 0.0), lim.base_min, lim.base_max),
                ("p1", "P1", runtime.get("p1_deg", 0.0), lim.p1_min, lim.p1_max),
                ("p2", "P2", runtime.get("p2_deg", 0.0), p2_lo, p2_hi),
                ("p3", "P3", runtime.get("p3_deg", 0.0), lim.p3_min, lim.p3_max),
                ("roll", "ROLL", runtime.get("roll_deg", 0.0), lim.roll_min, lim.roll_max),
            ]
            gap = int(16 * ui)
            n1 = len(joints) + 1  # + gripper
            bw1 = max(60, (avail - gap * (n1 - 1)) // n1)
            x = sx0 + inner
            for key, label, deg, lo, hi in joints:
                norm = (float(deg) - lo) / max(hi - lo, 1e-6)
                self._bar(canvas, key, label, f"{deg:5.1f}", norm,
                          x, row1_y, bw1, bar_h, ACCENT_SOFT, size_small)
                x += bw1 + gap
            grip_open = float(runtime.get("gripper_open", 0.0))
            self._bar(canvas, "grip", "GRIP", f"{int(grip_open * 100)}%", grip_open,
                      x, row1_y, bw1, bar_h, AMBER, size_small)

            r_height = runtime.get("r_height")
            l_height = runtime.get("l_height")
            if r_height is not None:
                # two-hand: height drives two joints (R->p2, L->p3) — show both,
                # each color-coded to its role (R green, L amber)
                height_bars = [
                    ("in_rh", "R HEIGHT", float(r_height), ACCENT_SOFT),
                    ("in_lh", "L HEIGHT", float(l_height), AMBER),
                ]
            else:
                height_bars = [
                    ("in_h", "HEIGHT", float(runtime.get("height_norm", 0.5)), ACCENT_SOFT),
                ]
            inputs = [
                ("in_x", "YAW X", float(runtime.get("x_norm", 0.5)), ACCENT_SOFT),
                *height_bars,
                ("in_d", "DEPTH", float(runtime.get("depth_norm", 0.5)),
                 AMBER if depth_hold else ACCENT_SOFT),
                ("in_r", "ROLL IN", 0.5 + 0.5 * float(runtime.get("roll_input", 0.0)),
                 AMBER if roll_active else ACCENT_SOFT),
                ("in_g", "GRIP IN", float(runtime.get("grip_norm", 0.0)), ACCENT_SOFT),
            ]
            n2 = len(inputs)
            bw2 = max(60, (avail - gap * (n2 - 1)) // n2)
            x = sx0 + inner
            for key, label, val, color in inputs:
                self._bar(canvas, key, label, f"{int(np.clip(val, 0, 1) * 100)}%", val,
                          x, row2_y, bw2, bar_h, color, size_small)
                x += bw2 + gap

        # ---------------- full-window border glow ----------------
        if hand_active:
            pulse = 0.82 + 0.18 * math.sin(time.perf_counter() * 2.0 * math.pi / 1.8)
            self._border_glow(canvas, 0.85 * pulse, ACCENT)
        elif runtime.get("tracking_ok"):
            self._border_glow(canvas, 0.30, (140, 140, 140))

        canvas = self._flush_texts(canvas)
        cv2.imshow(self.window, canvas)
