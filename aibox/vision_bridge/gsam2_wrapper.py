from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional

# Grounding DINO helper (tiny Swin-T backbone)
from groundingdino.util.inference import Model as GDINOModel
from torchvision.ops import box_convert


class GSAM2Wrapper:
    """A *minimal* wrapper that exposes a YOLO-compatible interface
    around Grounding-DINO (+ optional SAM-2 in future).

    In this first iteration we only rely on Grounding-DINO to localise the
    object every frame.  That is already enough for the bracelet navigation
    pipeline because it consumes plain bounding-boxes.  SAM-2 temporal
    tracking (mask propagation) can be plugged later without changing the
    public API.

    Public API
    ----------
    set_prompt(frame_rgb, text)
        Analyse *frame_rgb* once with *text* prompt and remember it.
        Currently we merely store the text; *frame_rgb* is ignored but kept
        for API parity with potential future SAM-2 initialisation.
    track(frame_rgb, depth_img=None) -> list[tuple]
        Return a list with **zero or one** detection tuples in the format
        required by `bracelet.py`::

            (xc, yc, w, h, track_id, class_id, conf, depth)
    """

    # ---------------------------------------------------------------------
    # Construction helpers - Updated paths for new structure
    # ---------------------------------------------------------------------
    _CONF_PATH = (
        Path(__file__).resolve().parent.parent
        / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    _CKPT_PATH = (
        Path(__file__).resolve().parent.parent
        / "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    )

    def __init__(
        self,
        device: str | torch.device | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        default_prompt: str = "coffee cup",
    ) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Load Grounding-DINO once.
        self._gdino = GDINOModel(
            model_config_path=str(self._CONF_PATH),
            model_checkpoint_path=str(self._CKPT_PATH),
            device=str(self.device),
        )

        self._prompt: str = default_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_prompt(self, frame_rgb: np.ndarray | None, text: str) -> None:
        """(Re)sets the text prompt for detection.

        Parameters
        ----------
        frame_rgb : np.ndarray | None
            Currently unused – kept for future SAM-2 priming.
        text : str
            The free-form text prompt (e.g. "red bottle").
        """
        self._prompt = text
        # Placeholder: when SAM-2 tracking is added we will prime it here.

    @torch.inference_mode()
    def track(
        self,
        frame_bgr: np.ndarray,
        depth_img: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Detect the target object in *frame_bgr*.

        Returns
        -------
        list[np.ndarray]
            Either empty or a single ndarray with shape (8,) holding:
            ``[xc, yc, w, h, track_id, class_id, conf, depth]`` in the
            Detection format expected by the controller.
        """
        # Grounding-DINO expects BGR OpenCV image (height, width, 3)
        detections, _ = self._gdino.predict_with_caption(
            image=frame_bgr,
            caption=self._prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        if len(detections.xyxy) == 0:
            return []

        # Pick the highest-confidence detection.
        idx = int(np.argmax(detections.confidence))
        x1, y1, x2, y2 = detections.xyxy[idx]
        conf = float(detections.confidence[idx])

        # Convert from xyxy to xywh (center format)
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Build detection tuple in Detection format: (xc, yc, w, h, track_id, class_id, conf, depth)
        # track_id = -1 (no tracker), class_id = 0 (placeholder), depth = -1 (placeholder)
        det = np.array([xc, yc, w, h, -1, 0, conf, -1], dtype=float)
        return [det] 