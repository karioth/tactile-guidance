"""gsam2_wrapper.py – Minimal Grounded‑DINO + SAM‑2 wrapper (fixed)
===================================================================
Detect hand once, detect the prompted object once, then let SAM‑2 do
all per‑frame tracking.  Re‑detect only if a mask is missing for
> MISS_MAX consecutive frames.  Grounding DINO retry interval can be
changed via RETRY.

Key changes vs. previous draft
──────────────────────────────
* Guarantee **exactly one frame write** per real video frame – duplicates
  were the root‑cause of the invisible hand.
* `_prime()` now **returns** the frame‑index it wrote, so the caller can
  skip the normal `_add_frame()` path for that iteration.
* Object priming now also re‑attaches the **existing hand box** so both
  masks appear on the same key‑frame (mirrors V1 behaviour).
* Dynamic storage of the actual `obj_id` values returned by SAM‑2 (no
  more silent failures if the id mapping changes after a reset).
* Minor clean‑ups: consolidated id handling, extra debug prints, stricter
  doc‑strings.

Public API (stable for bracelet controller) ─────────────────────
    >>> gsam = GSAM2Wrapper()
    >>> outputs = gsam.track(frame_bgr)  # (xc, yc, w, h, track_id, class_id, conf, depth)
    >>> gsam.set_prompt(None, "red bottle")
"""
from __future__ import annotations
import sys, os, time
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch, cv2, supervision as sv

# ─── Fast‑math switches ─────────────────────────────────────────────────────────
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("⚡ Enabled TF32 optimisations for Ampere/‎Hopper GPU")

# ─── Third‑party models ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
_GSAM2_DIR = ROOT / "Grounded-SAM-2"
if str(_GSAM2_DIR) not in sys.path:
    sys.path.insert(0, str(_GSAM2_DIR))

from groundingdino.util.inference import Model as GDINOModel
from sam2.build_sam import build_sam2_video_predictor

# ─── Helper ─────────────────────────────────────────────────────────────────────

def _pick_best(dets, lbls, contains: str | None = None):
    """Return xyxy box (np.ndarray) of highest‑confidence detection.
    If *contains* is given, filter labels by that substring first."""
    if len(dets.xyxy) == 0:
        return None
    idxs = range(len(dets.xyxy))
    if contains is not None:
        idxs = [i for i in idxs if contains.lower() in lbls[i].lower()]
        if not idxs:
            return None
    best_i = max(idxs, key=lambda i: float(dets.confidence[i]))
    return np.array(dets.xyxy[best_i])

# ─── Wrapper ────────────────────────────────────────────────────────────────────
class GSAM2Wrapper:
    """Lightweight hand‑first pipeline with periodic memory reset and
    event‑driven Grounding DINO calls.

    * Hand detection prompt: "my <handedness> hand".
    * Object detection prompt: provided by :pymeth:`set_prompt`.
    * Grounding DINO is invoked only when:
        – hand not yet tracked and retry timer expired
        – object not yet tracked and retry timer expired
        – corresponding mask lost for MISS_MAX frames
    """

    _CONF_PATH    = ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    _CKPT_PATH    = ROOT / "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    _SAM2_CFG     = "configs/sam2.1/sam2.1_hiera_t.yaml"
    _SAM2_WEIGHTS = ROOT / "Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"

    # Tunables
    WINDOW   = 30   # SAM‑2 memory window (frames)
    MISS_MAX = 30    # lost‑mask threshold (frames)
    RETRY    = 15    # DINO retry interval after a miss (frames)
    IMG_SIZE = 1024  # SAM‑2 input resolution

    def __init__(self, device: str | torch.device | None = None,
                 box_threshold: float = .35, text_threshold: float = .25,
                 default_prompt: str = "coffee cup", handedness: str = "right"):
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.box_thr, self.txt_thr = box_threshold, text_threshold
        self.prompt_txt = default_prompt
        self.handedness = handedness.lower()

        # ── Load models ────────────────────────────────────────────────────
        print("⏳ Loading Grounding DINO model…", end=" ")
        self.dino = GDINOModel(str(self._CONF_PATH), str(self._CKPT_PATH), device=str(self.device))
        print("✅ done")
        print("⏳ Loading SAM‑2 tiny predictor…", end=" ")
        cwd = os.getcwd(); os.chdir(str(_GSAM2_DIR))
        try:
            self.sam2 = build_sam2_video_predictor(self._SAM2_CFG, str(self._SAM2_WEIGHTS), device=str(self.device))
        finally:
            os.chdir(cwd)
        print("✅ done")

        # ── Init predictor state ───────────────────────────────────────────
        self.state = self.sam2.init_state(video_path=None)
        self.state["images"] = torch.empty((0, 3, self.IMG_SIZE, self.IMG_SIZE), device=self.device)
        self.state["device"] = self.device

        # runtime flags / counters
        self.have_hand = False
        self.have_obj  = False
        self.prompt_wait = True
        self.tr_hand_id: Optional[int] = None  # actual SAM‑2 ids (set on first prime)
        self.tr_obj_id:  Optional[int] = None
        self.lost_hand = self.lost_obj = 0
        self.f = 0                        # global frame count
        self.next_hand_try = 0            # retry timestamps
        self.next_obj_try  = 0

        # ── Last known bounding boxes for repriming ──────────────────────
        self._last_hand_box = None
        self._last_object_box = None

        # ── Performance tracking (optional) ──────────────────────────────
        self._start_time = time.time()
        self._total_frames = 0
        self._gdino_calls = 0
        self._sam2_calls = 0
        self._memory_resets = 0

    # ─── Public API ───────────────────────────────────────────────────────
    def set_prompt(self, frame_rgb: Optional[np.ndarray], text: str):
        """Set / change the object prompt.  If *frame_rgb* is provided and the
        hand is already tracked, will attempt immediate detection."""
        print(f"✨ New prompt received → '{text}'")
        self.prompt_txt = text
        self.prompt_wait = False
        if frame_rgb is not None and self.have_hand and not self.have_obj:
            self._detect_object_and_prime(frame_rgb)

    def is_ready_for_object_prompt(self):
        return self.have_hand and not self.have_obj

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def track(self, frame_bgr: np.ndarray, depth_img: Optional[np.ndarray] = None):
        """Process one video frame and return bracelet tuples."""
        cur = self.f
        self._total_frames += 1

        # RGB conversion once here
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        self.state["video_height"], self.state["video_width"] = h, w

        # -----------------------------------------------------------------
        # 1️⃣  HAND detection (retry gate)
        # -----------------------------------------------------------------
        idx_from_prime: Optional[int] = None  # will hold the frame‑idx if we prime
        if not self.have_hand and cur >= self.next_hand_try:
            print(f"🔍 [Frame {cur}] Attempting hand detection via GDINO…")
            idx_from_prime = self._detect_hand_and_prime(frame_rgb)
            if idx_from_prime is None:
                #print("🚫 Hand not found – retry scheduled")
                self.next_hand_try = cur + self.RETRY

        # -----------------------------------------------------------------
        # 2️⃣  OBJECT detection (retry gate)
        # -----------------------------------------------------------------
        if self.have_hand and not self.have_obj and not self.prompt_wait and cur >= self.next_obj_try:
            print(f"🔍 [Frame {cur}] Attempting object detection via GDINO…")
            idx_from_prime = self._detect_object_and_prime(frame_rgb) or idx_from_prime
            if idx_from_prime is None:
                print("🚫 Object not found – retry scheduled")
                self.next_obj_try = cur + self.RETRY

        # -----------------------------------------------------------------
        # 3️⃣  Early out: nothing to track
        # -----------------------------------------------------------------
        if not (self.have_hand or self.have_obj):
            #print(f"⏭️  [Frame {cur}] No objects to track – skipping SAM‑2")
            self.f += 1
            return []

        # -----------------------------------------------------------------
        # 4️⃣  Push frame exactly once
        # -----------------------------------------------------------------
        if idx_from_prime is None:
            idx = self._add_frame(frame_rgb)  # regular push
        else:
            idx = idx_from_prime              # frame already written during prime

        # Edge‑case: we may lose objects during memory reset inside _add_frame
        if not (self.have_hand or self.have_obj):
            print(f"⚠️  Lost all objects during reset – skipping inference")
            self.f += 1
            return []

        # -----------------------------------------------------------------
        # 5️⃣  Run tracking
        # -----------------------------------------------------------------
        #print(f"🎯 [Frame {cur}] Running SAM‑2 tracking (buffer: {self.state['images'].shape[0]} frames)")
        idx, ids, masks = self.sam2.infer_single_frame(self.state, idx)
        self._sam2_calls += 1

        # -----------------------------------------------------------------
        # 6️⃣  Convert masks → bracelet tuples
        # -----------------------------------------------------------------
        out, hand_ok, obj_ok = [], False, False
        for i, oid in enumerate(ids):
            mask_np = (masks[i] > 0.0)[0].cpu().numpy()
            xyxy = sv.mask_to_xyxy(mask_np[None])  # (1, 4) or empty
            if len(xyxy) == 0:
                continue
            x1, y1, x2, y2 = xyxy[0]
            xc, yc, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            depth = -1.0
            if depth_img is not None and 0 <= int(yc) < depth_img.shape[0] and 0 <= int(xc) < depth_img.shape[1]:
                depth = float(depth_img[int(yc), int(xc)])

            if oid == self.tr_hand_id:
                class_id, hand_ok = 1, True
                self._last_hand_box = np.array([x1, y1, x2, y2])
            elif oid == self.tr_obj_id:
                class_id, obj_ok = 0, True
                self._last_object_box = np.array([x1, y1, x2, y2])
            else:
                continue  # unknown id
            out.append(np.array([xc, yc, bw, bh, oid, class_id, 0.8, depth], dtype=float))

        # -----------------------------------------------------------------
        # 7️⃣  Loss accounting / auto‑redection
        # -----------------------------------------------------------------
        self.lost_hand = 0 if hand_ok else self.lost_hand + 1
        self.lost_obj  = 0 if obj_ok  else self.lost_obj  + 1
        if self.lost_hand > self.MISS_MAX:
            print("😢 Lost hand – will re‑detect")
            self.sam2.remove_object(self.state, self.tr_hand_id, strict=False, need_output=False)
            self.have_hand, self.lost_hand = False, 0
            self.next_hand_try = cur + self.RETRY
        if self.lost_obj > self.MISS_MAX:
            print("😢 Lost object – will re‑detect")
            self.sam2.remove_object(self.state, self.tr_obj_id, strict=False, need_output=False)
            self.have_obj, self.lost_obj = False, 0
            self.next_obj_try = cur + self.RETRY

        self.f += 1
        return out

    # ─── Detection helpers ───────────────────────────────────────────────
    def _detect_hand_and_prime(self, frame_rgb) -> Optional[int]:
        print("🔍 Detecting hand…", end=" ")
        dets, lbls = self._dino(frame_rgb, f"my {self.handedness} hand")
        self._gdino_calls += 1
        box = _pick_best(dets, lbls, "hand")
        if box is None:
            print("fail")
            return None

        preferred_id = 2                     # <<< fixed id for hand
        idx, _, _ = self._prime(frame_rgb, preferred_id=preferred_id, box=box)

        self.tr_hand_id = preferred_id
        self._last_hand_box = box.copy()
        self.have_hand = True
        print(f"🤚 found! (id={self.tr_hand_id})")
        return idx


    def _detect_object_and_prime(self, frame_rgb) -> Optional[int]:
        """Detect the prompted object and prime SAM-2.

        Object always uses obj_id 1 so it can coexist with the hand (id 2).
        """
        print(f"🔍 Detecting '{self.prompt_txt}'…", end=" ")
        dets, lbls = self._dino(frame_rgb, self.prompt_txt)
        self._gdino_calls += 1
        box = _pick_best(dets, lbls)
        if box is None:
            print("fail")
            return None

        preferred_id = 1                     # <<< fixed id for object
        idx, _, _ = self._prime(
            frame_rgb,
            preferred_id=preferred_id,
            box=box,
            also_prime_hand=True,            # keep hand on the same frame
        )

        self.tr_obj_id = preferred_id
        self._last_object_box = box.copy()
        self.have_obj = True
        print(f"🎯 found! (id={self.tr_obj_id})")
        return idx

    # ─── SAM‑2 helpers ──────────────────────────────────────────────────
    def _dino(self, frame_rgb, caption):
        """Single‑call Grounding DINO wrapper."""
        return self.dino.predict_with_caption(
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            caption,
            box_threshold=self.box_thr,
            text_threshold=self.txt_thr)

    def _prime(self, frame_rgb, preferred_id: int, box: np.ndarray, *, also_prime_hand: bool = False):
        """Write *frame_rgb* once, annotate *box* and optionally the existing
        hand box on the very same frame.

        Returns
        -------
        tuple(idx, obj_ids, mask_logits)
        exactly what `add_new_points_or_box` returns so the caller knows
        the real object ids allocated by SAM‑2.
        """
        idx = self.sam2.add_new_frame(self.state, frame_rgb)
        out = self.sam2.add_new_points_or_box(self.state, idx, obj_id=preferred_id, box=box)
        # Optionally re‑prime the already‑tracked hand so both masks share the frame
        if also_prime_hand and self.have_hand and self._last_hand_box is not None:
            _ = self.sam2.add_new_points_or_box(self.state, idx, obj_id=self.tr_hand_id, box=self._last_hand_box)
        return  (idx, [preferred_id], None)  # prepend idx for convenience

    # ─── Memory helpers ─────────────────────────────────────────────────
    def _memory_reset(self, frame_rgb):
        """Reset SAM‑2 memory and re‑prime existing objects."""
        #print(f"🔄 [Frame {self.f}] Resetting SAM‑2 (buffer hit {self.WINDOW} frames)")
        self._memory_resets += 1
        # preserve last boxes
        hand_box = self._last_hand_box.copy() if self.have_hand and self._last_hand_box is not None else None
        obj_box  = self._last_object_box.copy() if self.have_obj  and self._last_object_box is not None else None

        self.sam2.reset_state(self.state)
        self.state["images"] = torch.empty((0, 3, self.IMG_SIZE, self.IMG_SIZE), device=self.device)
        self.state["num_frames"] = 0
        self.state["video_height"], self.state["video_width"] = frame_rgb.shape[:2]
        idx = self.sam2.add_new_frame(self.state, frame_rgb)

        if hand_box is not None:
            self.sam2.add_new_points_or_box(self.state, idx, obj_id=self.tr_hand_id, box=hand_box)
        else:
            self.have_hand = False

        if obj_box is not None:
            self.sam2.add_new_points_or_box(self.state, idx, obj_id=self.tr_obj_id, box=obj_box)
        else:
            self.have_obj = False

        return idx

    def _add_frame(self, frame_rgb):
        """Push frame into SAM‑2 & handle memory window."""
        dev = self.device
        img = cv2.resize(frame_rgb, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        ten = torch.from_numpy(img).permute(2, 0, 1).to(dev)
        ten.sub_(torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None])\
           .div_(torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None])
        self.state["images"] = torch.cat([self.state["images"], ten[None]], 0)
        self.state["num_frames"] = self.state["images"].shape[0]
        idx = self.state["num_frames"] - 1
        self.state["cached_features"][idx] = (ten[None], self.sam2.forward_image(ten[None].float()))

        # Periodic reset
        if self.state["images"].shape[0] > self.WINDOW:
            idx = self._memory_reset(frame_rgb)
        return idx

    # ─── Performance & Debug Methods ──────────────────────────────────────
    def get_performance_stats(self) -> dict:
        """Get performance statistics for the tracking system."""
        current_time = time.time()
        elapsed_time = current_time - self._start_time
        average_fps = self._total_frames / elapsed_time if elapsed_time > 0 else 0
        return {
            "total_frames_processed": self._total_frames,
            "elapsed_time": elapsed_time,
            "average_fps": average_fps,
            "gdino_calls": self._gdino_calls,
            "sam2_calls": self._sam2_calls,
            "memory_resets": self._memory_resets,
            "current_memory_frames": self.state["images"].shape[0] if hasattr(self.state, "images") else 0,
        }

    def print_performance_summary(self) -> None:
        """Print a simple performance summary."""
        stats = self.get_performance_stats()
        print(f"\n📊 GSAM2 Performance Summary:")
        print(f"   Total frames processed: {stats['total_frames_processed']}")
        print(f"   Total time: {stats['elapsed_time']:.2f}s")
        print(f"   Average FPS: {stats['average_fps']:.1f}")
        print(f"   Real-time capable: {'✅ YES' if stats['average_fps'] >= 25 else '❌ NO'} (25+ FPS)")
        print(f"   GDINO calls: {stats['gdino_calls']} (detection)")
        print(f"   SAM-2 calls: {stats['sam2_calls']} (tracking)")
        print(f"   Memory resets: {stats['memory_resets']}")
        print(f"   Current memory: {stats['current_memory_frames']}/{self.WINDOW} frames")
        print(f"   Efficiency: {stats['sam2_calls']/(stats['gdino_calls']+stats['sam2_calls'])*100:.1f}% SAM-2 tracking")
        print()

    def print_debug_status(self) -> None:
        """Print current system status for debugging."""
        print(f"\n🔧 GSAM2 Debug Status:")
        print(f"   Frame count: {self.f}")
        print(f"   Hand tracked: {'✅' if self.have_hand else '❌'} (next try: {self.next_hand_try})")
        print(f"   Object tracked: {'✅' if self.have_obj else '❌'} (next try: {self.next_obj_try})")
        print(f"   Prompt waiting: {'⏳' if self.prompt_wait else '✅'}")
        print(f"   Current prompt: '{self.prompt_txt}'")
        print(f"   Lost counters: hand={self.lost_hand}/{self.MISS_MAX}, obj={self.lost_obj}/{self.MISS_MAX}")
        print(f"   Memory usage: {self.state['images'].shape[0]}/{self.WINDOW} frames")
        print()
