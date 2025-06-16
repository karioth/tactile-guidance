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
from contextlib import contextmanager

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

# ─── Performance Profiler ──────────────────────────────────────────────────────
class PerformanceProfiler:
    """Clean, toggleable performance profiler for GSAM2 pipeline functions."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timers = {}
        self.call_counts = {}
        self.start_time = time.time()
        
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing function calls."""
        if not self.enabled:
            yield
            return
            
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.timers:
                self.timers[name] = []
                self.call_counts[name] = 0
            self.timers[name].append(elapsed)
            self.call_counts[name] += 1
    
    def get_fps_breakdown(self, total_frames: int) -> dict:
        """Get FPS breakdown for each tracked function."""
        if not self.enabled or not self.timers:
            return {}
            
        breakdown = {}
        total_time = time.time() - self.start_time
        
        # Calculate tracking-only performance (exclude sparse detection calls)
        tracking_functions = ['sam2_inference', 'sam2_add_frame', 'mask_to_bbox', 'bgr_to_rgb', 'sam2_memory_reset']
        tracking_time = 0
        tracking_calls = 0
        
        for name, times in self.timers.items():
            avg_time = np.mean(times)
            total_func_time = sum(times)
            calls = self.call_counts[name]
            
            breakdown[name] = {
                'avg_time_ms': avg_time * 1000,
                'total_time_s': total_func_time,
                'calls': calls,
                'fps_if_every_frame': 1.0 / avg_time if avg_time > 0 else float('inf'),
                'actual_call_rate': calls / total_frames if total_frames > 0 else 0,
                'time_percentage': (total_func_time / total_time) * 100 if total_time > 0 else 0
            }
            
            # Accumulate tracking-only time (functions that run frequently)
            if name in tracking_functions:
                tracking_time += total_func_time
                tracking_calls += calls
        
        # Calculate tracking FPS (excluding sparse detection overhead)
        tracking_fps = tracking_calls / tracking_time if tracking_time > 0 else 0
        
        # Also calculate per-frame tracking time
        per_frame_tracking_time = tracking_time / total_frames if total_frames > 0 else 0
        tracking_only_fps = 1.0 / per_frame_tracking_time if per_frame_tracking_time > 0 else 0
        
        breakdown['overall'] = {
            'total_frames': total_frames,
            'total_time_s': total_time,
            'overall_fps': total_frames / total_time if total_time > 0 else 0,
            'tracking_fps': tracking_only_fps,
            'tracking_time_per_frame_ms': per_frame_tracking_time * 1000,
            'detection_overhead_s': total_time - tracking_time
        }
        
        return breakdown
    
    def print_fps_report(self, total_frames: int):
        """Print a clean, informative FPS breakdown report."""
        if not self.enabled:
            print("📊 Performance profiling disabled")
            return
            
        breakdown = self.get_fps_breakdown(total_frames)
        if not breakdown:
            print("📊 No performance data collected")
            return
        
        print(f"\n📊 GSAM2 Performance Breakdown ({total_frames} frames)")
        print("=" * 70)
        
        overall = breakdown.pop('overall')
        print(f"Overall FPS: {overall['overall_fps']:.1f} ({overall['total_time_s']:.1f}s total)")
        print(f"🎯 Tracking FPS: {overall['tracking_fps']:.1f} (excluding detection overhead)")
        print(f"   Tracking time per frame: {overall['tracking_time_per_frame_ms']:.1f}ms")
        print(f"   Detection overhead: {overall['detection_overhead_s']:.1f}s total")
        print()
        
        # Separate tracking vs detection functions
        tracking_funcs = []
        detection_funcs = []
        other_funcs = []
        
        for name, stats in breakdown.items():
            if name in ['sam2_inference', 'sam2_add_frame', 'mask_to_bbox', 'bgr_to_rgb', 'sam2_memory_reset']:
                tracking_funcs.append((name, stats))
            elif 'gdino' in name or 'prime' in name:
                detection_funcs.append((name, stats))
            else:
                other_funcs.append((name, stats))
        
        # Sort each category by time percentage
        tracking_funcs.sort(key=lambda x: x[1]['time_percentage'], reverse=True)
        detection_funcs.sort(key=lambda x: x[1]['time_percentage'], reverse=True)
        other_funcs.sort(key=lambda x: x[1]['time_percentage'], reverse=True)
        
        print("🏃 TRACKING FUNCTIONS (run frequently)")
        print(f"{'Function':<25} {'Avg(ms)':<8} {'Calls':<6} {'Rate':<6} {'FPS*':<8} {'Time%':<6}")
        print("-" * 70)
        
        for name, stats in tracking_funcs:
            rate = f"{stats['actual_call_rate']:.2f}" if stats['actual_call_rate'] < 1 else f"{stats['actual_call_rate']:.1f}"
            fps_star = f"{stats['fps_if_every_frame']:.1f}" if stats['fps_if_every_frame'] < 1000 else "999+"
            print(f"{name:<25} {stats['avg_time_ms']:<8.1f} {stats['calls']:<6} {rate:<6} {fps_star:<8} {stats['time_percentage']:<6.1f}")
        
        if detection_funcs:
            print(f"\n🔍 DETECTION FUNCTIONS (run sparsely)")
            print(f"{'Function':<25} {'Avg(ms)':<8} {'Calls':<6} {'Rate':<6} {'FPS*':<8} {'Time%':<6}")
            print("-" * 70)
            
            for name, stats in detection_funcs:
                rate = f"{stats['actual_call_rate']:.3f}" if stats['actual_call_rate'] < 0.1 else f"{stats['actual_call_rate']:.2f}"
                fps_star = f"{stats['fps_if_every_frame']:.1f}" if stats['fps_if_every_frame'] < 1000 else "999+"
                print(f"{name:<25} {stats['avg_time_ms']:<8.1f} {stats['calls']:<6} {rate:<6} {fps_star:<8} {stats['time_percentage']:<6.1f}")
        
        if other_funcs:
            print(f"\n⚙️  OTHER FUNCTIONS")
            print(f"{'Function':<25} {'Avg(ms)':<8} {'Calls':<6} {'Rate':<6} {'FPS*':<8} {'Time%':<6}")
            print("-" * 70)
            
            for name, stats in other_funcs:
                rate = f"{stats['actual_call_rate']:.3f}" if stats['actual_call_rate'] < 0.1 else f"{stats['actual_call_rate']:.2f}"
                fps_star = f"{stats['fps_if_every_frame']:.1f}" if stats['fps_if_every_frame'] < 1000 else "999+"
                print(f"{name:<25} {stats['avg_time_ms']:<8.1f} {stats['calls']:<6} {rate:<6} {fps_star:<8} {stats['time_percentage']:<6.1f}")
        
        print()
        print("🎯 Tracking FPS = 1 / (tracking_time_per_frame)")
        print("* FPS if this function ran every frame")
        print("Rate: calls per frame (1.0 = every frame, 0.01 = every 100 frames)")
        print()

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

    _CONF_PATH    = ROOT / "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    _CKPT_PATH    = ROOT / "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    # _SAM2_CFG     = "configs/efficienttam/efficienttam_ti.yaml"
    # _SAM2_WEIGHTS = ROOT / "Grounded-SAM-2/checkpoints/efficienttam_ti.pt"
    _SAM2_CFG     = "configs/efficienttam/efficienttam_ti_512x512.yaml"
    _SAM2_WEIGHTS = ROOT / "Grounded-SAM-2/checkpoints/efficienttam_ti_512x512.pt"
    # _SAM2_CFG     = "configs/sam2.1/sam2.1_hiera_t.yaml"
    # _SAM2_WEIGHTS = ROOT / "Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"

    # Tunables
    WINDOW   = 100   # SAM‑2 memory window (frames)
    MISS_MAX = 30    # lost‑mask threshold (frames)
    RETRY    = 15    # DINO retry interval after a miss (frames)
    IMG_SIZE = 512  # SAM‑2 input resolution

    def __init__(self, device: str | torch.device | None = None,
                 box_threshold: float = .35, text_threshold: float = .25,
                 default_prompt: str = "coffee cup", handedness: str = "right",
                 enable_profiling: bool = True, window: int = None, 
                 miss_max: int = None, retry: int = None):
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.box_thr, self.txt_thr = box_threshold, text_threshold
        self.prompt_txt = default_prompt
        self.handedness = handedness.lower()

        # Override class constants with provided parameters
        if window is not None:
            self.WINDOW = window
        if miss_max is not None:
            self.MISS_MAX = miss_max
        if retry is not None:
            self.RETRY = retry

        # ── Performance profiler ──────────────────────────────────────────
        self.profiler = PerformanceProfiler(enabled=enable_profiling)

        # ── Tracking-only performance measurement ──────────────────────────
        self._tracking_start_time = None  # When pure tracking phase begins
        self._tracking_frame_count = 0    # Frames processed during tracking phase

        # ── Load models ────────────────────────────────────────────────────
        print("⏳ Loading Grounding DINO model…", end=" ")
        self.dino = GDINOModel(str(self._CONF_PATH), str(self._CKPT_PATH), device=str(self.device))
        print("✅ done")
        print("⏳ Loading tracking model…", end=" ")
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

        # ── Performance tracking (legacy - kept for compatibility) ──────
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
        with self.profiler.timer("track_total"):
            cur = self.f
            self._total_frames += 1

            # RGB conversion once here
            with self.profiler.timer("bgr_to_rgb"):
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
            
            # Start tracking-only timer when both objects are being tracked
            if self._tracking_start_time is None and self.have_hand and self.have_obj:
                self._tracking_start_time = time.time()
                print(f"🎯 [Frame {cur}] Starting pure tracking phase measurement")
            
            # Count frames during tracking phase
            if self._tracking_start_time is not None and self.have_hand and self.have_obj:
                self._tracking_frame_count += 1
            
            with self.profiler.timer("sam2_inference"):
                idx, ids, masks = self.sam2.infer_single_frame(self.state, idx)
            self._sam2_calls += 1

            # -----------------------------------------------------------------
            # 6️⃣  Convert masks → bracelet tuples
            # -----------------------------------------------------------------
            with self.profiler.timer("mask_to_bbox"):
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
        with self.profiler.timer("gdino_hand_detection"):
            dets, lbls = self._dino(frame_rgb, f"my {self.handedness} hand")
        self._gdino_calls += 1
        box = _pick_best(dets, lbls, "hand")
        if box is None:
            print("fail")
            return None

        preferred_id = 2                     # <<< fixed id for hand
        with self.profiler.timer("sam2_hand_prime"):
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
        with self.profiler.timer("gdino_object_detection"):
            dets, lbls = self._dino(frame_rgb, self.prompt_txt)
        self._gdino_calls += 1
        box = _pick_best(dets, lbls)
        if box is None:
            print("fail")
            return None

        preferred_id = 1                     # <<< fixed id for object
        with self.profiler.timer("sam2_object_prime"):
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
        with self.profiler.timer("sam2_memory_reset"):
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
        # NEW METHOD: Use model's native add_new_frame
        with self.profiler.timer("sam2_add_frame"):
            idx = self.sam2.add_new_frame(self.state, frame_rgb)
            
            # Periodic reset (same logic as before)
            if self.state["images"].shape[0] > self.WINDOW:
                idx = self._memory_reset(frame_rgb)
            return idx

    # ─── Performance & Debug Methods ──────────────────────────────────────
    def get_tracking_only_fps(self) -> float:
        """Get FPS for pure tracking phase (after detection is complete)."""
        if self._tracking_start_time is None or self._tracking_frame_count == 0:
            return 0.0
        
        tracking_elapsed = time.time() - self._tracking_start_time
        return self._tracking_frame_count / tracking_elapsed if tracking_elapsed > 0 else 0.0

    def print_detailed_performance(self) -> None:
        """Print detailed FPS breakdown for each major function."""
        if not self.profiler.enabled:
            print("📊 Performance profiling disabled")
            return
            
        # Get tracking-only FPS
        tracking_only_fps = self.get_tracking_only_fps()
        
        print(f"\n📊 GSAM2 Performance Breakdown ({self._total_frames} frames)")
        print("=" * 70)
        
        # Print tracking-only performance first
        if tracking_only_fps > 0:
            print(f"🚀 Pure Tracking FPS: {tracking_only_fps:.1f} ({self._tracking_frame_count} frames after detection)")
            print(f"   Detection phase: {self._total_frames - self._tracking_frame_count} frames")
            print()
        
        # Then print the regular detailed breakdown
        self.profiler.print_fps_report(self._total_frames)

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

    def enable_profiling(self):
        """Enable performance profiling."""
        self.profiler.enabled = True
        
    def disable_profiling(self):
        """Disable performance profiling."""
        self.profiler.enabled = False
