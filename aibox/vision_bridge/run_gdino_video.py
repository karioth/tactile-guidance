import argparse
from pathlib import Path
import time

import cv2

# Local import – vision_bridge is now a proper Python package
from .gsam2_wrapper import GSAM2Wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-vocabulary detection (Grounding-DINO) on a video and save the overlay result.",
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video (e.g. testingvid.mp4)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="coffee cup",
        help="Text prompt describing the target object (e.g. 'red bottle').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the annotated video.  Defaults to '<video>_dino.mp4' in the same directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use (cuda, cpu, cuda:1, …).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else video_path.parent / f"{video_path.stem}_dino"
    ).with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialise model once.
    detector = GSAM2Wrapper(device=args.device)
    detector.set_prompt(None, args.prompt)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    tic = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.track(frame)
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det[:4]]
            conf = float(det[6])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        out.write(frame)
        frame_idx += 1
        print("frame_idx: ", frame_idx )

    duration = time.time() - tic
    if duration > 0:
        print(f"Processed {frame_idx} frames in {duration:.1f}s ({frame_idx / duration:.2f} FPS)")

    cap.release()
    out.release()
    print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    main() 