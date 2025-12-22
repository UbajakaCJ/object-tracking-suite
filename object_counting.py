# Object counting via reference line crossing using YOLO11
# Detects objects crossing a reference line and increments/decrements a counter

import os
import argparse
from collections import defaultdict
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Avoid OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_args():
    p = argparse.ArgumentParser(description="Count objects crossing a reference line")
    p.add_argument("--video", default="Resources/Videos/video8.mp4", help="Path to input video")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    p.add_argument("--orientation", choices=["horizontal", "vertical"], default="horizontal",
                   help="Reference line orientation")
    p.add_argument("--pos", type=float, default=0.8,
                   help="Relative position of the line (0.0-1.0) measured along orientation axis")
    p.add_argument("--space-side", choices=["above","below","left","right"], default="above",
                   help="Which side of the line is considered the 'space' to count as entering")
    p.add_argument("--out", default="output_videos/tracked_count.mp4", help="Output video path")
    p.add_argument("--debounce", type=int, default=5, help="Debounce frames to avoid double-counting")
    return p.parse_args()


class LineCounter:
    def __init__(self, orientation, pos_rel, frame_w, frame_h, space_side, debounce_frames=5):
        self.orientation = orientation
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.debounce = debounce_frames

        if orientation == "horizontal":
            self.line_y = int(pos_rel * frame_h)
            self.line_x = None
        else:
            self.line_x = int(pos_rel * frame_w)
            self.line_y = None

        # Track the last known side for each object id: 'above'/'below' or 'left'/'right'
        self.last_side = {}
        # Last frame index when this ID triggered a count (debounce)
        self.last_count_frame = {}

        # counters
        self.count = 0
        self.space_side = space_side

    def point_side(self, x, y):
        if self.orientation == "horizontal":
            return "below" if y >= self.line_y else "above"
        else:
            return "right" if x >= self.line_x else "left"

    def check_and_count(self, track_id, x, y, frame_idx):
        curr_side = self.point_side(x, y)
        prev_side = self.last_side.get(track_id)

        # If no previous side known, just set and return
        if prev_side is None:
            self.last_side[track_id] = curr_side
            return 0

        # If side changed, decide whether it is entering or exiting the 'space'
        if prev_side != curr_side:
            # debounce
            last = self.last_count_frame.get(track_id, -9999)
            if frame_idx - last < self.debounce:
                self.last_side[track_id] = curr_side
                return 0

            # Define crossing direction
            # Entering if moved onto the designated space_side
            entered = (curr_side == self.space_side)
            self.last_side[track_id] = curr_side
            self.last_count_frame[track_id] = frame_idx

            if entered:
                self.count += 1
                return +1
            else:
                # exiting the space
                self.count -= 1
                return -1

        return 0

    def draw_line_and_count(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.orientation == "horizontal":
            cv2.line(frame, (0, self.line_y), (self.frame_w, self.line_y), (0, 255, 255), 2)
        else:
            cv2.line(frame, (self.line_x, 0), (self.line_x, self.frame_h), (0, 255, 255), 2)

        text = f"Count: {self.count}"
        cv2.putText(frame, text, (10, 30), font, 1.0, (0, 255, 0), 2)


def main():
    args = parse_args()

    # Prepare output directory
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Get frame properties via VideoCapture to create writer
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (frame_w, frame_h))

    # Init model
    model = YOLO(args.model)

    # Detector/Counter
    counter = LineCounter(args.orientation, args.pos, frame_w, frame_h, args.space_side, debounce_frames=args.debounce)

    frame_idx = 0
    print("Starting processing...")
    print(f"Video: {frame_w}x{frame_h} @ {fps}fps, line: {args.orientation} pos={args.pos} ({args.space_side} is 'space')")

    # Use streaming mode so we maintain persistent track ids
    results = model.track(source=args.video, persist=True, stream=True)

    for r in results:
        frame_idx += 1
        # Annotated frame from YOLO (numpy BGR)
        annotated = r.plot()

        # Extract boxes and ids
        if hasattr(r, 'boxes') and getattr(r.boxes, 'id', None) is not None:
            # xywh (center x,y)
            try:
                boxes = r.boxes.xywh.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
            except Exception:
                # fallback: convert to list
                boxes = np.array(r.boxes.xywh).astype(float)
                ids = np.array(r.boxes.id).astype(int)

            for box, tid in zip(boxes, ids):
                cx, cy = float(box[0]), float(box[1])
                delta = counter.check_and_count(int(tid), cx, cy, frame_idx)
                if delta != 0:
                    sign = '+' if delta > 0 else '-'
                    print(f"Frame {frame_idx}: ID {tid} crossed line {sign} -> count={counter.count}")

                # Optionally draw the track centroid
                cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        # Draw reference line and counter
        counter.draw_line_and_count(annotated)

        out.write(annotated)
        cv2.imshow("Counting", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user")
            break

    out.release()
    cv2.destroyAllWindows()

    print("Done processing")
    print(f"Final count: {counter.count}")


if __name__ == "__main__":
    main()
