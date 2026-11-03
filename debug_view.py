#!/usr/bin/env python3
"""
See exactly what the filter sees, stage by stage, live from your webcam.

In PyCharm: click the green Run arrow (defaults to camera 0).
From a terminal:

    python debug_view.py                  # live webcam, camera 0
    python debug_view.py --camera 1       # a different webcam
    python debug_view.py --image photo.jpg

One window, four panels, updated live:

    top-left      original frame, with the detector's own overlay
                  (yellow = disc it's tracking, green/red = read/abstain box)
    top-right     gray after CLAHE (step 1 of prepare()) — this is the
                  image every threshold in the pipeline actually works on,
                  not the raw camera frame
    bottom-left   "dark" mask — every pixel darker than the frame's Otsu
                  cutoff (white = dark). The disc should show up as one
                  solid white blob here; stray white elsewhere is clutter
                  competing for the "nearest to center" pick
    bottom-right  bar mask inside the locked disc, after the contrast
                  stretch — this is what actually gets counted into bar
                  shapes. Each real bar should be a crisp separate white
                  blob; bars touching or blurring into one blob here is
                  exactly what trips a "bars merged" abstain

Press q / Esc to quit, s to save the current 4-panel view to ./captures/.
"""
import argparse
import os
import time

import cv2
import numpy as np

from bar_counter import BarCounter, prepare

CAPTURE_DIR = "captures"
PANEL = (480, 360)   # size each of the 4 panels is resized to


def to_bgr(gray_or_mask, scale255=False):
    """Make a single-channel image showable as a 3-channel BGR panel."""
    img = gray_or_mask
    if scale255 and img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if scale255 and img.max() <= 1:
        img = img * 255
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def label_panel(img, text):
    img = cv2.resize(img, PANEL)
    cv2.rectangle(img, (0, 0), (PANEL[0], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def build_view(frame, counter, prev_center):
    gray, value = prepare(frame)
    dark = counter.dark(gray) * 255

    disc = counter.nearest_disc(gray, value, prev_center)
    overlay = frame.copy()
    bars_panel = np.zeros_like(gray)
    reading_text = "no disc"
    new_center = None

    if disc is not None:
        cx, cy, r = int(disc.cx), int(disc.cy), int(disc.r)
        new_center = (disc.cx, disc.cy, disc.r)
        cv2.circle(overlay, (cx, cy), int(r), (0, 255, 255), 2)
        cv2.drawMarker(overlay, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

        bars, searched = counter.bar_pixels(gray, cx, cy, r)
        # Crop the bar mask to the disc's own box so it's not just a tiny
        # white speck in a huge black panel.
        pad = int(r * 1.1)
        y0, y1 = max(0, cy - pad), min(gray.shape[0], cy + pad)
        x0, x1 = max(0, cx - pad), min(gray.shape[1], cx + pad)
        bars_panel = (bars[y0:y1, x0:x1] * 255) if bars[y0:y1, x0:x1].size else bars_panel

        reading = counter.read(frame, prev_center=prev_center)
        new_center = reading.center or new_center
        if reading.bbox is not None:
            h, w = frame.shape[:2]
            bx1, by1, bx2, by2 = reading.bbox
            p1, p2 = (int(bx1 * w), int(by1 * h)), (int(bx2 * w), int(by2 * h))
            color = (0, 220, 0) if reading.label else (0, 0, 220)
            cv2.rectangle(overlay, p1, p2, color, 2)
        reading_text = f"{reading.label or 'unsure'} {reading.confidence:.2f}  {reading.note}"

    cv2.putText(overlay, reading_text, (8, overlay.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(overlay, reading_text, (8, overlay.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    top = np.hstack([
        label_panel(overlay, "camera + detection"),
        label_panel(to_bgr(gray), "gray after CLAHE"),
    ])
    bottom = np.hstack([
        label_panel(to_bgr(dark, scale255=True), "dark mask (step 1)"),
        label_panel(to_bgr(bars_panel, scale255=True), "bar mask (steps 2-4)"),
    ])
    return np.vstack([top, bottom]), new_center


def run(cap_or_image, is_image):
    counter = BarCounter()
    prev_center = None
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    print("q/Esc to quit, s to save the current view.")

    while True:
        if is_image:
            frame = cap_or_image
        else:
            ok, frame = cap_or_image.read()
            if not ok:
                print("Camera read failed.")
                break

        view, prev_center = build_view(frame, counter, prev_center)
        cv2.imshow("what the filter sees (q to quit)", view)

        key = cv2.waitKey(1 if not is_image else 0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            fname = os.path.join(CAPTURE_DIR, f"debug_{int(time.time())}.png")
            cv2.imwrite(fname, view)
            print(f"saved {fname}")
        if is_image:
            break

    if not is_image:
        cap_or_image.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--image", type=str, default=None, help="path to a single still image")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read {args.image}")
            return
        run(frame, is_image=True)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Could not open camera {args.camera}.")
            return
        run(cap, is_image=False)


if __name__ == "__main__":
    main()