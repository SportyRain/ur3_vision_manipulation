"""Briefly probe Windows camera indices 0, 1, and 2 via DirectShow."""

from __future__ import annotations

import cv2
import numpy as np


def probe(index: int) -> None:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        opened = capture.isOpened()
        read, frame = capture.read() if opened else (False, None)
        valid_frame = bool(read and frame is not None and frame.size > 0)
        print(f"index {index}:")
        print(f"  opened={str(opened).lower()}")
        print(f"  frame_read={str(valid_frame).lower()}")
        if valid_frame:
            height, width = frame.shape[:2]
            channels = 1 if frame.ndim == 2 else frame.shape[2]
            print(f"  resolution={width}x{height}")
            print(f"  channels={channels}")
            print(f"  dtype={frame.dtype}")
            print(f"  mean={float(np.mean(frame)):.6f}")
            print(f"  std={float(np.std(frame)):.6f}")
        else:
            print("  resolution=UNKNOWN")
            print("  channels=UNKNOWN")
            print("  dtype=UNKNOWN")
            print("  mean=UNKNOWN")
            print("  std=UNKNOWN")
        print("  device_name=UNKNOWN")
    finally:
        capture.release()


if __name__ == "__main__":
    for camera_index in (0, 1, 2):
        probe(camera_index)
