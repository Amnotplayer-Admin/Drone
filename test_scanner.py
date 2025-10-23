import os
import tempfile

import numpy as np
import cv2

from scanner import detect_crack_mask, measure_widths


def make_synthetic_crack_image(width=200, height=100):
    img = 255 * np.ones((height, width), dtype=np.uint8)
    # draw a thin curved crack
    pts = np.array([[10, 10], [50, 30], [100, 50], [150, 70], [190, 90]])
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), 0, 2)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr


def test_synthetic_crack_measurement():
    img = make_synthetic_crack_image()
    mask = detect_crack_mask(img)
    measurements = measure_widths(mask, meters_per_pixel=None)
    assert len(measurements) > 0
    # widths should be in pixels and >0
    for (_, _), w in measurements[:10]:
        assert w > 0
