import argparse
import csv
import math
import os
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize


def detect_aruco_scale(image: np.ndarray) -> Optional[float]:
    """Detect an ArUco marker and return meters per pixel scale if found.
    Assumes a 5x5 ArUco marker of known size (default 0.05 m).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco, parameters=params)
    if ids is None or len(corners) == 0:
        return None
    # take first marker
    c = corners[0][0]
    perimeter = 0.0
    for i in range(4):
        p1 = c[i]
        p2 = c[(i + 1) % 4]
        perimeter += math.hypot(*(p1 - p2))
    # marker is a square; side = perimeter/4
    side_px = perimeter / 4.0
    # assume marker real size in meters
    marker_size_m = 0.05
    meters_per_pixel = marker_size_m / side_px
    return meters_per_pixel


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Contrast limited adaptive histogram equalization for better cracks
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def detect_crack_mask(image: np.ndarray) -> np.ndarray:
    pre = preprocess(image)
    # Use adaptive threshold to catch thin dark cracks
    th = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    # Morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    # If adaptive threshold didn't find anything, try Canny edges as a fallback
    if np.count_nonzero(opened) < 50:
        edges = cv2.Canny(pre, 30, 100)
        mask_candidate = cv2.dilate(edges, kernel, iterations=1)
    else:
        mask_candidate = opened

    # remove small components (lower min size to keep thin cracks)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask_candidate, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    mask = np.zeros(output.shape, dtype=np.uint8)
    min_size = 5
    for i, sz in enumerate(sizes):
        if sz >= min_size:
            mask[output == i + 1] = 255
    return mask


def measure_widths(mask: np.ndarray, meters_per_pixel: Optional[float] = None) -> List[Tuple[Tuple[int, int], float]]:
    # skeletonize
    bw = mask > 0
    ske = skeletonize(bw).astype(np.uint8)
    # distance transform on the crack mask: distance from crack pixel to background
    dist = cv2.distanceTransform((bw.astype(np.uint8)) * 255, cv2.DIST_L2, 5)
    # For each skeleton pixel, the width is 2*dist at that point
    ys, xs = np.nonzero(ske)
    measurements: List[Tuple[Tuple[int, int], float]] = []
    for x, y in zip(xs, ys):
        d = dist[y, x]
        width_px = d * 2
        if width_px <= 0:
            continue
        width = width_px * meters_per_pixel if meters_per_pixel is not None else width_px
        measurements.append(((x, y), float(width)))
    return measurements


def annotate_and_save(image: np.ndarray, measurements: List[Tuple[Tuple[int, int], float]], out_image_path: str, meters_per_pixel: Optional[float]):
    out = image.copy()
    for (x, y), w in measurements:
        label = f"{w:.3f}m" if meters_per_pixel else f"{w:.1f}px"
        cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(out, label, (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.imwrite(out_image_path, out)


def write_csv(measurements: List[Tuple[Tuple[int, int], float]], out_csv_path: str):
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "width"])
        for (x, y), width in measurements:
            w.writerow([x, y, width])


def main():
    parser = argparse.ArgumentParser(description="Detect cracks and measure widths")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="out.png", help="Annotated output image")
    parser.add_argument("--csv", default="measurements.csv", help="CSV output")
    parser.add_argument("--aruco-scale", action="store_true", help="Attempt to detect an ArUco marker to set meters-per-pixel scale")
    args = parser.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read image", file=sys.stderr)
        sys.exit(2)
    meters_per_pixel = None
    if args.aruco_scale:
        meters_per_pixel = detect_aruco_scale(img)
        if meters_per_pixel is None:
            print("ArUco marker not found; continuing in pixels")
    mask = detect_crack_mask(img)
    measurements = measure_widths(mask, meters_per_pixel)
    write_csv(measurements, args.csv)
    annotate_and_save(img, measurements, args.out, meters_per_pixel)
    print(f"Saved {args.out} and {args.csv} with {len(measurements)} points")


if __name__ == "__main__":
    main()
