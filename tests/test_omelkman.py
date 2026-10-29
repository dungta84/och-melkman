# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
"""Check the O-Melkman output against an independent raster construction of the OCH (row/column gap filling)."""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import cv2
from och_melkman import geom

def random_region(n=40, seed=0):
    rng = random.Random(seed)
    m = np.zeros((n, n), np.uint8)
    cx, cy = n // 2, n // 2
    for _ in range(80):
        x, y = cx + rng.randint(-14, 14), cy + rng.randint(-14, 14)
        m[y - 1:y + 2, x - 4:x + 5] = 1; m[y - 4:y + 5, x - 1:x + 2] = 1
    num, lab = cv2.connectedComponents(m, connectivity=4)
    sizes = [(lab == i).sum() for i in range(1, num)]
    return (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)

def test_area_matches_independent_closure():
    for seed in range(20):
        r = geom.tidy(random_region(seed=seed))
        P = geom.outer_polygon(r)
        ext, poly, isoc = geom.coch_polygon(P)
        assert isoc
        assert abs(abs(geom.shoelace(poly)) - geom.coch_closure(r).sum()) < 1e-9

if __name__ == '__main__':
    test_area_matches_independent_closure(); print('ok')
