# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
"""Cross-check: run the pure-Python reference (src/och_melkman/omelkman.py) on the polygons in data/
and print n, k and the number a of is_in_out predicate evaluations during the hull construction.
n, k and a must coincide with the Rust implementation (see results/results_server_run.txt)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
from och_melkman import omelkman as om

calls = 0
_orig = om.is_in_out
def counted(a, b, c):
    global calls
    calls += 1
    return _orig(a, b, c)
om.is_in_out = counted

print('file & n & k & a & T(ms, python)')
for n in (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000):   # 500000 and 1000000 omitted: slow in pure Python
    pts = []
    with open(os.path.join(HERE, 'data', 'ogon_%d.txt' % n)) as f:
        for line in f:
            if line.strip():
                x, y = line.split(); pts.append(om.Point(int(x), int(y)))
    calls = 0
    t0 = time.time(); hull = om.ortho_hull(pts); t1 = time.time()
    a = calls
    print('ogon_%d.txt & %d & %d & %d & %.2f' % (n, len(pts) - 1, len(hull) - 1, a, (t1 - t0) * 1000))
