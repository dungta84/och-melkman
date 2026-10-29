# och-melkman

Code for the paper **"Online construction of the orthogonal convex hull of an orthogonal polygon and an application in image analysis"** (manuscript ACAP-D-26-00396).

**Authors**

- Phong Thi Thu Huyen, Institute of Mathematics, Vietnam Academy of Science and Technology (VAST), Hanoi, Vietnam (ptthuyen@math.ac.vn)
- Tran Anh Dung (corresponding author), Faculty of Applied Science, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM, Vietnam (tadung.sdh231@hcmut.edu.vn)
- Phan Thanh An, Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM, Vietnam (thanhan@hcmut.edu.vn)

Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM. Released under the MIT License (see `LICENSE`).

## What is in this repository

1. **O-Melkman algorithm** (`src/och_melkman/omelkman.py`): an online, deque-based algorithm that computes the extreme vertices of the orthogonal convex hull (OCH) of an orthogonal polygon in linear time, inspired by Melkman's convex-hull algorithm.
2. **Aircraft-recognition pipeline** (`src/och_melkman/`, `scripts/`): YOLOv8x detection, automatic silhouette extraction, orientation normalisation, pixel-edge contour tracing, OCH by O-Melkman versus the convex hull (CH), eight shape features, and an RBF-SVM evaluation on MTARSI-fixed (10 aircraft classes, 100 chips per class).
3. **Synthetic benchmark** (`synthetic/`): Rust generator and implementation, the polygons of Table 2 and the benchmark script.
4. **Results** (`results/`): the feature table (`features10.csv`), the YOLO detection boxes, the evaluation outputs, and the tables and figures used in the paper.

## Install

```bash
pip install -r requirements.txt
python tests/test_omelkman.py      # checks O-Melkman output against an independent raster construction
```

## Using the algorithm

```python
import sys; sys.path.insert(0, "src")
from och_melkman import Point, ortho_hull

pts = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]      # vertices of an orthogonal polygon, in order
deque_of_extreme_vertices = ortho_hull([Point(x, y) for x, y in pts])
```

`ortho_hull` returns the deque of extreme vertices of the OCH; `adding_points` and `is_ortho_convex` in the same module complete it to the full orthogonal polygon and test orthogonal convexity.

## Reproducing the aircraft experiment

Obtain the data and the detector first (neither is redistributed here):

- MTARSI-fixed: https://huggingface.co/datasets/amistele/MTARSI-fixed (original MTARSI archive: https://doi.org/10.5281/zenodo.3464319). Put the images in `data/MTARSI-fixed/<class>/` or set `MTARSI_DIR`.
- YOLOv8x weights released by Ilmak et al. (2025). Put them in `weights/best.pt` or set `YOLO_WEIGHTS`. They are used as released, without fine-tuning.

Classes: B-1, B-52, C-130, C-135, B-2, C-17, C-5, P-3, KC-10, E-3. Run each step from the repository root:

```bash
python scripts/detect_aircraft.py B-52          # detection boxes -> results/detections/det_B-52.json
python scripts/run_pipeline.py B-52             # segmentation, polygon, OCH, CH, features -> work/p_B-52.jsonl
python scripts/merge_features.py                # -> results/features10.csv
python scripts/evaluate.py S                    # nested CV, SVM; REP=20 SUBS=all,unique,qc EXCL=B-2 (optional)
python scripts/make_report.py                   # LaTeX table rows and confusion matrices
python scripts/make_figure_onepage.py           # one-page pipeline figure
```

The precomputed `results/features10.csv` and `results/detections/` allow the evaluation step to be run without the images or the detector. Evaluation protocol: stratified grouped 5-fold cross-validation (images with identical content stay in the same fold), 20 repetitions, inner 3-fold search over `C in {1,10,100}` and `gamma in {scale, 0.3, 0.03}`.

## Synthetic experiments (random orthogonal polygons)

The folder `synthetic/` contains the generator, the polygons and the benchmark behind Table 2 of the paper.

- `synthetic/rust/ogon_rs/` is a Rust crate with Python bindings (PyO3). It generates random orthogonal polygons (`generate_ogon(n, seed)`; the polygons here use `seed=0`) and contains the O-Melkman implementation used for the timings (`orthogonal_hull`, with an instrumented version that counts predicate evaluations `a` and deque insertions/deletions `b`). The deque is updated in place.
- `synthetic/data/ogon_<n>.txt` are the polygons used in Table 2 (n = 10, 50, ..., 100000, 500000, 1000000), one vertex `x y` per line, closed polygon.
- `synthetic/results/results_server_run.txt` is the output of the benchmark that produced the table (LaTeX table).
- `synthetic/benchmark_python.py` runs the pure-Python reference on the same files and reproduces n, k and a (it skips 500000 and 1000000 vertices, which are slow in pure Python).

Build and run (needs Rust and Python >= 3.9):

```bash
cd synthetic/rust/ogon_rs
pip install maturin numpy matplotlib
maturin develop --release
cd ../../..
mkdir -p outputs figures
python synthetic/rust/ogon_rs/python/plot_ogon_hull.py --input_dir synthetic/data --output_dir figures   # writes outputs/results.txt
python synthetic/benchmark_python.py
```

The numbers n, k, a and b printed by the benchmark are determined by the polygons and were reproduced exactly (the Rust run and the Python reference agree on n, k and a for every file). The running time T depends on the machine; it is not reproduced exactly (the values in the paper come from a different run on different hardware).

## Reported results (998 processed chips, 10 classes)

| Features | Accuracy (mean ± s.d. over 20 repetitions) |
|---|---|
| CH-based | 0.600 ± 0.009 |
| OCH-based | 0.623 ± 0.006 |

On the nine classes other than B-2 (899 chips) the accuracies are 0.578 ± 0.009 (CH) and 0.593 ± 0.009 (OCH); the difference is small. Silhouettes are obtained automatically, so segmentation quality limits the accuracy. See the paper for the discussion.

## Citation

See `CITATION.cff`. Please cite the paper; the journal reference will be added after publication.

---

## Tóm tắt (tiếng Việt)

Mã nguồn đi kèm bài báo về thuật toán O-Melkman dựng trực tuyến bao lồi trực giao (OCH) của đa giác trực giao và ứng dụng nhận dạng máy bay trên ảnh vệ tinh (MTARSI-fixed, 10 lớp). Thư mục `src/och_melkman` chứa thuật toán và các bước xử lý ảnh, `scripts` chứa các lệnh chạy từng bước, `results` chứa bảng đặc trưng, hộp phát hiện YOLO, kết quả đánh giá, bảng và hình dùng trong bài. Dữ liệu ảnh và trọng số YOLOv8x (Ilmak và cs., 2025) không được phát hành lại trong kho này; xem hướng dẫn ở trên. Bản quyền thuộc Viện Khoa học Toán và Tính toán (IMACS), Trường Đại học Bách khoa TP.HCM, ĐHQG-HCM; giấy phép MIT.
