# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import ogon_rs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an orthogonal polygon in Rust and plot it in Python."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional image path; if omitted, show interactively.",
    )
    args = parser.parse_args()

    seed = 0

    for n in (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500_000, 1_000_000):
        poly = ogon_rs.generate_ogon(n, seed)
        if not poly:
            raise RuntimeError("generator returned an empty polygon")

        save_path = args.output_dir/f"ogon_{n}.txt"

        with open(save_path, "a") as file:
            for x, y in poly:
                file.write(f"{x} {y}\n") 
        
if __name__ == "__main__":
    main()
