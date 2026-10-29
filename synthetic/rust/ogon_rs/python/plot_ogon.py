# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import ogon_rs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an orthogonal polygon in Rust and plot it in Python."
    )
    parser.add_argument("n", type=int, help="Target vertex count")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional image path; if omitted, show interactively.",
    )
    args = parser.parse_args()

    poly = ogon_rs.generate_ogon(args.n, args.seed)
    if poly.size == 0:
        raise RuntimeError("generator returned an empty polygon")

    fig, ax = plt.subplots()
    ax.plot(poly[:, 0], poly[:, 1], linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25)
    ax.set_title(f"Orthogonal polygon (n={args.n}, seed={args.seed})")

    if args.output is not None:
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
