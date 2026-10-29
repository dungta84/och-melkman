# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import ogon_rs
import time 

Point = tuple[int, int]

def read_polygon(path: Path) -> list[Point]:
    points: list[Point] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected exactly two integers per line, got: {raw_line.rstrip()!r}"
                )

            try:
                x = int(parts[0])
                y = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_no}: expected integer coordinates, got: {raw_line.rstrip()!r}"
                ) from exc

            points.append((x, y))

    if len(points) < 2:
        raise ValueError(f"{path}: expected at least two points")

    if points[0] != points[-1]:
        points.append(points[0])

    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read an orthogonal polygon from a text file, compute its orthogonal hull in Rust, "
            "and plot both in Python."
        )
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Coordinates path.",
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional image path; if omitted, show interactively.",
    )
    
    args = parser.parse_args()
    results = []

    for n in (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500_000, 1_000_000):
        input_file = args.input_dir/f"ogon_{n}.txt" 
        poly = read_polygon(input_file)
        hull, check_vertex, add_remove_vertex = ogon_rs.orthogonal_convex_hull_instrumented(poly)
        hull_path = ogon_rs.orthogonal_hull_path(poly)

        start_time = time.time()
        ogon_rs.orthogonal_hull(poly)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000 # in milliseconds
        orthogonally_convex_status = ogon_rs.is_orthogonally_convex(hull) 
        n = len(poly) - 1
        k = len(hull) - 1

        results.append(
            (input_file.name, orthogonally_convex_status, n, k, check_vertex, add_remove_vertex, elapsed_time)
        )
        
        fig, ax = plt.subplots()
        px, py = zip(*poly)
        ax.plot(px, py, linewidth=0.8, color="gray", label="Contour")

        if hull_path:
            hx, hy = zip(*hull_path)
            ax.plot(hx, hy, linewidth=1.0, color="tab:blue", label="Orthogonal convex hull")
    
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.25)
        ax.set_title(
            f"Orthogonal polygon + orthogonal hull (n={n}, k={k})"
        )
        ax.legend()
        fig.savefig(args.output_dir/f"orthogonal_convex_hull_{n}", dpi=180, bbox_inches="tight")
        plt.close()

    latex_table = r"""
\begin{table}[H]
\centering
\begin{tabular}{ l  l  l l l l l}
\hline
Size & isOrthoConvex & n & k & a & b & {T} \\
\hline
"""
    # Fill the table with results
    for result in results:
        latex_table += f"{result[0]} & {result[1]} & {result[2]} & {result[3]} & {result[4]} & {result[5]} & {result[6]:.2f} \\\\\n"

    latex_table += r"""
\hline
\end{tabular}
\caption{Summary of computational data. Number of vertices ($n$), the orthogonal convexity status (isOrthoConvex), number of extreme vertices ($k$), size, number of vertices to check ($a$), number of vertices added/removed ($b$), and corresponding computation time ($T$ in milliseconds)}
\label{table1}
\end{table}
"""
    
    with open(Path("outputs/results.txt"), 'w') as file: 
        file.write(latex_table) 

if __name__ == "__main__":
    main()
    