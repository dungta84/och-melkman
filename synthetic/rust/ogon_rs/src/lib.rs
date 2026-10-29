// Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
// Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
//! Orthogonal grid polygon generator + orthogonal hull with PyO3 bindings.
//!
//! Exposes:
//! - `ogon_rs.generate_ogon(n, seed=0)`
//! - `ogon_rs.orthogonal_hull(points)`
//! - `ogon_rs.orthogonal_hull_path(points)`
//! - `ogon_rs.orthogonal_convex_hull_instrumented(points)`
//! - `ogon_rs.is_orthogonally_convex(points)`

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod geometry;
pub mod ogon;
pub mod orthogonal_hull;

pub use geometry::Point;
pub use ogon::generate_ogon;
pub use orthogonal_hull::{
    add_corner_points,
    is_orthogonally_convex,
    orthogonal_hull,
    orthogonal_hull_instrumented,
    HullStats,
};

#[inline]
fn tuples_to_points(points: Vec<(i32, i32)>) -> Vec<Point> {
    points
        .into_iter()
        .map(|(x, y)| Point::new(x, y))
        .collect()
}

#[inline]
fn points_to_tuples(points: Vec<Point>) -> Vec<(i32, i32)> {
    points.into_iter().map(|p| (p.x, p.y)).collect()
}

#[pyfunction(name = "generate_ogon")]
#[pyo3(signature = (n, seed=0))]
fn generate_ogon_py(n: usize, seed: u64) -> Vec<(i32, i32)> {
    points_to_tuples(generate_ogon(n, seed))
}

#[pyfunction(name = "orthogonal_hull")]
fn orthogonal_hull_py(points: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    points_to_tuples(orthogonal_hull(&tuples_to_points(points)))
}

#[pyfunction(name = "orthogonal_hull_path")]
fn orthogonal_hull_path_py(points: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    let hull = orthogonal_hull(&tuples_to_points(points));
    points_to_tuples(add_corner_points(&hull))
}

#[pyfunction(name = "orthogonal_convex_hull_instrumented")]
fn orthogonal_convex_hull_instrumented_py(
    points: Vec<(i32, i32)>,
) -> (Vec<(i32, i32)>, usize, usize) {
    let (hull, stats) = orthogonal_hull_instrumented(&tuples_to_points(points));
    (
        points_to_tuples(hull),
        stats.check_vertex,
        stats.add_remove_vertex,
    )
}

#[pyfunction(name = "is_orthogonally_convex")]
fn is_orthogonally_convex_py(points: Vec<(i32, i32)>) -> bool {
    is_orthogonally_convex(&tuples_to_points(points))
}

#[pymodule]
fn ogon_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_ogon_py, m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_hull_py, m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_hull_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_convex_hull_instrumented_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_orthogonally_convex_py, m)?)?;
    Ok(())
}
