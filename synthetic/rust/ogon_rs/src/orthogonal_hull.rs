// Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
// Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
//! Orthogonal convex hull core logic ported from `orthogonal_convex_hull.py`.
//!
//! The Python reference uses a deque-based incremental algorithm over the input
//! contour order. This port keeps the same logical tests (`case_type`,
//! `is_in_out`, `build_right`, `build_left`, `ortho_hull`), but performs the
//! updates in place to avoid the Python script's repeated deque copying.
//!
//! This version also supports optional instrumentation for:
//! - `check_vertex`: number of `is_in_out` predicate evaluations during hull construction
//! - `add_remove_vertex`: number of deque vertex mutations (push/pop) during hull construction

use std::collections::VecDeque;

use crate::geometry::Point;

#[repr(i8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum CaseType {
    Same = 0,
    Left = -1,
    Down = -2,
    Right = -3,
    Up = -4,
    UpLeft = 1,
    DownLeft = 2,
    DownRight = 3,
    UpRight = 4,
}

/// Optional instrumentation counters for orthogonal hull construction.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct HullStats {
    pub check_vertex: usize,
    pub add_remove_vertex: usize,
}

trait StatsSink {
    fn record_check_vertex(&mut self);
    fn record_add_remove_vertex(&mut self, amount: usize);
}

#[derive(Copy, Clone, Debug, Default)]
struct NoStats;

impl StatsSink for NoStats {
    #[inline(always)]
    fn record_check_vertex(&mut self) {}

    #[inline(always)]
    fn record_add_remove_vertex(&mut self, _amount: usize) {}
}

impl StatsSink for HullStats {
    #[inline(always)]
    fn record_check_vertex(&mut self) {
        self.check_vertex += 1;
    }

    #[inline(always)]
    fn record_add_remove_vertex(&mut self, amount: usize) {
        self.add_remove_vertex += amount;
    }
}

#[inline(always)]
fn case_type(a: Point, b: Point) -> CaseType {
    if a.x == b.x && a.y == b.y {
        CaseType::Same
    } else if a.x > b.x && a.y == b.y {
        CaseType::Left
    } else if a.x == b.x && a.y > b.y {
        CaseType::Down
    } else if a.x < b.x && a.y == b.y {
        CaseType::Right
    } else if a.x == b.x && a.y < b.y {
        CaseType::Up
    } else if a.x > b.x && a.y < b.y {
        CaseType::UpLeft
    } else if a.x > b.x && a.y > b.y {
        CaseType::DownLeft
    } else if a.x < b.x && a.y > b.y {
        CaseType::DownRight
    } else {
        CaseType::UpRight
    }
}

/// Mirrors Python `is_in_out(a, b, c)`.
///
/// Return values intentionally match the Python script:
/// - `1`  => "inside" / admissible
/// - `-1` => "outside" / violates the supporting orthogonal wedge
/// - `99` => degenerate (`a == b`)
#[inline(always)]
fn is_in_out_impl(a: Point, b: Point, c: Point) -> i8 {
    match case_type(a, b) {
        CaseType::Left => {
            if c.y > a.y
                || (c.y == b.y && c.x < b.x)
                || (c.y == b.y && c.x > a.x)
            {
                -1
            } else {
                1
            }
        }
        CaseType::Down => {
            if c.x < a.x
                || (c.x == a.x && c.y > a.y)
                || (c.x == b.x && c.y < b.y)
            {
                -1
            } else {
                1
            }
        }
        CaseType::Right => {
            if c.y < a.y
                || (c.y == a.y && c.x < a.x)
                || (c.y == b.y && c.x > b.x)
            {
                -1
            } else {
                1
            }
        }
        CaseType::Up => {
            if c.x > a.x
                || (c.x == a.x && c.y < a.y)
                || (c.x == b.x && c.y > b.y)
            {
                -1
            } else {
                1
            }
        }
        CaseType::UpLeft => {
            if c.y < a.y
                || c.x < b.x
                || (c.y == a.y && b.x <= c.x && c.x <= a.x)
                || (c.x == b.x && a.y <= c.y && c.y <= b.y)
            {
                1
            } else {
                -1
            }
        }
        CaseType::DownLeft => {
            if c.x > a.x
                || c.y < b.y
                || (c.x == a.x && b.y <= c.y && c.y <= a.y)
                || (c.y == b.y && b.x <= c.x && c.x <= a.x)
            {
                1
            } else {
                -1
            }
        }
        CaseType::DownRight => {
            if c.x > b.x
                || c.y > a.y
                || (c.x == b.x && b.y <= c.y && c.y <= a.y)
                || (c.y == a.y && a.x <= c.x && c.x <= b.x)
            {
                1
            } else {
                -1
            }
        }
        CaseType::UpRight => {
            if c.x < a.x
                || c.y > b.y
                || (c.x == a.x && a.y <= c.y && c.y <= b.y)
                || (c.y == b.y && a.x <= c.x && c.x <= b.x)
            {
                1
            } else {
                -1
            }
        }
        CaseType::Same => 99,
    }
}

#[inline(always)]
fn is_in_out(a: Point, b: Point, c: Point) -> i8 {
    is_in_out_impl(a, b, c)
}

#[inline(always)]
fn is_in_out_tracked<S: StatsSink + ?Sized>(a: Point, b: Point, c: Point, stats: &mut S) -> i8 {
    stats.record_check_vertex();
    is_in_out_impl(a, b, c)
}

#[inline(always)]
fn position_tracked<S: StatsSink + ?Sized>(v: Point, d: &VecDeque<Point>, stats: &mut S) -> bool {
    if d.len() < 2 {
        return false;
    }
    is_in_out_tracked(d[0], d[1], v, stats) > 0
        && is_in_out_tracked(d[d.len() - 2], d[d.len() - 1], v, stats) > 0
}

#[inline(always)]
fn build_right_in_place_tracked<S: StatsSink + ?Sized>(d: &mut VecDeque<Point>, v: Point, stats: &mut S) {
    while d.len() >= 2 && is_in_out_tracked(d[d.len() - 2], d[d.len() - 1], v, stats) < 0 {
        d.pop_back();
        stats.record_add_remove_vertex(1);
    }
    d.push_back(v);
    stats.record_add_remove_vertex(1);
}

#[inline(always)]
fn build_left_in_place_tracked<S: StatsSink + ?Sized>(d: &mut VecDeque<Point>, v: Point, stats: &mut S) {
    while d.len() >= 2 && is_in_out_tracked(d[0], d[1], v, stats) < 0 {
        d.pop_front();
        stats.record_add_remove_vertex(1);
    }
    d.push_front(v);
    stats.record_add_remove_vertex(1);
}

fn orthogonal_hull_impl<S: StatsSink + ?Sized>(points: &[Point], stats: &mut S) -> Vec<Point> {
    match points.len() {
        0..=2 => return points.to_vec(),
        _ => {}
    }

    let mut d = VecDeque::with_capacity(points.len() + 1);

    if is_in_out_tracked(points[0], points[1], points[2], stats) > 0 {
        d.push_back(points[2]);
        d.push_back(points[0]);
        d.push_back(points[1]);
        d.push_back(points[2]);
    } else {
        d.push_back(points[2]);
        d.push_back(points[1]);
        d.push_back(points[0]);
        d.push_back(points[2]);
    }
    stats.record_add_remove_vertex(4);

    for &v in &points[3..] {
        if position_tracked(v, &d, stats) {
            continue;
        }
        build_right_in_place_tracked(&mut d, v, stats);
        build_left_in_place_tracked(&mut d, v, stats);
    }

    d.into_iter().collect()
}

/// Port of Python `ortho_hull(p)`.
///
/// The input is interpreted exactly as the Python script does: a contour-ordered
/// point sequence, often including a repeated final point if the contour is
/// closed. The returned vector preserves the deque order used by the reference.
pub fn orthogonal_hull(points: &[Point]) -> Vec<Point> {
    let mut stats = NoStats;
    orthogonal_hull_impl(points, &mut stats)
}

/// Instrumented orthogonal hull construction.
///
/// Returns the hull together with counters for the number of `is_in_out`
/// predicate checks and deque vertex mutations performed during construction.
pub fn orthogonal_hull_instrumented(points: &[Point]) -> (Vec<Point>, HullStats) {
    let mut stats = HullStats::default();
    let hull = orthogonal_hull_impl(points, &mut stats);
    (hull, stats)
}

/// Port of Python `is_ortho_convex(d)`.
pub fn is_orthogonally_convex(points: &[Point]) -> bool {
    if points.len() <= 1 {
        return true;
    }

    for i in 0..(points.len() - 1) {
        let mut count = 0usize;
        for &p in &points[..points.len() - 1] {
            if is_in_out(points[i], points[i + 1], p) > 0 {
                count += 1;
            }
        }
        if count != points.len() - 1 {
            return false;
        }
    }

    true
}

/// Port of Python `adding_points(d)` for plotting.
///
/// This keeps the same sequence shape as the reference helper, including the
/// duplicated initial point it emits before entering the main loop.
pub fn add_corner_points(points: &[Point]) -> Vec<Point> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut h = Vec::with_capacity(points.len() * 2 + 2);
    h.push(points[0]);

    if points.len() == 1 {
        return h;
    }

    for i in 0..(points.len() - 1) {
        let a = points[i];
        let b = points[i + 1];
        h.push(a);

        match case_type(a, b) {
            CaseType::UpLeft | CaseType::DownRight => h.push(Point::new(b.x, a.y)),
            CaseType::DownLeft | CaseType::UpRight => h.push(Point::new(a.x, b.y)),
            _ => {}
        }
    }

    h.push(points[points.len() - 1]);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_input_round_trips() {
        let pts = [Point::new(1, 2), Point::new(3, 4)];
        assert_eq!(orthogonal_hull(&pts), pts);
    }

    #[test]
    fn corner_expansion_keeps_endpoints() {
        let pts = [Point::new(0, 0), Point::new(2, 2)];
        let expanded = add_corner_points(&pts);
        assert_eq!(expanded.first().copied(), Some(Point::new(0, 0)));
        assert_eq!(expanded.last().copied(), Some(Point::new(2, 2)));
        assert!(expanded.len() >= 3);
    }

    #[test]
    fn instrumented_matches_plain_hull() {
        let pts = [
            Point::new(1, 1),
            Point::new(5, 1),
            Point::new(5, 2),
            Point::new(4, 2),
            Point::new(4, 4),
            Point::new(3, 4),
            Point::new(3, 5),
            Point::new(2, 5),
            Point::new(2, 3),
            Point::new(1, 3),
            Point::new(1, 1),
        ];
        let plain = orthogonal_hull(&pts);
        let (instrumented, stats) = orthogonal_hull_instrumented(&pts);
        assert_eq!(instrumented, plain);
        assert!(stats.check_vertex > 0);
        assert!(stats.add_remove_vertex >= 4);
    }
}
