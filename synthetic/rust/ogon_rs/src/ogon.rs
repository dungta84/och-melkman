// Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
// Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
//! Port of the core logic from `orthogonal_polygon.py`.
//!
//! This version keeps the same inflate+cut algorithm, but removes several
//! avoidable full-polygon rescans from the first port:
//! - nearest probe intersections are found in one pass (no temporary Vecs),
//! - the owning edge for each chosen intersection is captured during that pass,
//! - the target vertex index is carried through directly (no point re-scan),
//! - replacing one vertex with three performs a single tail shift.

use std::ptr;

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

use crate::geometry::{
    det, intersect_with_horizontal_probe, intersect_with_vertical_probe, Point, Rect, Segment,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct VertexRef {
    point: Point,
    idx: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct CutChoice {
    s: Point,
    // Python's list_v_m order is [edge[1], edge[0]] for each chosen intersection.
    candidates: [VertexRef; 2],
}

/// In-place version of Python `inflate(p, c)`.
///
/// For each vertex coordinate strictly greater than `c`, shift by +1.
#[inline(always)]
pub fn inflate_in_place(p: &mut [Point], c: Point) {
    for v in p.iter_mut() {
        if v.x > c.x {
            v.x += 1;
        }
        if v.y > c.y {
            v.y += 1;
        }
    }
}

/// Python's `min_max_coor` has a bug: it returns `(min_x, max_y)`.
///
/// We preserve that behavior for parity.
#[inline(always)]
fn min_x_and_max_y_like_python(points: &[Point]) -> (i32, i32) {
    let mut min_x = i32::MAX;
    let mut max_y = i32::MIN;
    for &p in points {
        if p.x < min_x {
            min_x = p.x;
        }
        if p.y > max_y {
            max_y = p.y;
        }
    }
    (min_x, max_y)
}

#[inline(always)]
fn canonical_index_for_closed_polygon(polygon_len: usize, idx: usize) -> usize {
    if idx + 1 == polygon_len {
        0
    } else {
        idx
    }
}

#[inline(always)]
fn build_cut_choice(polygon_len: usize, edge_idx: usize, s: Point, edge: Segment) -> CutChoice {
    let a = VertexRef {
        point: edge.a,
        idx: edge_idx,
    };
    let b = VertexRef {
        point: edge.b,
        idx: canonical_index_for_closed_polygon(polygon_len, edge_idx + 1),
    };

    CutChoice {
        s,
        candidates: [b, a],
    }
}

#[inline(always)]
fn update_left(best: &mut Option<CutChoice>, polygon_len: usize, edge_idx: usize, edge: Segment, p: Point, target_x: i32) {
    if p.x < target_x {
        match best {
            Some(current) if current.s.x >= p.x => {}
            _ => *best = Some(build_cut_choice(polygon_len, edge_idx, p, edge)),
        }
    }
}

#[inline(always)]
fn update_right(best: &mut Option<CutChoice>, polygon_len: usize, edge_idx: usize, edge: Segment, p: Point, target_x: i32) {
    if p.x > target_x {
        match best {
            Some(current) if current.s.x <= p.x => {}
            _ => *best = Some(build_cut_choice(polygon_len, edge_idx, p, edge)),
        }
    }
}

#[inline(always)]
fn update_down(best: &mut Option<CutChoice>, polygon_len: usize, edge_idx: usize, edge: Segment, p: Point, target_y: i32) {
    if p.y < target_y {
        match best {
            Some(current) if current.s.y >= p.y => {}
            _ => *best = Some(build_cut_choice(polygon_len, edge_idx, p, edge)),
        }
    }
}

#[inline(always)]
fn update_up(best: &mut Option<CutChoice>, polygon_len: usize, edge_idx: usize, edge: Segment, p: Point, target_y: i32) {
    if p.y > target_y {
        match best {
            Some(current) if current.s.y <= p.y => {}
            _ => *best = Some(build_cut_choice(polygon_len, edge_idx, p, edge)),
        }
    }
}

#[inline(always)]
fn shuffle4<T, R: Rng>(arr: &mut [T; 4], rng: &mut R) {
    let j3 = rng.random_range(0..4);
    arr.swap(3, j3);
    let j2 = rng.random_range(0..3);
    arr.swap(2, j2);
    let j1 = rng.random_range(0..2);
    arr.swap(1, j1);
}

/// Mirrors Python `intersection_endpoints(p_tilde, c)`, but in two tight passes:
///
/// 1) scan points once for Python's buggy `(min_x, max_y)` bounds,
/// 2) scan edges once and directly keep only the nearest left/right/down/up
///    intersections *and* the corresponding candidate vertices.
fn intersection_choices<R: Rng>(p_tilde: &[Point], c: Point, rng: &mut R) -> [CutChoice; 4] {
    let (min_x, max_y_like) = min_x_and_max_y_like_python(p_tilde);

    // These match the Python construction:
    // line1 = [[min_x - 1, c[1] + 1], [max_x + 1, c[1] + 1]]
    // line2 = [[c[0] + 1, min_x - 1], [c[0] + 1, max_x + 1]]
    let hy = c.y + 1;
    let hx0 = min_x - 1;
    let hx1 = max_y_like + 1;

    let vx = c.x + 1;
    let vy0 = min_x - 1;
    let vy1 = max_y_like + 1;

    let polygon_len = p_tilde.len();
    let v_c = Point::new(vx, hy);

    let mut left: Option<CutChoice> = None;
    let mut right: Option<CutChoice> = None;
    let mut down: Option<CutChoice> = None;
    let mut up: Option<CutChoice> = None;

    // Python iterates `for i in range(len(polygon) - 1)` and treats polygon as closed (last==first).
    for i in 0..polygon_len.saturating_sub(1) {
        let edge = Segment::new(p_tilde[i], p_tilde[i + 1]);

        if let Some(p) = intersect_with_horizontal_probe(edge, hy, hx0, hx1) {
            update_left(&mut left, polygon_len, i, edge, p, v_c.x);
            update_right(&mut right, polygon_len, i, edge, p, v_c.x);
        }
        if let Some(p) = intersect_with_vertical_probe(edge, vx, vy0, vy1) {
            update_down(&mut down, polygon_len, i, edge, p, v_c.y);
            update_up(&mut up, polygon_len, i, edge, p, v_c.y);
        }
    }

    // Python: list_s = [up, right, down, left]
    let mut choices = [
        up.expect("expected an intersection above v_c"),
        right.expect("expected an intersection right of v_c"),
        down.expect("expected an intersection below v_c"),
        left.expect("expected an intersection left of v_c"),
    ];

    // Python: mixed_list = list_s.copy(); random.shuffle(mixed_list)
    shuffle4(&mut choices, rng);
    choices
}

/// Mirrors Python `can_cut(p_tilde, v_m, v_c)`.
///
/// Note: Python returns `True` if count==1, `False` if count>1, and `None` if count==0.
/// In the calling code it's used as a predicate; we treat `None` as `false`.
#[inline(always)]
fn can_cut(p_tilde: &[Point], v_m: Point, v_c: Point) -> bool {
    let rect = Rect::from_points(v_m, v_c);
    let mut count = 0u32;
    for &p in p_tilde {
        if rect.contains(p) {
            count += 1;
            if count > 1 {
                return false;
            }
        }
    }
    count == 1
}

/// Replace `p[idx]` with `[p0, p1, p2]` using a single tail shift.
#[inline(always)]
fn replace_one_with_three_in_place(p: &mut Vec<Point>, idx: usize, p0: Point, p1: Point, p2: Point) {
    let old_len = p.len();
    debug_assert!(idx < old_len);

    p.reserve(2);

    unsafe {
        let ptr_points = p.as_mut_ptr();
        let tail_len = old_len - (idx + 1);

        // Shift the tail right by two slots. `ptr::copy` supports overlap.
        ptr::copy(
            ptr_points.add(idx + 1),
            ptr_points.add(idx + 3),
            tail_len,
        );

        ptr::write(ptr_points.add(idx), p0);
        ptr::write(ptr_points.add(idx + 1), p1);
        ptr::write(ptr_points.add(idx + 2), p2);

        p.set_len(old_len + 2);
    }
}

/// In-place version of Python `cut(p_tilde, c)`.
///
/// Returns `true` on success (polygon mutated), `false` on failure.
fn cut_in_place<R: Rng>(p_tilde: &mut Vec<Point>, c: Point, rng: &mut R) -> bool {
    let v_c = Point::new(c.x + 1, c.y + 1);
    let choices = intersection_choices(p_tilde, c, rng);

    for choice in choices {
        for candidate in choice.candidates {
            let v = candidate.point;
            if !can_cut(p_tilde, v, v_c) {
                continue;
            }

            let s = choice.s;
            let s_prime = Point::new(v_c.x + v.x - s.x, v_c.y + v.y - s.y);

            let (p0, p1, p2) = if det(s, v_c, s_prime) < 0 {
                (s, v_c, s_prime)
            } else {
                (s_prime, v_c, s)
            };

            replace_one_with_three_in_place(p_tilde, candidate.idx, p0, p1, p2);
            return true;
        }
    }

    false
}

/// Mirrors Python `choose_c(p)`.
#[inline(always)]
fn choose_c<R: Rng>(p: &[Point], rng: &mut R) -> (Point, Point) {
    let n = p.len();
    assert!(n >= 2, "polygon must have at least 2 points");

    // Python: i = random.randint(0, len(p) - 2)
    let i = rng.random_range(0..n);
    let a = p[i];
    let b = p[i + 1];

    let (cx, cy) = if a.x < b.x && a.y == b.y {
        (a.x, a.y)
    } else if a.x == b.x && a.y < b.y {
        (a.x - 1, a.y)
    } else if a.x > b.x && a.y == b.y {
        (a.x - 1, a.y - 1)
    } else {
        (a.x, a.y - 1)
    };

    (a, Point::new(cx, cy))
}

/// Mirrors Python `choose_c_n(p)`.
#[inline(always)]
fn choose_c_n(p: &[Point]) -> Option<(Point, Point)> {
    for i in 0..p.len().saturating_sub(1) {
        let a = p[i];
        let b = p[i + 1];

        if a.x - b.x >= 2 {
            let c = Point::new(a.x - 2, a.y - 1);
            return Some((a, c));
        } else if b.x - a.x >= 2 {
            let c = Point::new(b.x - 1, b.y);
            return Some((a, c));
        } else if b.y - a.y >= 2 {
            let c = Point::new(b.x - 1, b.y - 1);
            return Some((a, c));
        } else if a.y - b.y >= 2 {
            let c = Point::new(a.x, a.y - 1);
            return Some((a, c));
        }
    }
    None
}

/// Generate an orthogonal grid polygon with approximately `n` vertices.
///
/// This mirrors Python `generate_ogon(n)`.
///
/// Notes:
/// - The Python script treats the polygon as *closed* (last vertex equals first).
/// - The reference sets `r = n/2 - 3` and applies `r` inflate+cut steps.
/// - `seed` is optional but recommended for reproducibility.
pub fn generate_ogon(n: usize, seed: u64) -> Vec<Point> {
    // Start polygon (closed). The Python file has a tuple literal here; we use the intended list.
    let mut p: Vec<Point> = vec![
        Point::new(1, 1),
        Point::new(3, 1),
        Point::new(3, 3),
        Point::new(2, 3),
        Point::new(2, 2),
        Point::new(1, 2),
        Point::new(1, 1),
    ];

    if n <= 6 {
        return p;
    }

    let r = (n / 2).saturating_sub(3);

    // Reserve to avoid repeated reallocation: each iteration increases length by 2.
    let final_len = p.len() + 2 * r;
    if final_len > p.capacity() {
        p.reserve_exact(final_len - p.len());
    }

    let mut rng = SmallRng::seed_from_u64(seed);

    for k in 0..r {
        let (_v, c) = if k % 2 == 0 {
            choose_c_n(&p).unwrap_or_else(|| choose_c(&p, &mut rng))
        } else {
            choose_c(&p, &mut rng)
        };

        inflate_in_place(&mut p, c);
        let ok = cut_in_place(&mut p, c, &mut rng);
        if !ok {
            // Matches Python's "cut fails" outcome; we keep the current polygon.
            break;
        }
    }

    // Ensure closed.
    if p.first().copied() != p.last().copied() {
        let first = p[0];
        p.push(first);
    }

    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_closed_polygon() {
        let p = generate_ogon(10, 0);
        assert_eq!(p.first(), p.last());
        assert!(p.len() >= 7);
    }
}
