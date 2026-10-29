// Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
// Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
//! Minimal, memory-conscious orthogonal geometry primitives.
//!
//! The Python reference script operates on integer grid coordinates and only
//! uses axis-aligned edges. We keep everything integral to avoid float
//! roundoff and to preserve exact equality semantics.

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    #[inline]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub const fn translated(self, dx: i32, dy: i32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Segment {
    pub a: Point,
    pub b: Point,
}

impl Segment {
    #[inline]
    pub const fn new(a: Point, b: Point) -> Self {
        Self { a, b }
    }

    #[inline]
    pub const fn is_vertical(&self) -> bool {
        self.a.x == self.b.x
    }

    #[inline]
    pub const fn is_horizontal(&self) -> bool {
        self.a.y == self.b.y
    }

    #[inline]
    pub const fn x_min(&self) -> i32 {
        if self.a.x < self.b.x { self.a.x } else { self.b.x }
    }

    #[inline]
    pub const fn x_max(&self) -> i32 {
        if self.a.x > self.b.x { self.a.x } else { self.b.x }
    }

    #[inline]
    pub const fn y_min(&self) -> i32 {
        if self.a.y < self.b.y { self.a.y } else { self.b.y }
    }

    #[inline]
    pub const fn y_max(&self) -> i32 {
        if self.a.y > self.b.y { self.a.y } else { self.b.y }
    }
}

/// Inclusive axis-aligned rectangle.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Rect {
    pub min: Point,
    pub max: Point,
}

impl Rect {
    #[inline]
    pub const fn from_points(p: Point, q: Point) -> Self {
        let min = Point::new(if p.x < q.x { p.x } else { q.x }, if p.y < q.y { p.y } else { q.y });
        let max = Point::new(if p.x > q.x { p.x } else { q.x }, if p.y > q.y { p.y } else { q.y });
        Self { min, max }
    }

    #[inline]
    pub const fn contains(&self, p: Point) -> bool {
        self.min.x <= p.x && p.x <= self.max.x && self.min.y <= p.y && p.y <= self.max.y
    }
}

/// Twice-signed area (determinant) used by the Python script (`det`).
#[inline]
pub fn det(p1: Point, p2: Point, p3: Point) -> i64 {
    // i64 to reduce overflow risk if coordinates grow.
    (p1.x as i64) * ((p2.y - p3.y) as i64)
        + (p2.x as i64) * ((p3.y - p1.y) as i64)
        + (p3.x as i64) * ((p1.y - p2.y) as i64)
}

/// Matches the Python `is_point_on_edge` semantics for orthogonal segments.
#[inline]
pub fn is_point_on_segment(p: Point, s: Segment) -> bool {
    if s.is_vertical() {
        p.x == s.a.x && s.y_min() <= p.y && p.y <= s.y_max()
    } else if s.is_horizontal() {
        p.y == s.a.y && s.x_min() <= p.x && p.x <= s.x_max()
    } else {
        // The reference generator only expects orthogonal edges; treat non-orthogonal as miss.
        false
    }
}

/// Intersection between an orthogonal segment `s` and a horizontal probe segment.
///
/// IMPORTANT: The reference `line_intersection` returns `None` for parallel or
/// coincident segments (including colinear overlap). We replicate that: if `s`
/// is horizontal (parallel to probe), we return `None` even if colinear.
#[inline]
pub fn intersect_with_horizontal_probe(s: Segment, y: i32, x0: i32, x1: i32) -> Option<Point> {
    if s.is_horizontal() {
        return None;
    }
    if !s.is_vertical() {
        return None;
    }
    let xmin = if x0 < x1 { x0 } else { x1 };
    let xmax = if x0 > x1 { x0 } else { x1 };
    let x = s.a.x;
    if xmin <= x && x <= xmax && s.y_min() <= y && y <= s.y_max() {
        Some(Point::new(x, y))
    } else {
        None
    }
}

/// Intersection between an orthogonal segment `s` and a vertical probe segment.
/// See `intersect_with_horizontal_probe` for parallel/colinear behavior.
#[inline]
pub fn intersect_with_vertical_probe(s: Segment, x: i32, y0: i32, y1: i32) -> Option<Point> {
    if s.is_vertical() {
        return None;
    }
    if !s.is_horizontal() {
        return None;
    }
    let ymin = if y0 < y1 { y0 } else { y1 };
    let ymax = if y0 > y1 { y0 } else { y1 };
    let y = s.a.y;
    if ymin <= y && y <= ymax && s.x_min() <= x && x <= s.x_max() {
        Some(Point::new(x, y))
    } else {
        None
    }
}
