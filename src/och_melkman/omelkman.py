# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
"""O-Melkman reference implementation, copied verbatim from the author's o_hull_3_2_26.py
(functions Point, case_type, is_in_out, build_right/left, ortho_hull, is_ortho_convex, adding_points).
Only the top-level file-reading / plotting / benchmarking code was removed."""
from collections import deque
operation_counts = {"check_vertex": 0, "add_remove_vertex": 0}
bot, top = 0, -1
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def case_type(a: Point, b: Point):
    if a.x == b.x and a.y == b.y:
        d = 0
    else:
        if a.x > b.x and a.y == b.y: d = -1
        elif a.x == b.x and a.y > b.y: d = -2
        elif a.x < b.x and a.y == b.y: d = -3
        elif a.x == b.x and a.y < b.y: d = -4
        elif a.x > b.x and a.y < b.y: d = 1
        elif a.x > b.x and a.y > b.y: d = 2
        elif a.x < b.x and a.y > b.y: d = 3
        elif a.x < b.x and a.y < b.y: d = 4
    return d

def is_in_out(a: Point, b: Point, c: Point):
    global operation_counts
    operation_counts["check_vertex"] += 1  # Ghi nhận số lần kiểm tra đỉnh
    d_type = case_type(a, b)
    match d_type:
        case -1:
            if c.y > a.y: chk = -1
            elif c.y == b.y and c.x < b.x: chk = -1
            elif c.y == b.y and c.x > a.x: chk = -1
            else: chk = 1
        case -2:
            if c.x < a.x: chk = -1
            elif c.x == a.x and c.y > a.y: chk = -1
            elif c.x == b.x and c.y < b.y: chk = -1
            else: chk = 1
        case -3:
            if c.y < a.y: chk = -1
            elif c.y == a.y and c.x < a.x: chk = -1
            elif c.y == b.y and c.x > b.x: chk = -1
            else: chk = 1
        case -4:
            if c.x > a.x: chk = -1
            elif c.x == a.x and c.y < a.y: chk = -1
            elif c.x == b.x and c.y > b.y: chk = -1
            else: chk = 1
        case 1:
            if c.y < a.y: chk = 1
            elif c.x < b.x: chk = 1
            elif c.y == a.y and b.x <= c.x <= a.x: chk = 1
            elif c.x == b.x and a.y <= c.y <= b.y: chk = 1
            else: chk = -1
        case 2:
            if c.x > a.x: chk = 1
            elif c.y < b.y: chk = 1
            elif c.x == a.x and b.y <= c.y <= a.y: chk = 1
            elif c.y == b.y and b.x <= c.x <= a.x: chk = 1
            else: chk = -1
        case 3:
            if c.x > b.x: chk = 1
            elif c.y > a.y: chk = 1
            elif c.x == b.x and b.y <= c.y <= a.y: chk = 1
            elif c.y == a.y and a.x <= c.x <= b.x: chk = 1
            else: chk = -1
        case 4:
            if c.x < a.x: chk = 1
            elif c.y > b.y: chk = 1
            elif c.x == a.x and a.y <= c.y <= b.y: chk = 1
            elif c.y == b.y and a.x <= c.x <= b.x: chk = 1
            else: chk = -1
        case 0: chk = 99
    return chk
#
def position(v: Point, D: deque):
    return is_in_out(D[0], D[1], v) > 0 and is_in_out(D[-2], D[-1], v) > 0
#

#
bot, top = 0, -1
def build_right(d: deque, v: Point):
    global operation_counts
    while is_in_out(d[top-1], d[top],v) < 0:
        d.pop()
        #operation_counts['add_remove_vertex'] += 1
    d.append(v)
    return d
#
def build_left(d: deque, v: Point):
    global operation_counts
    while is_in_out(d[bot], d[bot+1], v) < 0:
        d.popleft()
        #operation_counts['add_remove_vertex'] += 1
    d.appendleft(v)
    return d
#
def ortho_hull(p: list[Point]):
    global operation_counts
    d = []
    if is_in_out(p[0], p[1], p[2]) > 0:
        d.append(p[0])
        d.append(p[1])
    else:
        d.append(p[1])
        d.append(p[0]) 
    d.insert(0, p[2])
    d.append(p[2])
    d = deque(d)
    #operation_counts["add_remove_vertex"] += 4  # Ghi nhận thao tác thêm đỉnh ban đầu
    for i in range(3, len(p)):
        # if POSITION(v, d) == True: 
        if is_in_out(d[0], d[1], p[i]) > 0 and is_in_out(d[-2], d[-1], p[i]) > 0:
            #operation_counts['check_vertex'] += 1  # Ghi nhận thao tác kiểm tra đỉnh
            continue
        else:
            d = build_right(d, p[i])
            d = build_left(d, p[i])
    return d
#
def is_ortho_convex(d):
    for i in range(len(d) - 1):
        count = 0
        for k in range(len(d) - 1):
            if is_in_out(d[i], d[i + 1], d[k]) > 0:
                count += 1
        if count == len(d) - 1:
            continue
        else:
            return False
    return True
#
def adding_points(d):
    """Adding points in corners to draw the current o-hull of d"""
    h = []
    h.append(d[0])  # Thêm điểm đầu tiên vào danh sách

    for i in range(len(d) - 1):
        h.append(d[i])  # Thêm điểm hiện tại vào danh sách
        d_type = case_type(d[i], d[i + 1])  # Xác định loại trường hợp

        match d_type:
            case -1:
                pass  # Do nothing
            case -2:
                pass  # Do nothing
            case -3:
                pass  # Do nothing
            case -4:
                pass  # Do nothing
            case 1:
                h.append(Point(d[i + 1].x, d[i].y))  # Thêm điểm mới
            case 2:
                h.append(Point(d[i].x, d[i + 1].y))  # Thêm điểm mới
            case 3:
                h.append(Point(d[i + 1].x, d[i].y))  # Thêm điểm mới
            case 4:
                h.append(Point(d[i].x, d[i + 1].y))  # Thêm điểm mới

    h.append(d[-1])  # Thêm điểm cuối cùng vào danh sách
    return h
#
