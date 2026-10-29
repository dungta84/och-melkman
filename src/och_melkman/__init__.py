# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
"""O-Melkman: online construction of the orthogonal convex hull (OCH) of an orthogonal polygon,
and the aircraft-recognition pipeline of the accompanying paper."""
from .omelkman import Point, ortho_hull, is_ortho_convex, adding_points
__all__ = ["Point", "ortho_hull", "is_ortho_convex", "adding_points"]
