"""Geometry — the words for position and shape with distance.

Points are "(x, y)", polygons are lists of them. The faculty
hears "the area of the triangle (0,0) (4,0) (0,3)", "how far
apart", "is the point inside", "the convex hull" — and answers
with a number and its derivation. Convexity of a hull is decided
by walking it, and winding is honest orientation.

The machine half is ``src/math/geometry.rs``: distance, polygon
area (the shoelace), point-in-polygon (ray casting), convex hull
(the monotone chain). Here: the coordinate words and the reply.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeoResult:
    """The outcome of a geometric question."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


Point = tuple[float, float]

_POINT_LIT = re.compile(r"\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?")


def parse_points(text: str) -> list[Point]:
    """Read "(0,0) (4,0) (0,3)" or "[0,0; 4,0]" into points."""
    return [(float(a), float(b)) for a, b in _POINT_LIT.findall(text)]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polygon_area(pts: list[Point]) -> float:
    """The shoelace — half the sum of cross products of edges."""
    n = len(pts)
    s = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n))
    return abs(s) / 2


def signed_area(pts: list[Point]) -> float:
    """Signed shoelace — positive means counterclockwise winding."""
    n = len(pts)
    return sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
               for i in range(n)) / 2


def point_in_polygon(p: Point, pts: list[Point]) -> bool:
    """Ray casting — count edge crossings of a ray going right."""
    inside = False
    n = len(pts)
    x, y = p
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


def convex_hull(pts: list[Point]) -> list[Point]:
    """The monotone chain — sorted points walked around the hull
    keeping only counterclockwise turns."""
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts
    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


_GEO_WORD = re.compile(
    r"\b(?:distance|area of|perimeter|point inside|inside the|"
    r"convex hull|hull of|triangle|polygon|circle|rectangle|"
    r"centroid|circum|collinear|angle between|points?)\b",
    re.IGNORECASE,
)


def looks_like_geometry(text: str) -> bool:
    """Whether the text asks about points, areas, hulls — shape
    with distance."""
    return bool(_GEO_WORD.search(text) and _POINT_LIT.search(text))


def distance_text(text: str) -> GeoResult:
    """"distance between (0,0) and (3,4)" -> 5."""
    pts = parse_points(text)
    if len(pts) < 2:
        return GeoResult("distance", text, "", False, "need two points")
    d = distance(pts[0], pts[1])
    dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    return GeoResult("distance", text,
                     f"{d:.6g}   (√({dx:g}² + {dy:g}²) = √{dx*dx+dy*dy:g})",
                     True, data={"distance": d})


def area_text(text: str) -> GeoResult:
    """"area of (0,0) (4,0) (0,3)" -> 6 — the shoelace's answer."""
    pts = parse_points(text)
    if len(pts) < 3:
        return GeoResult("area", text, "", False, "need at least three points")
    a = polygon_area(pts)
    wind = "counterclockwise" if signed_area(pts) > 0 else "clockwise"
    return GeoResult("area", text, f"{a:.6g}   (wound {wind})",
                     True, data={"area": a})


def hull_text(text: str) -> GeoResult:
    """"convex hull of ..." -> the vertices that hold the shape."""
    pts = parse_points(text)
    if len(pts) < 3:
        return GeoResult("hull", text, "", False, "need at least three points")
    hull = convex_hull(pts)
    out = " ".join(f"({x:g}, {y:g})" for x, y in hull)
    return GeoResult("hull", text,
                     f"{len(hull)} vertices: {out}", True,
                     data={"hull": hull, "area": polygon_area(hull)})


def inside_text(point_text: str, poly_text: str) -> GeoResult:
    """"is (1,1) inside (0,0) (4,0) (0,4)" -> yes."""
    p = parse_points(point_text)
    poly = parse_points(poly_text)
    if not p or len(poly) < 3:
        return GeoResult("inside", f"{point_text} in {poly_text}", "", False,
                         "need a point and a polygon")
    inside = point_in_polygon(p[0], poly)
    return GeoResult("inside", f"{point_text} in {poly_text}",
                     f"{p[0]} is {'inside' if inside else 'outside'} the polygon",
                     True, data={"inside": inside})
