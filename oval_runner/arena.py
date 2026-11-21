"""Static collision queries for the competition map."""
from array import array
import json
import math
from pathlib import Path
import sys


CELL = 28
EDGE_CELL = 8
EPS = 1e-6


def _edges(points):
    pairs = tuple(zip(points[::2], points[1::2]))
    return tuple((x, y, nx - x, ny - y)
                 for (x, y), (nx, ny) in zip(pairs, pairs[1:] + pairs[:1]))


def _polygon(edges):
    if len(edges) < 40:
        return edges, None
    rows = {}
    for edge in edges:
        _, y, _, dy = edge
        for row in range(math.floor(min(y, y + dy) / EDGE_CELL),
                         math.floor(max(y, y + dy) / EDGE_CELL) + 1):
            rows.setdefault(row, []).append(edge)
    return edges, rows


def _inside(x, y, polygon):
    edges, rows = polygon
    if rows is not None:
        edges = rows.get(math.floor(y / EDGE_CELL), ())
    inside = False
    for ax, ay, dx, dy in edges:
        if (ay > y) != (ay + dy > y) and x < ax + dx * (y - ay) / dy:
            inside = not inside
    return inside


class _Prism:
    __slots__ = ("x0", "y0", "x1", "y1", "base", "top", "ring", "holes", "edges", "edge_cells")

    def __init__(self, raw):
        ring = raw["ring"]
        self.x0, self.x1 = min(ring[::2]), max(ring[::2])
        self.y0, self.y1 = min(ring[1::2]), max(ring[1::2])
        self.base = raw["base"]
        self.top = raw["top"]
        self.ring = _polygon(_edges(ring))
        self.holes = tuple(_polygon(_edges(hole)) for hole in raw["holes"])
        self.edges = self.ring[0] + tuple(edge for edges, _ in self.holes for edge in edges)
        self.edge_cells = None
        if len(self.edges) >= 40:
            self.edge_cells = {}
            for edge in self.edges:
                x, y, dx, dy = edge
                cols = range(math.floor(min(x, x + dx) / EDGE_CELL),
                             math.floor(max(x, x + dx) / EDGE_CELL) + 1)
                rows = range(math.floor(min(y, y + dy) / EDGE_CELL),
                             math.floor(max(y, y + dy) / EDGE_CELL) + 1)
                for col in cols:
                    for row in rows:
                        self.edge_cells.setdefault((col, row), []).append(edge)

    def contains(self, x, y):
        return _inside(x, y, self.ring) and not any(
            _inside(x, y, hole) for hole in self.holes
        )

    def overlaps_disk(self, x, y, radius):
        if self.contains(x, y):
            return True
        edges = self.edges
        if self.edge_cells is not None:
            edges = [
                edge
                for col in range(math.floor((x - radius) / EDGE_CELL), math.floor((x + radius) / EDGE_CELL) + 1)
                for row in range(math.floor((y - radius) / EDGE_CELL), math.floor((y + radius) / EDGE_CELL) + 1)
                for edge in self.edge_cells.get((col, row), ())
            ]
        radius2 = radius * radius
        for ax, ay, dx, dy in edges:
            ox, oy = x - ax, y - ay
            length2 = dx * dx + dy * dy
            if length2 > EPS:
                normal = min(1.0, max(0.0, (ox * dx + oy * dy) / length2))
                ox -= normal * dx
                oy -= normal * dy
            if ox * ox + oy * oy < radius2:
                return True
        return False

    def blocks_ray(self, ax, ay, az, dx, dy, dz):
        near, far = 0.0, 1.0
        # Clip to the solid's z range and footprint bounding box before
        # checking its polygon. This preserves bridge undersides and holes.
        for origin, delta, lower, upper in (
            (ax, dx, self.x0, self.x1), (ay, dy, self.y0, self.y1),
            (az, dz, self.base, self.top),
        ):
            if abs(delta) < EPS:
                if origin < lower or origin > upper:
                    return False
                continue
            t0, t1 = (lower - origin) / delta, (upper - origin) / delta
            if t0 > t1:
                t0, t1 = t1, t0
            near, far = max(near, t0), min(far, t1)
            if near > far:
                return False
        if self.contains(ax + near * dx, ay + near * dy):
            return True
        for ex, ey, ux, uy in self.edges:
            cross = dx * uy - dy * ux
            if abs(cross) < EPS:
                continue
            ox, oy = ex - ax, ey - ay
            t = (ox * uy - oy * ux) / cross
            if near - EPS <= t <= far + EPS and -EPS <= (ox * dy - oy * dx) / cross <= 1 + EPS:
                return True
        return False


class Arena:
    """Read-only map index. Missing map data disables only the affected query."""

    def __init__(self, hello):
        self.cells = {}
        self.terrain = None
        self.bounds = None
        self.slope = math.tan(math.radians(
            hello.get("config", {}).get("mech", {}).get("max_slope_deg", 50)
        ))
        info = hello.get("map", {})
        if not info.get("dir"):
            return
        root = Path(info["dir"])
        files = info.get("files", ())
        world_path = next((name for name in files if name.endswith("world.json")), "v2/world.json")
        try:
            world = json.loads((root / world_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        bounds = world["bounds"]
        self.bounds = bounds["minX"], bounds["minY"], bounds["maxX"], bounds["maxY"]
        for raw in world["prisms"]:
            prism = _Prism(raw)
            for col in range(math.floor(prism.x0 / CELL), math.floor(prism.x1 / CELL) + 1):
                for row in range(math.floor(prism.y0 / CELL), math.floor(prism.y1 / CELL) + 1):
                    self.cells.setdefault((col, row), []).append(prism)
        grid = world["terrain"]
        terrain_path = next((name for name in files if name.endswith("terrain.i16")),
                            str(Path(world_path).with_name(grid["file"])))
        try:
            terrain = array("h", (root / terrain_path).read_bytes())
        except (OSError, ValueError):
            return
        if sys.byteorder != "little":
            terrain.byteswap()
        self.grid = grid["minX"], grid["minY"], grid["nx"], grid["ny"], grid["step"], grid.get("scale", 0.01)
        if len(terrain) == grid["nx"] * grid["ny"]:
            self.terrain = terrain
            self.highest_ground = max(terrain) * self.grid[-1]

    def ground(self, x, y):
        data = self.terrain
        if data is None:
            return None
        x0, y0, nx, ny, step, scale = self.grid
        gx, gy = (x - x0) / step, (y - y0) / step
        i, j = int(gx), int(gy)
        if not (0 <= gx < nx - 1 and 0 <= gy < ny - 1):
            return None
        tx, ty = gx - i, gy - j
        offset = j * nx + i
        a = data[offset] * (1 - tx) + data[offset + 1] * tx
        b = data[offset + nx] * (1 - tx) + data[offset + nx + 1] * tx
        return (a * (1 - ty) + b * ty) * scale

    def _nearby(self, x0, y0, x1, y1):
        col0, row0 = math.floor(x0 / CELL), math.floor(y0 / CELL)
        col1, row1 = math.floor(x1 / CELL), math.floor(y1 / CELL)
        if col0 == col1 and row0 == row1:
            return self.cells.get((col0, row0), ())
        prisms = {}
        for col in range(col0, col1 + 1):
            for row in range(row0, row1 + 1):
                for prism in self.cells.get((col, row), ()):
                    prisms[prism] = None
        return prisms

    def _ray_prisms(self, ax, ay, bx, by):
        """Visit only bins crossed by the ray, in traversal order."""
        col, row = math.floor(ax / CELL), math.floor(ay / CELL)
        end_col, end_row = math.floor(bx / CELL), math.floor(by / CELL)
        dx, dy = bx - ax, by - ay
        step_x, step_y = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        tx_stride = CELL / abs(dx) if dx else math.inf
        ty_stride = CELL / abs(dy) if dy else math.inf
        tx = ((col + (dx > 0)) * CELL - ax) / dx if dx else math.inf
        ty = ((row + (dy > 0)) * CELL - ay) / dy if dy else math.inf
        seen = set()
        while True:
            for prism in self.cells.get((col, row), ()):
                if prism not in seen:
                    seen.add(prism)
                    yield prism
            if col == end_col and row == end_row:
                return
            if col != end_col and (row == end_row or tx <= ty):
                col += step_x
                tx += tx_stride
            else:
                row += step_y
                ty += ty_stride

    def clear(self, a, b):
        """Return whether the shot segment avoids static solid geometry."""
        ax, ay, az = a
        bx, by, bz = b
        dx, dy, dz = bx - ax, by - ay, bz - az
        zmin, zmax = min(az, bz), max(az, bz)
        for prism in self._ray_prisms(ax, ay, bx, by):
            if prism.base <= zmax and prism.top >= zmin and prism.blocks_ray(ax, ay, az, dx, dy, dz):
                return False
        if self.terrain is None or zmin >= self.highest_ground:
            return True
        # Sample in horizontal distance because the height field is 2-D.
        steps = max(1, math.ceil(math.hypot(dx, dy) / 3))
        for i in range(1, steps + 1):
            t = i / steps
            z = self.ground(ax + t * dx, ay + t * dy)
            if z is not None and z > az + t * dz + 0.05:
                return False
        return True

    def walk_fraction(self, pos, dx, dy, radius=2.5, height=9.0, step_up=1.2):
        """Estimate the reachable fraction of a short horizontal step."""
        if self.bounds is None:
            return 1.0
        ax, ay, az = pos
        xmin, ymin, xmax, ymax = self.bounds
        # The server tracks feet and resolves wall sliding. Keep a small margin
        # to avoid predicting a collision just because reported coordinates round.
        radius = max(0.0, radius - 0.2)
        length = math.hypot(dx, dy)
        steps = max(1, math.ceil(length / max(1.5, radius)))
        stride = length / steps
        max_rise = max(step_up, stride * self.slope) + 0.05
        prisms = self._nearby(min(ax, ax + dx) - radius, min(ay, ay + dy) - radius,
                              max(ax, ax + dx) + radius, max(ay, ay + dy) + radius)
        feet = az
        for i in range(1, steps + 1):
            fraction = i / steps
            x, y = ax + fraction * dx, ay + fraction * dy
            if not (xmin + radius < x < xmax - radius and ymin + radius < y < ymax - radius):
                return (i - 1) / steps
            ground = self.ground(x, y)
            if ground is not None and ground > feet + max_rise:
                return (i - 1) / steps
            if ground is not None:
                feet = max(feet, ground)
            for prism in prisms:
                if (prism.top <= feet + step_up + 0.05 or prism.base >= feet + height - 0.05
                    or x + radius <= prism.x0 or x - radius >= prism.x1
                    or y + radius <= prism.y0 or y - radius >= prism.y1):
                    continue
                if prism.overlaps_disk(x, y, radius):
                    return (i - 1) / steps
        return 1.0
