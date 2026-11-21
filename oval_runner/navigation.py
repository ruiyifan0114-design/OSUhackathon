"""Incremental ground routes for zone rotation and disengagement."""
from heapq import heappop, heappush
import math


CELL = 8.0
NEIGHBORS = ((1, 0), (0, 1), (-1, 0), (0, -1),
             (1, 1), (-1, 1), (-1, -1), (1, -1))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Search:
    def __init__(self, arena, pos, goal, clearance):
        self.arena, self.origin, self.goal = arena, tuple(pos), tuple(goal)
        self.clearance = clearance
        self.points = {(0, 0): self.origin}
        self.cost = {(0, 0): 0.0}
        self.parents = {}
        self.frontier = [(distance(pos, goal), 0.0, (0, 0))]
        self.expanded = 0

    def point(self, node):
        if node not in self.points:
            x = self.origin[0] + CELL * node[0]
            y = self.origin[1] + CELL * node[1]
            z = self.arena.ground(x, y)
            self.points[node] = (x, y, self.origin[2] if z is None else z)
        return self.points[node]

    def clear(self, start, end):
        return self.arena.walk_fraction(start, end[0] - start[0], end[1] - start[1],
                                        *self.clearance) >= 0.999

    def advance(self, zone, threats, budget=12):
        """Return a route, an empty tuple on exhaustion, or None while searching."""
        for _ in range(budget):
            if not self.frontier or self.expanded >= 600:
                return ()
            _, cost, node = heappop(self.frontier)
            if cost > self.cost[node]:
                continue
            self.expanded += 1
            point = self.point(node)
            remaining = distance(point, self.goal)
            if remaining <= 17 and self.clear(point, self.goal):
                route = [self.goal]
                while node in self.parents:
                    route.append(self.point(node))
                    node = self.parents[node]
                return tuple(reversed(route))
            for dx, dy in NEIGHBORS:
                other = node[0] + dx, node[1] + dy
                step = CELL * (math.sqrt(2) if dx and dy else 1)
                if cost + step >= self.cost.get(other, math.inf):
                    continue
                endpoint = self.point(other)
                if not self.clear(point, endpoint):
                    continue
                outside = max(0.0, distance(endpoint, zone["center"]) - zone["radius"])
                crowd = sum(max(0.0, 32 - distance(endpoint, foe["pos"])) for foe in threats)
                new_cost = cost + step * (1 + min(2.0, outside * 0.05 + crowd * 0.027))
                if new_cost < self.cost.get(other, math.inf):
                    self.cost[other] = new_cost
                    self.parents[other] = node
                    heappush(self.frontier, (new_cost + distance(endpoint, self.goal), new_cost, other))
        return None


class Navigator:
    def __init__(self, arena):
        self.arena = arena
        self.goal = None
        self.search = None
        self.route = ()
        self.best = math.inf
        self.progress_at = self.retry_at = 0.0

    def waypoint(self, obs, goal, threats):
        me, now = obs["self"], obs["t"]
        pos = me["pos"]
        if goal is None:
            self.goal, self.search, self.route = None, None, ()
            return None
        remaining = distance(pos, goal)
        if remaining < 3.5:
            self.goal, self.search, self.route = None, None, ()
            return goal
        if self.arena.bounds is None or not me["grounded"]:
            return goal
        ground = self.arena.ground(*pos[:2])
        if ground is not None and abs(pos[2] - ground) > 3.5:
            # Rooftop transitions are handled by the local controller.
            self.goal, self.search, self.route = None, None, ()
            return goal
        if self.goal is None or distance(self.goal, goal) > 10:
            self.goal, self.search, self.route = goal, None, ()
            self.best, self.progress_at, self.retry_at = remaining, now, now
        if remaining < self.best - 2.5:
            self.best, self.progress_at = remaining, now
        stats = me["stats"]
        clearance = stats["radius"], stats["height"], stats["step_up"]
        # Cheap straight-line probe before launching a search.
        if not self.route and self.search is None and now >= self.retry_at:
            scale = min(1.0, 22 / remaining)
            clear = self.arena.walk_fraction(pos, (goal[0] - pos[0]) * scale,
                                             (goal[1] - pos[1]) * scale, *clearance)
            if clear < 0.999 or now - self.progress_at > 1.35:
                self.search = Search(self.arena, pos, goal, clearance)
            else:
                self.retry_at = now + 0.35
        if self.search is not None:
            route = self.search.advance(obs["zone"], threats[:3])
            if route is not None:
                self.search, self.route, self.retry_at = None, route, now + 0.6
        while self.route and distance(pos, self.route[0]) < 3.2:
            self.route = self.route[1:]
        if self.route:
            for index in range(min(3, len(self.route)) - 1, -1, -1):
                point = self.route[index]
                if distance(pos, point) <= 22 and self.arena.walk_fraction(
                    pos, point[0] - pos[0], point[1] - pos[1], *clearance
                ) >= 0.999:
                    self.route = self.route[index:]
                    return point
            self.route = ()
            self.retry_at = now + 0.15
        return goal
