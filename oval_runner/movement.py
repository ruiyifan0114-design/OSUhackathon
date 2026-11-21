"""Bounded local steering, cover selection, and projectile avoidance."""
import math


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(x, y):
    length = math.hypot(x, y)
    return (x / length, y / length) if length > 1e-6 else (0.0, 0.0)


ANGLES = tuple((math.cos(a), math.sin(a)) for a in
               (0, 0.36, -0.36, 0.74, -0.74, 1.17, -1.17, 1.57, -1.57, 2.18, -2.18, math.pi))


class Movement:
    def __init__(self, hello, arena):
        self.arena = arena
        self.config = hello["config"]
        self.side = 1 if hello["slot"] % 2 else -1
        self.anchor = None
        self.anchor_t = -1
        self.last_heading = (0, 0)
        self.was_traveling = False
        self.recover_until = 0
        self.boost_until = 0
        self.jet_until = 0

    def engage(self, obs, target, preferred, retreat, safe_point):
        pos = obs["self"]["pos"]
        x, y = unit(target["pos"][0] - pos[0], target["pos"][1] - pos[1])
        d = distance(pos, target["pos"])
        radial = max(-1, min(1, (d - preferred) / 23))
        tangent = self.side * (0.89 if abs(radial) < 0.7 else 0.56)
        if retreat:
            radial, tangent = -1.25, self.side * 0.6
        if target.get("charging") and not retreat:
            tangent *= 1.65
        dx, dy = x * radial - y * tangent, y * radial + x * tangent
        zone = obs["zone"]
        margin = zone["radius"] - distance(pos, zone["center"])
        if margin < 45 or distance(pos, safe_point) > 35:
            safe_x, safe_y = unit(safe_point[0] - pos[0], safe_point[1] - pos[1])
            weight = 1.5 if margin < 30 else 0.33
            dx += safe_x * weight
            dy += safe_y * weight
        return pos[0] + dx * 27, pos[1] + dy * 27, pos[2]

    def hazards(self, obs):
        me = obs["self"]
        x, y, z = me["pos"]
        vx, vy, _ = me["vel"]
        danger = []
        mortar = self.config["weapons"]["mortar"]
        for shot in obs["projectiles"]:
            px, py, pz = shot["pos"]
            sx, sy, sz = shot["vel"]
            lx, ly = sx - vx, sy - vy
            speed2 = lx * lx + ly * ly
            if speed2 < 1:
                continue
            tx, ty = px - x, py - y
            until = -(lx * tx + ly * ty) / speed2
            shell = shot["kind"] == "mortar"
            horizon = 1.1 if shell else 0.50
            if not -0.04 <= until <= horizon:
                continue
            until = max(0.035, until)
            impact = (px + sx * until, py + sy * until,
                      pz + sz * until - (0.5 * mortar["gravity"] * until * until if shell else 0))
            miss = math.hypot(impact[0] - x - vx * until, impact[1] - y - vy * until)
            reach = me["stats"]["radius"] + (mortar["splash_radius"] if shell else 0.8)
            if miss > reach + until * 20 or not z - reach - 3 < impact[2] < z + me["stats"]["height"] + reach + 3:
                continue
            danger.append((miss + until * 2, until, impact, reach, 6.0 if shell else 2.7))
        return sorted(danger)[:8]

    def apply(self, action, obs, goal, mode, enemies):
        me = obs["self"]
        now = obs["t"]
        pos = me["pos"]
        vx, vy, _ = me["vel"]
        stats = me["stats"]
        zone = obs["zone"]
        threats = enemies[:2]
        hazards = self.hazards(obs)
        goal_distance = distance(pos, goal) if goal is not None else 0
        stop_radius = {"farm": 18, "drop": 2.5, "cube": 1.8, "weapon": 2.5}.get(mode, 1.4)
        if mode == "farm" and goal is not None:
            eye = (pos[0], pos[1], pos[2] + stats["muzzle_height"])
            if not self.arena.clear(eye, (goal[0], goal[1], goal[2] + 1.25)):
                stop_radius = 2
        traveling = goal_distance > stop_radius
        dx, dy = unit(goal[0] - pos[0], goal[1] - pos[1]) if traveling else (0, 0)

        if self.anchor is None or now - self.anchor_t >= 0.9:
            if self.anchor and self.was_traveling and traveling and distance(pos, self.anchor) < 2.4:
                self.recover_until = now + 1.65
                self.side *= -1
            self.anchor = pos
            self.anchor_t = now
            self.was_traveling = traveling
        if not traveling and not hazards:
            self.last_heading = (0, 0)
            action["move"] = [0, 0]
            return
        if not traveling:
            dx, dy = unit(-vy, vx)
            if dx == dy == 0:
                dx, dy = (1, 0)

        energy = me["energy"]
        near = distance(pos, threats[0]["pos"]) if threats else math.inf
        lightning = next((w for w in me["weapons"] if w and w["kind"] == "lightning"), None)
        closing = lightning and mode == "fight" and lightning["stats"]["range"] + 4 < near < 95
        if (traveling and mode in ("zone", "retreat") and energy > 32 or
                traveling and closing and energy > 77 or
                traveling and mode in ("weapon", "drop") and near < 110 and goal_distance > 17 and energy > 77):
            self.boost_until = max(self.boost_until, now + 0.18)
        ground_boost = me["grounded"] and now < self.boost_until and energy > 15
        speed = stats["boost_speed"] if ground_boost else stats["walk_speed"]
        reach = min(14, max(3, speed * 0.76), max(3, goal_distance))
        radius, height, step = (stats[key] for key in ("radius", "height", "step_up"))
        max_score, best = -math.inf, (dx, dy, 0)
        room = zone["radius"] - distance(pos, zone["center"])

        # Outside combat the clear direct route is usually optimal.
        direct_clear = self.arena.walk_fraction(pos, dx * reach, dy * reach, radius, height, step)
        if direct_clear == 1 and traveling and not hazards and not threats and now >= self.recover_until:
            best = dx, dy, direct_clear
        else:
            last_x, last_y = self.last_heading
            origins = [
                (foe, (foe["pos"][0], foe["pos"][1], foe["pos"][2] + 8))
                for foe in threats
            ]
            for along, across in ANGLES:
                ux, uy = dx * along - dy * across, dy * along + dx * across
                fraction = direct_clear if across == 0 and along == 1 else self.arena.walk_fraction(
                    pos, ux * reach, uy * reach, radius, height, step
                )
                travel = reach * fraction
                endpoint = (pos[0] + ux * travel, pos[1] + uy * travel, pos[2] + height * 0.58)
                score = ((2.35 if traveling else 0.15) * along * fraction + 3.55 * fraction
                         + 0.64 * (ux * last_x + uy * last_y) + 0.08 * across * self.side)
                if room < 38:
                    future_room = zone["radius"] - distance(endpoint, zone["center"])
                    score -= max(0, 14 - future_room) * (0.36 if room >= 0 else 0.52)

                for foe, origin in origins:
                    before, after = distance(pos, origin), distance(endpoint, origin)
                    if mode == "retreat" and travel > 2 and after < 135:
                        if not self.arena.clear(origin, endpoint):
                            score += 2.5
                        score += 0.11 * (after - before)
                    else:
                        score -= 0.19 * max(0, 20 - after)
                    if foe.get("charging") and travel > 2 and after < 180:
                        yaw, pitch = foe["aim"]
                        along_ray = (endpoint[0] - origin[0]) * math.cos(yaw) + (endpoint[1] - origin[1]) * math.sin(yaw)
                        if along_ray > 0:
                            lateral = abs((endpoint[0] - origin[0]) * math.sin(yaw) -
                                          (endpoint[1] - origin[1]) * math.cos(yaw))
                            vertical = abs(endpoint[2] - origin[2] - along_ray * math.tan(pitch))
                            score -= 0.44 * max(0, 9 - lateral) if vertical < height else 0

                for _, t, hit, safe, severity in hazards:
                    # Approximate acceleration over the projectile's remaining flight.
                    accel = stats["ground_accel"] if me["grounded"] else stats["air_accel"]
                    change = min(1.0, 0.5 * t * accel / max(1, speed))
                    sx = pos[0] + t * (vx + (ux * speed * fraction - vx) * change)
                    sy = pos[1] + t * (vy + (uy * speed * fraction - vy) * change)
                    margin = math.hypot(hit[0] - sx, hit[1] - sy) - safe
                    score -= severity * max(0, 1 - margin / max(1, safe))
                if score > max_score:
                    max_score, best = score, (ux, uy, fraction)

        ux, uy, fraction = best
        stalled = traveling and (fraction < 0.31 or now < self.recover_until and fraction < 0.65)
        if stalled and me["grounded"] and energy > 30:
            action["jump"] = True
            self.jet_until = now + 0.85
        if stalled and fraction < 0.18:
            ux, uy = dx, dy
        self.last_heading = (ux, uy)
        action["move"] = [ux, uy]
        action["boost"] = bool(ground_boost or not me["grounded"] and now < self.jet_until and energy > 18)
