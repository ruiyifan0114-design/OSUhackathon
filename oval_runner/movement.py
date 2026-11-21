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

    def engage(self, obs, target, preferred, retreat, safe_point, finish=False):
        me = obs["self"]
        pos = me["pos"]
        tx, ty, _ = target["pos"]
        d = distance(pos, target["pos"])
        if finish:
            vx, vy, _ = target.get("vel", (0, 0, 0))
            lead = min(1.25, max(0, d - preferred) / max(1, me["stats"]["boost_speed"]))
            tx += vx * lead
            ty += vy * lead
        x, y = unit(tx - pos[0], ty - pos[1])
        radial = max(-1, min(1, (d - preferred) / 23))
        tangent = self.side * (0.89 if abs(radial) < 0.7 else 0.56)
        if finish and d > preferred + 6:
            radial, tangent = 1.0, self.side * 0.12
        elif retreat:
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

    def cover_goal(self, obs, enemies, zone_goal):
        """Find a nearby reachable firing-line break, or hold existing cover."""
        me = obs["self"]
        pos, stat, zone = me["pos"], me["stats"], obs["zone"]
        foes = [foe for foe in enemies[:2] if distance(pos, foe["pos"]) < 155]
        if not foes:
            return None
        eyes = [(foe["pos"][0], foe["pos"][1], foe["pos"][2] + 7) for foe in foes]
        height = stat["height"]
        margin = zone["radius"] - distance(pos, zone["center"])

        def exposure(point):
            chest = (point[0], point[1], point[2] + height * 0.66)
            head = (point[0], point[1], point[2] + height * 0.9)
            return sum((1.0 if index == 0 else 0.8) for index, eye in enumerate(eyes)
                       if self.arena.clear(eye, chest) or self.arena.clear(eye, head))

        current = exposure(pos)
        if not current and margin > 10 and distance(pos, zone_goal) < 27:
            return pos
        away_x, away_y = unit(pos[0] - eyes[0][0], pos[1] - eyes[0][1])
        if not (away_x or away_y):
            away_x, away_y = (1, 0)
        vx, vy, _ = me["vel"]
        now = obs["t"]
        base_ground = self.arena.ground(pos[0], pos[1])
        on_ground = base_ground is not None and pos[2] - base_ground < 1.4
        best, best_score = None, 0
        for along, across in ANGLES[:11]:
            ux, uy = away_x * along - away_y * across, away_y * along + away_x * across
            for length in (12, 24):
                dx, dy = ux * length, uy * length
                if self.arena.walk_fraction(pos, dx, dy, stat["radius"], height, stat["step_up"]) < 1.0:
                    break
                point = [pos[0] + dx, pos[1] + dy, pos[2]]
                if on_ground:
                    ground = self.arena.ground(point[0], point[1])
                    if ground is not None:
                        point[2] = ground
                room = zone["radius"] - distance(point, zone["center"])
                if room < min(2, margin):
                    continue
                next_room = zone["next_radius"] - distance(point, zone["next_center"])
                next_cost = max(0, -next_room) * max(0.03, min(0.4, (28 - (zone["shrink_end"] - now)) * 0.016))
                remaining = exposure(point)
                score = (11 * (current - remaining) - length * 0.14 - next_cost
                         - max(0, 15 - room) * 0.36 + (ux * vx + uy * vy) * 0.10)
                if score > best_score and remaining < current:
                    best, best_score = point, score
                if not remaining:
                    break
        return best

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

    @staticmethod
    def beam_margin(point, beam, height):
        origin, cos, sin, slope, reach = beam
        ox, oy, oz = (coord - source for coord, source in zip(point, origin))
        forward = ox * cos + oy * sin
        if not 0 < forward < reach or abs(oz - forward * slope) >= height * 0.58:
            return math.inf
        return abs(ox * sin - oy * cos)

    def apply(self, action, obs, goal, mode, enemies, ring=None):
        me = obs["self"]
        now = obs["t"]
        pos = me["pos"]
        vx, vy, _ = me["vel"]
        stats = me["stats"]
        zone = obs["zone"]
        threats = enemies[:2]
        hazards = self.hazards(obs)
        radius, height, step = (stats[key] for key in ("radius", "height", "step_up"))
        torso = (pos[0], pos[1], pos[2] + height * 0.58)
        beams = []
        for foe in enemies[:5]:
            if not foe.get("charging"):
                continue
            yaw, pitch = foe["aim"]
            origin = (foe["pos"][0], foe["pos"][1], foe["pos"][2] + 7)
            beam = (origin, math.cos(yaw), math.sin(yaw), math.tan(pitch),
                    self.config["weapons"]["laser"]["range"])
            if self.beam_margin(torso, beam, height) < radius + 8 and self.arena.clear(origin, torso):
                beams.append(beam)
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
        if not traveling and not hazards and not beams:
            self.last_heading = (0, 0)
            action["move"] = [0, 0]
            return
        if not traveling:
            if beams:
                dx, dy = -beams[0][2] * self.side, beams[0][1] * self.side
            else:
                dx, dy = unit(-vy, vx)
            if dx == dy == 0:
                dx, dy = (1, 0)

        energy = me["energy"]
        near = distance(pos, threats[0]["pos"]) if threats else math.inf
        lightning = next((w for w in me["weapons"] if w and w["kind"] == "lightning"), None)
        closing = lightning and mode == "fight" and lightning["stats"]["range"] + 4 < near < 95
        reserve = 32 if lightning else 22
        chasing = mode == "finish" and near > (lightning["stats"]["range"] * 0.77 if lightning else 28)
        if (traveling and mode in ("zone", "retreat") and energy > 32 or
                traveling and closing and energy > 77 or
                traveling and chasing and energy > reserve + 9 or
                traveling and mode in ("weapon", "drop") and near < 110 and goal_distance > 17 and energy > 77):
            self.boost_until = max(self.boost_until, now + 0.18)
        ground_boost = me["grounded"] and now < self.boost_until and energy > 15
        speed = stats["boost_speed"] if ground_boost else stats["walk_speed"]
        if any(w and w.get("charge", 0) > 0 for w in me["weapons"]):
            speed *= stats.get("charge_move_mult", self.config["mech"].get("charge_move_mult", 0.5))
        reach = min(14, max(3, speed * 0.76), max(3, goal_distance))
        max_score, best = -math.inf, (dx, dy, 0)
        room = zone["radius"] - distance(pos, zone["center"])

        # Outside combat the clear direct route is usually optimal.
        direct_clear = self.arena.walk_fraction(pos, dx * reach, dy * reach, radius, height, step)
        if direct_clear == 1 and traveling and not hazards and not threats and not beams and ring is None and now >= self.recover_until:
            best = dx, dy, direct_clear
        else:
            last_x, last_y = self.last_heading
            origins = [(p[0], p[1], p[2] + 8) for p in (foe["pos"] for foe in threats)]
            origins = [(origin, distance(pos, origin)) for origin in origins]
            accel = stats["ground_accel"] if me["grounded"] else stats["air_accel"]
            turn = 0.5 * accel / max(1, speed)
            # Acceleration is common to all headings. Keep only the steering-dependent term
            # inside the candidate loop.
            predictions = []
            for _, t, hit, safe, severity in hazards:
                gain = min(1.0, t * turn)
                predictions.append((hit[0] - pos[0] - t * vx * (1 - gain),
                                    hit[1] - pos[1] - t * vy * (1 - gain),
                                    t * speed * gain, safe, severity))
            beam_gain = min(1.0, 0.36 * turn)
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
                if ring:
                    center, ring_radius = ring
                    score -= 2.3 * max(0, distance(endpoint, center) - max(0, ring_radius - 1))

                for origin, before in origins:
                    after = distance(endpoint, origin)
                    if mode == "retreat" and travel > 2 and after < 135:
                        if not self.arena.clear(origin, endpoint):
                            score += 2.5
                        score += 0.11 * (after - before)
                    else:
                        score -= 0.19 * max(0, 20 - after)
                if beams:
                    # A charged shot can fire before reaching the route endpoint.
                    # Reward lateral acceleration at short and steering horizons.
                    near_point = (pos[0] + 0.36 * (vx + (ux * speed * fraction - vx) * beam_gain),
                                  pos[1] + 0.36 * (vy + (uy * speed * fraction - vy) * beam_gain), torso[2])
                    for beam in beams:
                        score -= 1.35 * max(0, radius + 2 - self.beam_margin(near_point, beam, height))
                        score -= 0.35 * max(0, radius + 5 - self.beam_margin(endpoint, beam, height))

                for x, y, gain, safe, severity in predictions:
                    advance = gain * fraction
                    margin = math.hypot(x - ux * advance, y - uy * advance) - safe
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
