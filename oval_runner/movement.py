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

    def enemy_arc_range(self, target):
        if "lightning" not in target.get("weapons", ()):
            return 0.0
        base = self.config["weapons"]["lightning"]["range"]
        # Enemy upgrades are private. Leave space for a range upgrade after
        # multiple cube offers instead of assuming the stock weapon.
        cubes = target.get("cubes", 0)
        return base * (1.35 if cubes >= 6 else 1.16 if cubes >= 3 else 1.0)

    def preferred_range(self, obs, target):
        me = obs["self"]
        secondary = me["weapons"][1]
        kind = secondary["kind"] if secondary else None
        foes = target.get("weapons", ())
        enemy_arc = self.enemy_arc_range(target)
        if kind == "lightning":
            own_arc = secondary["stats"]["range"]
            if me["energy"] >= secondary["stats"]["energy"] + 14:
                if own_arc > enemy_arc + 6 and enemy_arc:
                    return min(own_arc - 3, enemy_arc + 5)
                return own_arc * (0.73 if "laser" in foes else 0.84)
        if enemy_arc:
            return max(enemy_arc + 9, 82 if kind == "laser" else 0)
        if kind == "laser":
            return 92 if "mortar" in foes else 75
        if kind == "mortar":
            return 55 if "laser" in foes else 49
        return 55 if "laser" in foes else 63

    def engage(self, obs, target, preferred, retreat, safe_point, finish=False):
        me = obs["self"]
        pos = me["pos"]
        tx, ty, _ = target["pos"]
        d = distance(pos, target["pos"])
        vx, vy, _ = target.get("vel", (0, 0, 0))
        lead = 0.34
        if finish:
            lead = min(1.25, max(lead, (d - preferred) / max(1, me["stats"]["boost_speed"])))
        tx += vx * lead
        ty += vy * lead
        x, y = unit(tx - pos[0], ty - pos[1])
        predicted_range = math.hypot(tx - pos[0], ty - pos[1])
        radial = max(-1, min(1, (predicted_range - preferred) / 19))
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
            # Keep threatening splash shells in the small candidate set even
            # when several bullets have a smaller absolute miss distance.
            priority = miss / reach + 0.75 * until + (0.0 if shell else 0.55)
            danger.append((priority, until, impact, reach,
                           6.8 if shell else 2.7, (sx, sy)))
        return sorted(danger)[:8]

    @staticmethod
    def beam_margin(point, beam, height):
        origin, cos, sin, slope, reach = beam
        ox, oy, oz = (coord - source for coord, source in zip(point, origin))
        forward = ox * cos + oy * sin
        if not 0 < forward < reach or abs(oz - forward * slope) >= height * 0.58:
            return math.inf
        return abs(ox * sin - oy * cos)

    def apply(self, action, obs, goal, mode, enemies, ring=None, target=None, alert=None):
        me = obs["self"]
        now = obs["t"]
        pos = me["pos"]
        vx, vy, _ = me["vel"]
        stats = me["stats"]
        zone = obs["zone"]
        threats = enemies[:2]
        attack = target if mode in ("fight", "finish") else None
        secondary = me["weapons"][1]
        lightning = secondary if secondary and secondary["kind"] == "lightning" else None
        own_arc = lightning["stats"]["range"] if lightning else 0.0
        arc_active = bool(lightning and me["energy"] > lightning["stats"]["energy"] + 12)
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
            elif hazards:
                # Move across the projectile's flight path rather than choosing
                # an arbitrary direction.  If the predicted impact is offset,
                # blend in the radial escape direction as well.
                _, _, hit, _, _, shot_vel = hazards[0]
                sx, sy = shot_vel
                px, py = -sy, sx
                if px == py == 0:
                    px, py = -vy, vx
                rx, ry = unit(pos[0] - hit[0], pos[1] - hit[1])
                if px * rx + py * ry < 0:
                    px, py = -px, -py
                dx, dy = unit(px + 0.6 * rx, py + 0.6 * ry)
            else:
                dx, dy = unit(-vy, vx)
            if dx == dy == 0:
                dx, dy = (1, 0)

        energy = me["energy"]
        near = distance(pos, threats[0]["pos"]) if threats else math.inf
        engagement_distance = distance(pos, attack["pos"]) if attack else near
        arc_opening = self.enemy_arc_range(attack) if attack else 0
        closing = arc_active and mode == "fight" and own_arc + 3 < engagement_distance < 85
        closing_energy = 68 if arc_opening else 52
        reserve = 32 if lightning else 22
        chasing = mode == "finish" and engagement_distance > (own_arc * 0.77 if arc_active else 28)
        if (traveling and mode in ("zone", "retreat", "caution", "laser_break") and energy > 32 or
                traveling and closing and energy > closing_energy or
                traveling and chasing and energy > reserve + 9 or
                traveling and mode in ("weapon", "drop") and near < 110 and goal_distance > 17 and energy > 77):
            self.boost_until = max(self.boost_until, now + 0.18)
        ground_boost = me["grounded"] and now < self.boost_until and energy > 15
        speed = stats["boost_speed"] if ground_boost else stats["walk_speed"]
        fire = action.get("fire", (False, False))
        if any(w and w["kind"] == "laser" and (w.get("charge", 0) > 0 or trigger)
               for w, trigger in zip(me["weapons"], fire)):
            speed *= stats.get("charge_move_mult", self.config["mech"].get("charge_move_mult", 0.5))
        reach = min(14, max(3, speed * 0.76), max(3, goal_distance))
        max_score, best = -math.inf, (dx, dy, 0)
        room = zone["radius"] - distance(pos, zone["center"])

        # Outside combat the clear direct route is usually optimal.
        direct_clear = self.arena.walk_fraction(pos, dx * reach, dy * reach, radius, height, step)
        alert_near = bool(alert is not None and alert.age(now) < 1.8 and alert.distance < 120)
        if (direct_clear == 1 and traveling and not hazards and not threats and not beams
                and ring is None and not alert_near and now >= self.recover_until):
            best = dx, dy, direct_clear
        else:
            last_x, last_y = self.last_heading
            origins = []
            for foe in threats:
                px, py, pz = foe["pos"]
                origin = (px, py, pz + 8)
                hostile_arc = self.enemy_arc_range(foe)
                shared_duel = attack is not None and foe["id"] == attack["id"]
                if hostile_arc and shared_duel and arc_active and own_arc <= hostile_arc + 5:
                    # Closing is worthwhile when both mechs have similar arc
                    # range. The aim layer can still fire while we strafe.
                    hostile_arc = min(hostile_arc, 22)
                origins.append((origin, distance(pos, origin), hostile_arc))
            if attack:
                ax, ay, az = attack["pos"]
                avx, avy, avz = attack.get("vel", (0, 0, 0))
                aim_point = (ax + 0.35 * avx, ay + 0.35 * avy, az + 4.8 + 0.16 * avz)
                gun = me["weapons"][0]
                ready_gun = gun and gun["kind"] == "gun" and gun.get("reload", 0) < 0.45
                straight_shot = ready_gun or arc_active or secondary and secondary["kind"] == "laser"
                attack_reach = max(128 if ready_gun else 0, own_arc if arc_active else 0,
                                   175 if secondary and secondary["kind"] == "laser" else 0)
                yaw = attack["aim"][0]
                attack_facing = math.cos(yaw), math.sin(yaw)
            accel = stats["ground_accel"] if me["grounded"] else stats["air_accel"]
            turn = 0.5 * accel / max(1, speed)
            # Acceleration is common to all headings. Keep only the steering-dependent term
            # inside the candidate loop.
            predictions = []
            for _, t, hit, safe, severity, _ in hazards:
                gain = min(1.0, t * turn)
                predictions.append((hit[0] - pos[0] - t * vx * (1 - gain),
                                    hit[1] - pos[1] - t * vy * (1 - gain),
                                    t, safe, severity))
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
                if alert_near:
                    toward = ux * math.cos(alert.bearing) + uy * math.sin(alert.bearing)
                    score -= (1.05 if alert.distance < 45 else 0.55) * max(0.0, toward)
                if room < 38:
                    future_room = zone["radius"] - distance(endpoint, zone["center"])
                    score -= max(0, 14 - future_room) * (0.36 if room >= 0 else 0.52)
                if ring:
                    center, ring_radius = ring
                    score -= 2.3 * max(0, distance(endpoint, center) - max(0, ring_radius - 1))

                for origin, before, hostile_arc in origins:
                    after = distance(endpoint, origin)
                    if mode in ("retreat", "laser_break") and travel > 2 and after < 135:
                        if not self.arena.clear(origin, endpoint):
                            score += 2.5
                        score += 0.11 * (after - before)
                    else:
                        score -= 0.19 * max(0, 20 - after)
                    if attack or mode in ("retreat", "laser_break"):
                        score -= 0.17 * max(0, hostile_arc + 3 - after) if hostile_arc else 0
                if attack:
                    attack_distance = distance(endpoint, aim_point)
                    if arc_active and engagement_distance < own_arc + 12:
                        score -= 0.25 * max(0, attack_distance - own_arc + 2)
                    if straight_shot and attack_distance < attack_reach and travel > 2:
                        muzzle = (endpoint[0], endpoint[1], endpoint[2] + stats["muzzle_height"] - height * 0.58)
                        if self.arena.clear(muzzle, aim_point):
                            side = ((muzzle[0] - aim_point[0]) * attack_facing[0] +
                                    (muzzle[1] - aim_point[1]) * attack_facing[1])
                            score += (2.2 if mode == "finish" else 1.65) + 0.5 * (side < -7)
                if beams:
                    # A charged shot can fire before reaching the route endpoint.
                    # Reward lateral acceleration at short and steering horizons.
                    near_point = (pos[0] + 0.36 * (vx + (ux * speed * fraction - vx) * beam_gain),
                                  pos[1] + 0.36 * (vy + (uy * speed * fraction - vy) * beam_gain), torso[2])
                    for beam in beams:
                        score -= 1.35 * max(0, radius + 2 - self.beam_margin(near_point, beam, height))
                        score -= 0.35 * max(0, radius + 5 - self.beam_margin(endpoint, beam, height))

                for x, y, t, safe, severity in predictions:
                    # The candidate can displace at most speed * t (and only
                    # as far as the clear fraction allows).  This used to be a
                    # dimensionless gain*fraction, which capped the dodge at
                    # one metre regardless of speed or warning time.
                    available = min(t, reach / max(1.0, speed)) * speed * fraction
                    margin = math.hypot(x - ux * available, y - uy * available) - safe
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
