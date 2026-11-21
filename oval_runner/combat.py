"""Aim and weapon timing for the mechfight/1 simulator."""

from dataclasses import dataclass, replace
import math


def angle_error(first, second):
    return math.remainder(first - second, math.tau)


@dataclass(slots=True)
class Shot:
    yaw: float
    pitch: float
    distance: float
    horizontal_tolerance: float
    vertical_tolerance: float
    flight: float = 0.0
    impact: tuple | None = None
    splash: float | None = None
    collateral: float = 0.0


@dataclass(slots=True, frozen=True)
class Engagement:
    """Predicted first damage, lethal damage, and expected damage within 1 s."""

    first_shot: float = math.inf
    kill_time: float = math.inf
    damage_1s: float = 0.0
    interrupt: float = 0.0


class Combat:
    def __init__(self, hello, arena=None):
        self.arena = arena
        self.config = hello["config"]
        match = self.config["match"]
        self.step = 1 / match["tick_hz"]
        self.steps = max(1, int(match["decide_every"]))
        self.frame = self.step * self.steps
        self.laser_wait = 0
        self.observation = None
        self.solutions = {}
        self.arc_lanes = {}
        self.arc_results = {}

    def dimensions(self, target):
        if target.get("kind") == "mech" or target["id"] < 16:
            stats = self.config["mech"]
            scale = 1 + self.config["cubes"]["size_per_cube"] * target.get("cubes", 0)
            return stats["radius"] * scale, stats["height"] * scale
        group = "npc" if target.get("kind") == "npc" or target["id"] < 1000 else "crates"
        stats = self.config[group]
        return stats["radius"], stats["height"]

    def _lead(self, dx, dy, dz, velocity, speed, offset, gravity, ttl):
        """Solve the first ballistic intercept, including the muzzle offset."""
        flight = max(0, (math.sqrt(dx * dx + dy * dy + dz * dz) - offset) / speed)
        if not gravity and not any(velocity):
            return flight
        vx, vy, vz = velocity
        for _ in range(7):
            x = dx + vx * flight
            y = dy + vy * flight
            z = dz + (vz + 0.5 * gravity * flight) * flight
            length = math.sqrt(x * x + y * y + z * z)
            miss = length - (speed * flight + offset)
            if abs(miss) < 0.04:
                return flight
            derivative = (x * vx + y * vy + z * (vz + gravity * flight)) / max(length, 0.01) - speed
            if derivative > -0.05 * speed:
                return None
            flight = max(0, flight - miss / derivative)
            if flight > ttl:
                return None
        return None

    def _trajectory_clear(self, start, end, flight, gravity):
        if self.arena is None:
            return True
        if not gravity or flight < 0.25:
            return self.arena.clear(start, end)
        # Check short chords, with their endpoints on the simulated parabola.
        chords = min(10, max(2, math.ceil(flight * 3.5)))
        dx, dy, dz = (end[i] - start[i] for i in range(3))
        bend = 0.5 * gravity * flight * flight
        previous = start
        for i in range(1, chords + 1):
            fraction = i / chords
            point = (
                start[0] + dx * fraction,
                start[1] + dy * fraction,
                start[2] + (dz + bend * (1 - fraction)) * fraction,
            )
            if not self.arena.clear(previous, point):
                return False
            previous = point
        return True

    def _solution(self, me, target, weapon, latency, aim_at):
        stats = weapon["stats"]
        kind = weapon["kind"]
        own = me["stats"]
        radius, height = self.dimensions(target)
        p = me["pos"]
        own_v = me["vel"]
        muzzle = own["height"] + 0.5 if kind == "mortar" else own["muzzle_height"]
        origin = (p[0] + own_v[0] * latency, p[1] + own_v[1] * latency,
                  p[2] + own_v[2] * latency + muzzle)
        target_pos = target["pos"]
        velocity = target.get("vel", (0, 0, 0))
        dx = target_pos[0] + velocity[0] * latency - origin[0]
        dy = target_pos[1] + velocity[1] * latency - origin[1]
        dz = target_pos[2] + velocity[2] * latency + height * aim_at - origin[2]
        offset = own["radius"] + 0.5
        gravity = stats.get("gravity", 0)
        flight = 0.0
        projectile_speed = stats.get("speed", 0)
        # Skip obviously out-of-range trajectories before solving or ray tracing.
        if projectile_speed:
            ttl = stats["ttl"] if gravity else stats["range"] / projectile_speed
            flight = self._lead(dx, dy, dz, velocity, projectile_speed, offset, gravity, ttl)
            if flight is None or flight > ttl:
                return None
        ex, ey, ez = dx + velocity[0] * flight, dy + velocity[1] * flight, dz + velocity[2] * flight
        horiz = math.hypot(ex, ey)
        distance = math.hypot(horiz, ez)
        if kind == "lightning" and distance > stats["range"]:
            return None
        if not projectile_speed and kind != "lightning" and distance > stats["range"] + offset:
            return None
        if gravity and horiz < own["radius"] + stats["splash_radius"] + 2:
            return None
        arc_rise = 0.5 * gravity * flight * flight
        pitch = math.atan2(ez + arc_rise, horiz)
        if not own["min_pitch"] <= pitch <= own["max_pitch"] and kind != "lightning":
            return None
        pitch = min(own["max_pitch"], max(own["min_pitch"], pitch))
        endpoint = (origin[0] + ex, origin[1] + ey, origin[2] + ez)
        if gravity:
            # The shell spawns offset along the ray and only then starts falling.
            arc_length = math.hypot(horiz, ez + arc_rise)
            fraction = min(1.0, offset / max(arc_length, 0.01))
            start = (origin[0] + ex * fraction, origin[1] + ey * fraction,
                     origin[2] + (ez + arc_rise) * fraction)
            if self.arena is not None and not self.arena.clear(origin, start):
                return None
        else:
            start = origin
        if not self._trajectory_clear(start, endpoint, flight, gravity):
            return None
        horizontal = math.atan2(radius * 0.90, max(0.1, horiz))
        vertical = math.atan2(height * min(aim_at, 1 - aim_at) * 0.9, max(0.1, horiz))
        if kind == "lightning":
            horizontal = math.radians(stats["cone_deg"]) * 0.49
            vertical = math.pi
        return Shot(math.atan2(ey, ex), pitch, distance, horizontal, vertical, flight, endpoint)

    def _cluster_mortar(self, obs, target, weapon, latency, direct):
        """Try one ground shot between two nearby mechs using predicted positions."""
        if target.get("kind") != "mech" or self.arena is None or weapon["cooldown"] > 0.25:
            return direct
        stats = weapon["stats"]
        splash_range = stats["splash_radius"]
        x, y, z = target["pos"]
        neighbors = []
        for foe in obs["mechs"]:
            if foe["id"] == target["id"] or foe.get("kind") != "mech":
                continue
            fx, fy, fz = foe["pos"]
            d = math.hypot(fx - x, fy - y)
            if abs(fz - z) <= 3 and 3 <= d <= splash_range * 1.55:
                neighbors.append((d, foe["id"], foe))
        if not neighbors:
            return direct
        _, _, other = min(neighbors)
        point = tuple((a + b) * 0.5 for a, b in zip(target["pos"], other["pos"]))
        ground = self.arena.ground(*point[:2])
        if ground is None or abs(point[2] - ground) > 1.2:
            return direct
        velocity = tuple((a + b) * 0.5 for a, b in zip(target["vel"], other["vel"]))
        virtual = {"id": target["id"], "kind": "mech", "pos": (*point[:2], ground + 0.20),
                   "vel": (*velocity[:2], 0)}
        shot = self._solution(obs["self"], virtual, weapon, latency, 0)
        if shot is None:
            return direct
        landing = self.arena.ground(*shot.impact[:2])
        if landing is None or abs(landing - shot.impact[2]) > 0.6:
            return direct

        def factor(foe):
            travel = shot.flight + latency
            feet = tuple(p + v * travel for p, v in zip(foe["pos"], foe["vel"]))
            separation = math.dist(shot.impact, (feet[0], feet[1], feet[2] + 0.8))
            if separation >= splash_range or not self.arena.clear(
                    (shot.impact[0], shot.impact[1], landing + 0.4), (feet[0], feet[1], feet[2] + 1.5)):
                return 0.0
            return 1 - (1 - stats["splash_edge"]) * separation / splash_range

        primary, extra = factor(target), factor(other)
        if primary <= 0 or extra <= 0:
            return direct
        tolerance = math.atan2(1.3, max(1.0, shot.distance))
        cluster = replace(shot, horizontal_tolerance=tolerance, vertical_tolerance=tolerance,
                          splash=primary, collateral=stats["splash"] * extra * 0.5)
        cluster_damage, _ = self._shot_damage(obs["self"], target, weapon, cluster)
        if direct is None:
            return cluster
        direct_damage, _ = self._shot_damage(obs["self"], target, weapon, direct)
        # Allow a cluster shot to beat a long, uncertain direct shot, but do
        # not give up an immediate direct finish on the selected target.
        hp = target.get("hp") or target.get("max_hp") or 1400
        if direct_damage < hp and cluster_damage + cluster.collateral > direct_damage:
            return cluster
        return direct

    def _shot(self, me, target, weapon, latency):
        shot = self._solution(me, target, weapon, latency, 0.52)
        if shot is None and weapon["kind"] in ("gun", "laser"):
            shot = self._solution(me, target, weapon, latency, 0.83)
        return shot

    def _begin_frame(self, obs):
        if obs is not self.observation:
            self.observation = obs
            self.solutions.clear()
            self.arc_lanes.clear()
            self.arc_results.clear()

    def _frame_shot(self, obs, target, weapon):
        # Target ranking and execution use exactly the same firing lane.
        self._begin_frame(obs)
        key = target["id"], weapon["kind"]
        if key not in self.solutions:
            cooldown = weapon.get("cooldown", 0)
            latency = self.step * 0.55 + min(self.step, max(0, cooldown - self.step))
            shot = self._shot(obs["self"], target, weapon, latency)
            if weapon["kind"] == "mortar":
                shot = self._cluster_mortar(obs, target, weapon, latency, shot)
            self.solutions[key] = shot
        return self.solutions[key]

    def _heading_hits(self, me, aimed, shot, delay=0):
        """Check the aim after the first simulation step on which a shot can fire."""
        yaw, pitch = self._slewed_aim(me, aimed, delay)
        return (
            abs(angle_error(yaw, shot.yaw)) <= shot.horizontal_tolerance
            and abs(pitch - shot.pitch) <= shot.vertical_tolerance
        )

    def _slewed_aim(self, me, aimed, delay):
        steps = min(self.steps, max(1, math.ceil((delay - 1e-6) / self.step)))
        elapsed = steps * self.step
        own_yaw, own_pitch = me["aim"]
        yaw_delta = angle_error(aimed.yaw, own_yaw)
        pitch_delta = aimed.pitch - own_pitch
        yaw_step = me["stats"]["turn_rate"] * elapsed
        pitch_step = me["stats"]["pitch_rate"] * elapsed
        yaw = own_yaw + min(yaw_step, max(-yaw_step, yaw_delta))
        pitch = own_pitch + min(pitch_step, max(-pitch_step, pitch_delta))
        return yaw, pitch

    def _arc_clear(self, start, end):
        key = start, end
        if key not in self.arc_lanes:
            self.arc_lanes[key] = self.arena is None or self.arena.clear(start, end)
        return self.arc_lanes[key]

    def arc_hits(self, obs, weapon, yaw):
        """Resolve the nearest legal first hit and nearest visible bounces.

        Uses observed positions. The server may also know actors hidden from us.
        """
        self._begin_frame(obs)
        key = id(weapon), yaw
        if key in self.arc_results:
            return self.arc_results[key]
        me, stats = obs["self"], weapon["stats"]
        x, y, z = me["pos"]
        origin = (x, y, z + me["stats"]["muzzle_height"])
        reach = stats["range"] + stats["chain"] * stats["chain_range"]
        cone = math.radians(stats["cone_deg"]) * 0.5
        pool = []
        for actor in (*obs["mechs"], *obs["crates"]):
            if actor["id"] == me["id"] or actor.get("hp") == 0:
                continue
            ax, ay, az = actor["pos"]
            if math.hypot(ax - x, ay - y) > reach:
                continue
            _, height = self.dimensions(actor)
            point = (ax, ay, az + height * 0.52)
            pool.append((actor["id"], actor, point))
        hits = []
        previous = origin
        for jump in range(int(stats["chain"]) + 1):
            limit = stats["chain_range"] if jump else stats["range"]
            candidates = []
            for identity, actor, point in pool:
                if not jump and abs(angle_error(math.atan2(point[1] - y, point[0] - x), yaw)) > cone:
                    continue
                d = math.dist(previous, point)
                if d <= limit:
                    candidates.append((d, identity, actor, point))
            chosen = next((entry for entry in sorted(candidates)
                           if self._arc_clear(previous, entry[3])), None)
            if chosen is None:
                break
            _, identity, actor, point = chosen
            incoming = math.atan2(point[1] - previous[1], point[0] - previous[0])
            hits.append((actor, stats["chain_falloff"] ** jump, incoming))
            pool = [entry for entry in pool if entry[0] != identity]
            previous = point
        self.arc_results[key] = hits
        return hits

    def _arc_value(self, obs, target, weapon, shot, yaw):
        expected = connected = utility = interrupt = 0.0
        for actor, falloff, incoming in self.arc_hits(obs, weapon, yaw):
            arc = Shot(incoming, shot.pitch, shot.distance, 0, 0)
            mean, direct = self._shot_damage(obs["self"], actor, weapon, arc)
            if actor["id"] == target["id"]:
                expected, connected = mean * falloff, direct * falloff
            elif actor.get("kind") == "mech":
                utility += mean * falloff * 0.30
            if actor.get("charging") and "laser" in actor.get("weapons", ()):
                interrupt += 48
        return expected, connected, utility, interrupt

    def _turn_time(self, me, shot):
        yaw, pitch = me["aim"]
        stats = me["stats"]
        return max(
            max(0.0, abs(angle_error(shot.yaw, yaw)) - shot.horizontal_tolerance) / stats["turn_rate"],
            max(0.0, abs(shot.pitch - pitch) - shot.vertical_tolerance) / stats["pitch_rate"],
        )

    def _turn_between(self, me, previous, shot):
        stats = me["stats"]
        return max(
            max(0.0, abs(angle_error(shot.yaw, previous.yaw)) - shot.horizontal_tolerance) / stats["turn_rate"],
            max(0.0, abs(shot.pitch - previous.pitch) - shot.vertical_tolerance) / stats["pitch_rate"],
        )

    def _shared_aim(self, first, second):
        """Find a command in the intersection of both useful aim windows."""
        second_yaw = first.yaw + angle_error(second.yaw, first.yaw)
        low = max(first.yaw - first.horizontal_tolerance, second_yaw - second.horizontal_tolerance)
        high = min(first.yaw + first.horizontal_tolerance, second_yaw + second.horizontal_tolerance)
        bottom = max(first.pitch - first.vertical_tolerance, second.pitch - second.vertical_tolerance)
        top = min(first.pitch + first.vertical_tolerance, second.pitch + second.vertical_tolerance)
        if low >= high or bottom >= top:
            return None
        return Shot(math.remainder(0.5 * (low + high), math.tau), 0.5 * (bottom + top),
                    first.distance, 0.0, 0.0)

    def _rear_bonus(self, target, yaw, flight):
        """Discount rear armor near the edge or when the target has time to turn."""
        if target.get("kind") != "mech" or "aim" not in target:
            return 1.0
        stats = self.config["mech"]
        rear_margin = abs(angle_error(yaw + math.pi, target["aim"][0])) - math.radians(stats["front_arc_deg"]) / 2
        confidence = min(1.0, max(0.0, (rear_margin - 0.10) / (0.35 + flight * 2)))
        return 1 + (stats["rear_mult"] - 1) * confidence

    def _shot_damage(self, me, target, weapon, shot, charge_fraction=1.0):
        """Return expected damage and damage conditional on a direct hit."""
        stats = weapon["stats"]
        kind = weapon["kind"]
        damage = stats["damage"]
        accuracy = 1.0
        cubes = self.config["cubes"]
        damage *= 1 + cubes["damage_per_cube"] * min(me["cubes"], cubes["stat_cap"])
        if kind == "laser":
            minimum = stats["min_frac"]
            damage *= minimum + (1 - minimum) * charge_fraction
            if "mortar" in target.get("weapons", ()):
                damage *= 1 + stats["cookoff"]
        elif kind == "gun":
            # Forecast the part of the spread cone inside the target width.
            spread = max(0.001, math.radians(stats["spread_deg"]))
            accuracy = math.erf(shot.horizontal_tolerance / (math.sqrt(2) * spread))
        elif kind == "mortar":
            if shot.splash is None:
                damage += stats["splash"] * 0.70
            else:
                damage *= stats["splash"] * shot.splash / stats["damage"]
        if target.get("kind") == "mech":
            # A longer flight gives the opponent time to invalidate its current lead.
            vx, vy, _ = target.get("vel", (0, 0, 0))
            accuracy /= 1 + 0.55 * shot.flight * (1 + 0.2 * math.hypot(vx, vy))
            rear = self._rear_bonus(target, shot.yaw, shot.flight)
            # The mortar's blast approaches from its impact point.
            damage *= 0.94 * (rear if kind != "mortar" else 1 + 0.5 * (rear - 1))
        return damage * accuracy, damage

    def _ready_delay(self, me, weapon):
        """Earliest trigger without changing the aim, excluding laser charge."""
        kind = weapon["kind"]
        cooldown = weapon.get("cooldown", 0)
        if kind == "gun":
            reload = weapon.get("reload", 0)
            if weapon.get("ammo", 0) <= 0 and reload <= 0:
                reload = weapon["stats"]["reload"]
            return max(cooldown, reload)
        if kind == "lightning":
            reserve = 5 if me["grounded"] else 14
            if me["energy"] < weapon["stats"]["energy"] + reserve:
                return math.inf
        return cooldown

    def estimate(self, obs, target):
        """Estimate burst and time to kill if current firing lanes persist.

        The horizon is bounded. No reachable shot or kill within it returns
        infinity. This is a tactical comparison, not a movement prediction.
        """
        me = obs["self"]
        schedules = []
        interrupt = 0.0
        horizon = 8.0
        for weapon in me["weapons"]:
            if weapon is None:
                continue
            kind, stats = weapon["kind"], weapon["stats"]
            delay = self._ready_delay(me, weapon)
            if delay >= horizon:
                continue
            shot = self._frame_shot(obs, target, weapon)
            if shot is None:
                continue
            turn = self._turn_time(me, shot)
            damage, _ = self._shot_damage(me, target, weapon, shot)
            interval = max(self.step, stats["cooldown"])
            ammo = weapon.get("ammo", 0)
            limit = 40
            opening_damage = damage
            if kind == "laser":
                if "lightning" in target.get("weapons", ()) and shot.distance < 48 and not weapon.get("charge", 0):
                    continue
                goal = self._charge_goal(me, target, weapon)
                charge = weapon.get("charge", 0)
                delay += max(0.0, goal - charge)
                opening_damage, _ = self._shot_damage(
                    me, target, weapon, shot, min(1, max(goal, charge) / stats["charge"])
                )
                interval += stats["charge"]
            elif kind == "gun" and (ammo <= 0 or weapon.get("reload", 0) > 0):
                ammo = int(stats["mag"])
            elif kind == "lightning":
                damage, _, _, interrupt = self._arc_value(obs, target, weapon, shot, shot.yaw)
                opening_damage = damage
                if damage <= 0:
                    continue
                reserve = 5 if me["grounded"] else 14
                limit = min(limit, max(0, int((me["energy"] - reserve) / max(0.01, stats["energy"]))))
            trigger = max(self.step, turn, delay)
            events = []
            for index in range(limit):
                if trigger + shot.flight > horizon:
                    break
                events.append((trigger, opening_damage if index == 0 else damage))
                if kind == "gun":
                    ammo -= 1
                    if ammo <= 0:
                        trigger += stats["reload"]
                        ammo = int(stats["mag"])
                        continue
                trigger += interval
            if events:
                schedules.append((shot, events))
        if not schedules:
            return Engagement()
        if len(schedules) == 1:
            shot, triggers = schedules[0]
            events = [(trigger + shot.flight, value) for trigger, value in triggers]
        else:
            events = self._combine_schedules(me, schedules, horizon)
        events.sort()
        if not events:
            return Engagement()
        hp = target.get("hp") or target.get("max_hp") or 1400
        dealt = 0.0
        kill_time = math.inf
        damage_1s = 0.0
        for impact, damage in events:
            dealt += damage
            if impact <= 1.0:
                damage_1s += damage
            if dealt >= hp and kill_time == math.inf:
                kill_time = impact
        return Engagement(events[0][0], kill_time, damage_1s, interrupt)

    def _combine_schedules(self, me, schedules, horizon):
        (first, left), (second, right) = schedules
        if self._shared_aim(first, second) is not None:
            return [(trigger + first.flight, damage) for trigger, damage in left] + [
                (trigger + second.flight, damage) for trigger, damage in right
            ]
        # One torso cannot schedule two independent aim angles. Delay each
        # trigger by the required slew and preserve that slot's next cooldown.
        shots = (first, second)
        triggers = (left, right)
        shifts = [0.0, 0.0]
        indices = [0, 0]
        slew = (self._turn_between(me, second, first), self._turn_between(me, first, second))
        active = -1
        previous = 0.0
        impacts = []
        while True:
            options = []
            for slot, queue in enumerate(triggers):
                if indices[slot] == len(queue):
                    continue
                booked, value = queue[indices[slot]]
                when = max(booked + shifts[slot], previous + slew[slot] if active == 1 - slot else previous)
                if when + shots[slot].flight <= horizon:
                    options.append((when, when + shots[slot].flight, -value, slot))
            if not options:
                return impacts
            when, impact, negative, slot = min(options)
            shifts[slot] = when - triggers[slot][indices[slot]][0]
            indices[slot] += 1
            active, previous = slot, when
            impacts.append((impact, -negative))

    def apply(self, action, obs, target, scan_yaw, allow_new_charge=True):
        me = obs["self"]
        weapons = me["weapons"]
        action["aim"] = [math.remainder(scan_yaw, math.tau), 0]
        action["fire"] = [False, False]
        gun = next((weapon for weapon in weapons if weapon and weapon["kind"] == "gun"), None)
        if target is None:
            self.laser_wait = 0
            action["reload"] = bool(gun and gun.get("reload", 0) <= 0
                                    and gun.get("ammo", 0) < gun["stats"]["mag"])
            return action

        p = me["pos"]
        t = target["pos"]
        _, height = self.dimensions(target)
        dx, dy = t[0] - p[0], t[1] - p[1]
        distance = math.hypot(dx, dy)
        own = me["stats"]
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(t[2] - p[2] + 0.52 * height - own["muzzle_height"], distance)
        default = Shot(yaw, min(own["max_pitch"], max(own["min_pitch"], pitch)), distance, 0, 0)
        action["aim"] = [default.yaw, default.pitch]
        shots = {}
        laser = None
        charge_goal = 0.0
        for slot, weapon in enumerate(weapons):
            if weapon is None:
                continue
            kind = weapon["kind"]
            cooldown = weapon.get("cooldown", 0)
            if kind == "gun" and weapon.get("reload", 0) > 0.25:
                continue
            if kind == "mortar" and cooldown > self.frame + 0.20:
                continue
            if kind == "lightning" and (cooldown > self.frame + 0.05 or distance > weapon["stats"]["range"] + 2):
                continue
            if kind == "laser":
                if not allow_new_charge and not weapon.get("charge", 0):
                    continue
                if cooldown > self.frame and not weapon.get("charge", 0):
                    continue
                laser = slot
                charge_goal = self._charge_goal(me, target, weapon)
                if not weapon.get("charge", 0) and "lightning" in target.get("weapons", ()) and distance < 48:
                    continue
            shot = self._frame_shot(obs, target, weapon)
            if shot is not None:
                shots[slot] = shot

        ready_weapons = {}
        preparation = {}
        for slot, shot in shots.items():
            weapon = weapons[slot]
            kind = weapon["kind"]
            delay = self._ready_delay(me, weapon)
            fraction = 1.0
            if kind == "laser":
                charge = weapon.get("charge", 0)
                delay = max(delay, charge_goal - charge)
                fraction = min(1, max(charge, charge_goal) / weapon["stats"]["charge"])
                if not charge or charge < charge_goal - 1e-3:
                    delay = max(delay, self.frame + self.step)
            if delay <= self.frame + 1e-4 and not (kind == "gun" and weapon.get("ammo", 0) <= 0):
                expected, connected = self._shot_damage(me, target, weapon, shot, fraction)
                ready_weapons[slot] = (delay, max(self.step, delay) + shot.flight, expected, connected)
            # Prefer aim that can use a soon-to-ready secondary when no shot
            # can land this frame. Do not let pre-aim block a payable shot.
            preparation[slot] = max(self._turn_time(me, shot), delay) + shot.flight * 0.22

        aim = default
        fire_now = ()
        best = None
        hp = target.get("hp") if target.get("hp") is not None else math.inf
        candidates = [(shot, preparation[slot]) for slot, shot in shots.items()]
        if len(ready_weapons) == 2:
            shared = self._shared_aim(shots[0], shots[1])
            if shared is not None:
                candidates.append((shared, max(preparation.values())))
        for shot, prepare in candidates:
            data = ready_weapons.copy()
            tactical_value = 0.0
            for slot, item in ready_weapons.items():
                weapon = weapons[slot]
                if weapon["kind"] != "lightning":
                    continue
                yaw, _ = self._slewed_aim(me, shot, item[0])
                mean, direct, extra, interrupt = self._arc_value(obs, target, weapon, shots[slot], yaw)
                if not mean and not interrupt and extra < 22:
                    del data[slot]
                    continue
                data[slot] = (item[0], item[1], mean, direct)
                tactical_value += extra + interrupt
            ready = tuple(slot for slot, item in data.items()
                          if weapons[slot]["kind"] == "lightning"
                          or self._heading_hits(me, shot, shots[slot], item[0]))
            lethal, time, value, preaim = self._burst_score(ready, data, hp, prepare)
            tactical_value += sum(shots[slot].collateral for slot in ready)
            score = lethal, time, value + tactical_value, preaim
            if best is None or score > best:
                best, aim, fire_now = score, shot, ready
        action["aim"] = [aim.yaw, aim.pitch]
        for slot in fire_now:
            if slot != laser:
                action["fire"][slot] = True

        if laser is not None:
            weapon = weapons[laser]
            charge = weapon.get("charge", 0)
            shot = shots.get(laser)
            if charge:
                ready = charge >= charge_goal - 1e-3
                aligned = laser in fire_now
                self.laser_wait = self.laser_wait + self.frame if ready or shot is None else 0
                action["fire"][laser] = not (ready and aligned or self.laser_wait > 0.45)
            elif shot is not None:
                self.laser_wait = 0
                action["fire"][laser] = abs(angle_error(shot.yaw, me["aim"][0])) < 0.32
        return action

    @staticmethod
    def _burst_score(ready, data, hp, prepare):
        impact_order = sorted(ready, key=lambda slot: data[slot][1])
        expected = connected = value = 0.0
        kill = math.inf
        for slot in impact_order:
            _, impact, mean, hit = data[slot]
            expected += mean
            connected += hit
            value += mean / (1 + 0.8 * impact)
            # A quick shot with good hit probability can finish before a
            # stronger, slower round reaches the same target.
            if kill == math.inf and (expected >= hp or
                                     impact < 0.45 and connected >= hp and expected >= hp * 0.76):
                kill = impact
        return math.isfinite(kill), -kill, value, -prepare

    def _charge_goal(self, me, target, weapon):
        stats = weapon["stats"]
        full = stats["charge"]
        hp = target.get("hp")
        if hp is None:
            return full
        cubes = self.config["cubes"]
        damage = stats["damage"] * (1 + cubes["damage_per_cube"] * min(me["cubes"], cubes["stat_cap"]))
        if "mortar" in target.get("weapons", ()):
            damage *= 1 + stats["cookoff"]
        if target.get("kind") == "mech":
            # Enemy armor is private. Do not release early on a marginal estimate.
            yaw = math.atan2(target["pos"][1] - me["pos"][1], target["pos"][0] - me["pos"][0])
            damage *= 0.92 * self._rear_bonus(target, yaw, full - weapon.get("charge", 0))
        needed = (hp / damage - stats["min_frac"]) / (1 - stats["min_frac"])
        return min(full, max(stats["min_charge"], needed * full + self.step))
