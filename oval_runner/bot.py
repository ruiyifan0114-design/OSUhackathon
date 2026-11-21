"""Choose one match intent, then steer and shoot for it."""
import math

import mechfight as mf
from arena import Arena
from combat import Combat
from movement import Movement, distance


UPGRADES = {
    "nanites": 9, "gun_mag": 8, "lightning_range": 8, "plating": 7,
    "gun_damage": 6, "armor": 6, "lightning_efficiency": 6,
    "mortar_reload": 6, "servos": 5, "laser_damage": 5, "mortar_splash": 5,
    "laser_charge": 4, "jets": 4, "gun_velocity": 4, "gun_rate": 3,
    "reactor": 3, "capacitor": 2, "sensors": 2, "gyros": 2,
}


class Bot:
    name = "Oval Runner III"

    def __init__(self, hello):
        self.config = hello["config"]
        self.arena = Arena(hello)
        self.combat = Combat(hello, self.arena)
        self.movement = Movement(hello, self.arena)
        self.recovering = False
        self.hit_rate = 0
        self.last_time = 0
        self.last_bearing = None
        self.last_damage = -100
        self.combat_target = None
        self.goal_id = None
        self.loot = {}
        self.explore_angle = (hello["slot"] / 16) * math.tau
        self.last_trace = None
        self.trace_at = -1
        self.cover_cache = None
        self.first_weapon = self.first_kill = None
        self.damage_done = self.damage_taken = self.kills = 0

    def update(self, obs):
        now = obs["t"]
        dt = min(2, max(0, now - self.last_time))
        self.hit_rate *= math.exp(-dt / 2.0)
        self.last_time = now
        for event in obs["events"]:
            kind = event["type"]
            if kind == "damaged" and event["weapon"] != "zone":
                self.hit_rate += event["amount"] / 2.0
                self.damage_taken += event["amount"]
                self.last_bearing = event["bearing"]
                self.last_damage = now
            elif kind == "hit" and event["target"] < 16:
                self.damage_done += event["amount"]
            elif kind == "kill":
                self.kills += 1
                if self.first_kill is None:
                    self.first_kill = now
            elif kind == "took" and event["kind"] != "gun" and self.first_weapon is None:
                self.first_weapon = now

        visible = set()
        for kind in ("cubes", "pickups", "crates"):
            for item in obs[kind]:
                visible.add(item["id"])
                self.loot[item["id"]] = (kind, item, now)
        pos = obs["self"]["pos"]
        for key, (kind, item, seen) in list(self.loot.items()):
            if key not in visible and (now - seen > 16 or distance(pos, item["pos"]) < 8):
                del self.loot[key]

    def act(self, obs):
        self.update(obs)
        me = obs["self"]
        pos = me["pos"]
        now = obs["t"]
        zone = obs["zone"]
        secondary = me["weapons"][1]
        enemies = sorted((x for x in obs["mechs"] if x["kind"] == "mech"),
                         key=lambda x: distance(pos, x["pos"]))
        nearby = [x for x in enemies if distance(pos, x["pos"]) < 115]
        zone_goal, urgent = self.zone_goal(obs)
        danger, exposed = self.pressure(obs, nearby)
        hp_fraction = me["hp"] / me["max_hp"]
        if zone["next_radius"] == 0 and zone["shrink_end"] - now < 8:
            self.recovering = False
        elif hp_fraction < (0.57 if exposed > 1 else 0.44) or danger * 2.4 > me["hp"]:
            self.recovering = True
        elif hp_fraction >= (0.77 if exposed else 0.70):
            self.recovering = False

        action = {}
        if obs["offer"]:
            choices = obs["offer"]["choices"]
            action["pick"] = max(range(len(choices)), key=lambda i: UPGRADES.get(choices[i]["id"], 0))
        ratings = self.weapon_ratings(obs, enemies)
        rating = ratings[secondary["kind"]] if secondary else 0
        reach = self.config["loot"]["pickup_radius"]
        options = [p for p in obs["pickups"] if p["kind"] != "gun"
                   and ratings[p["kind"]] > rating + (0.8 if secondary else 0)
                   and distance(pos, p["pos"]) < reach and abs(p["pos"][2] - pos[2]) < 4]
        if options and not (secondary and secondary.get("charge", 0)):
            action["take"] = max(options, key=lambda p: ratings[p["kind"]])["id"]
            if secondary:
                action["drop"] = 1

        goal, mode, target, ring = self.plan(
            obs, enemies, nearby, danger, exposed, zone_goal, urgent, ratings, rating
        )
        if mode in ("fight", "finish"):
            self.combat_target = target["id"]
        else:
            self.combat_target = None
        if target is None and self.last_bearing is not None and now - self.last_damage < 1.6:
            scan = self.last_bearing
        elif goal is not None and distance(pos, goal) > 5:
            scan = math.atan2(goal[1] - pos[1], goal[0] - pos[0]) + 0.50 * math.sin(now * 1.25)
        else:
            scan = me["aim"][0] + 0.31
        self.combat.apply(action, obs, target, scan, allow_new_charge=mode != "zone")
        if secondary and secondary["kind"] == "laser" and mode == "zone":
            # Release an existing charge instead of slowing the zone rotation.
            action["fire"][1] = False
        self.movement.apply(action, obs, goal, mode, nearby, ring=ring)
        self.trace(obs, mode, target)
        return action

    def pressure(self, obs, nearby):
        me = obs["self"]
        pos = me["pos"]
        body = (pos[0], pos[1], pos[2] + me["stats"]["height"] * 0.6)
        visible_dps = 0
        exposed = 0
        for foe in nearby[:5]:
            x, y, z = foe["pos"]
            d = distance(pos, foe["pos"])
            if not self.arena.clear((x, y, z + 7), body):
                continue
            exposed += 1
            yaw = math.atan2(pos[1] - y, pos[0] - x)
            facing = 1.0 if abs(math.remainder(foe["aim"][0] - yaw, math.tau)) < 0.8 else 0.55
            kinds = foe.get("weapons", ())
            dps = max(12, 86 - d * 0.50)
            if "lightning" in kinds and d < 53:
                dps += 150
            if "laser" in kinds:
                dps += 115 if foe.get("charging") else 58
            if "mortar" in kinds:
                dps += 65 if d < 75 else 28
            visible_dps += dps * facing
        return max(visible_dps, self.hit_rate), exposed

    def plan(self, obs, enemies, nearby, danger, exposed, zone_goal, urgent, ratings, rating):
        me = obs["self"]
        pos = me["pos"]
        secondary = me["weapons"][1]
        hp = me["hp"]
        nearest = distance(pos, enemies[0]["pos"]) if enemies else math.inf
        goal, resource, mode = self.objective(obs, zone_goal, urgent, nearby, ratings, rating)

        # Spend the short-horizon combat estimate on nearby, plausible opponents.
        ranked = []
        for foe in enemies[:4]:
            d = distance(pos, foe["pos"])
            if d > 170 or urgent and d > 55:
                break
            estimate = self.combat.estimate(obs, foe)
            if not math.isfinite(estimate.first_shot):
                continue
            stick = 10 if foe["id"] == self.combat_target else 0
            score = (0.58 * estimate.damage_1s + 165 / (0.75 + estimate.kill_time)
                     - 0.29 * d - 19 * min(2, estimate.first_shot) + stick)
            ranked.append((score, foe, estimate))
        _, target, estimate = max(ranked, key=lambda item: item[0], default=(0, None, None))
        if urgent:
            return zone_goal, "zone", target, None

        reserve = me["max_hp"] * (0.10 if obs["alive"] == 2 else 0.18)
        finish = max(
            (entry for entry in ranked if entry[2].kill_time <= 1.9
             and hp - danger * (entry[2].kill_time + 0.32) > reserve),
            key=lambda entry: entry[0], default=None,
        )
        if finish:
            foe, eta = finish[1:]
            # Stay in the firing lane for a shot already available; intercept only a fleeing target.
            d = distance(pos, foe["pos"])
            chase = eta.first_shot > 0.32 and d > 31
            return self.movement.engage(obs, foe, min(43, d), False, zone_goal, finish=chase), "finish", foe, None

        if mode == "drop":
            drop = resource
            contested = any(distance(drop["pos"], foe["pos"]) <= drop["radius"] + 2 for foe in enemies)
            travel = max(0, distance(pos, goal) - drop["radius"]) / me["stats"]["walk_speed"]
            progress = drop["progress"] if drop["holder"] == me["id"] else 0
            duration = max(travel, drop["lands_in"]) + max(0, drop["capture_s"] - progress)
            if not contested and (danger * duration < hp * 0.62 or exposed == 0):
                return goal, "drop", target, (drop["pos"], drop["radius"])
            mode, goal, resource = "travel", zone_goal, None

        if self.recovering:
            immediate = nearby or (enemies[:1] if nearest < 150 and obs["t"] - self.last_damage < 3 else ())
            cover = self.retreat_cover(obs, immediate, zone_goal)
            freshly_hit = obs["t"] - self.last_damage < 1.8
            if cover is None and immediate and (exposed or freshly_hit or not me["in_zone"]):
                cover = self.movement.engage(obs, immediate[0], 70, True, zone_goal)
            if not immediate and self.last_bearing is not None and obs["t"] - self.last_damage < 1.4:
                bearing = self.last_bearing
                cover = (pos[0] - 24 * math.cos(bearing), pos[1] - 24 * math.sin(bearing), pos[2])
            return cover, "retreat", None, None

        if mode == "weapon" and not secondary and nearest > 36:
            return goal, mode, target if estimate and estimate.first_shot < 0.35 else None, None
        if mode == "farm" and (nearest > 90 and danger < 85 or target is None):
            return goal, mode, resource, None
        if (target is not None and distance(pos, target["pos"]) < 155
                and (mode not in ("weapon", "cube") or nearest < 42)):
            preferred = 67
            if secondary and secondary["kind"] == "lightning" and me["energy"] > 15:
                preferred = secondary["stats"]["range"] * 0.79
            elif secondary and secondary["kind"] == "mortar":
                preferred = 51
            return self.movement.engage(obs, target, preferred, False, zone_goal), "fight", target, None
        return goal, mode, None, None

    def retreat_cover(self, obs, enemies, zone_goal):
        if not enemies:
            return None
        zone = obs["zone"]
        key = tuple(foe["id"] for foe in enemies[:2]), zone["shrink_end"]
        if self.cover_cache:
            cached_key, expires, point = self.cover_cache
            if (cached_key == key and obs["t"] < expires
                    and (point is None or distance(point, zone["center"]) < zone["radius"] - 2)):
                return point
        point = self.movement.cover_goal(obs, enemies, zone_goal)
        self.cover_cache = key, obs["t"] + 0.35, point
        return point

    def trace(self, obs, mode, target):
        target_id = target["id"] if target is not None else None
        key = mode, target_id, self.kills
        now = obs["t"]
        if key == self.last_trace and now < self.trace_at:
            return
        self.last_trace, self.trace_at = key, now + 1
        weapon = "-" if self.first_weapon is None else f"{self.first_weapon:.1f}"
        kill = "-" if self.first_kill is None else f"{self.first_kill:.1f}"
        mf.draw.label(f"{mode}:{target_id} k{self.kills} dmg{int(self.damage_done)}/{int(self.damage_taken)} w{weapon} f{kill}")

    def weapon_ratings(self, obs, enemies):
        small = obs["zone"]["next_radius"] <= 70
        has_mortar_enemy = any("mortar" in e.get("weapons", ()) for e in enemies[:3])
        ratings = {"gun": 0, "mortar": 5.1, "laser": 4.4, "lightning": 6.0}
        if small:
            ratings["lightning"] += 2
        elif has_mortar_enemy:
            ratings["laser"] += 1.6
        for upgrade in obs["self"]["upgrades"]:
            prefix = upgrade.split("_", 1)[0]
            if prefix in ratings and prefix != "gun":
                ratings[prefix] += 0.65
        return ratings

    @staticmethod
    def zone_goal(obs):
        zone = obs["zone"]
        me = obs["self"]
        pos = me["pos"]
        now = obs["t"]
        zone_room = zone["radius"] - distance(pos, zone["center"])
        center = zone["next_center"]
        radius = zone["next_radius"]
        d = distance(pos, center)
        speed = me["stats"]["walk_speed"] * 0.62
        required = max(0, d - max(0, radius - 18)) / speed
        urgent = (
            not me["in_zone"]
            or zone_room < (12 if now >= zone["shrink_start"] else 1)
            or required > max(3, zone["shrink_end"] - now - 12)
            or radius == 0 and zone["shrink_end"] - now < 10
        )
        if d <= 2 or radius <= 15:
            return center, urgent
        inner = 0 if radius <= 25 else max(0, radius - (42 if urgent else 75))
        ratio = min(d, inner) / d
        return (center[0] + (pos[0] - center[0]) * ratio,
                center[1] + (pos[1] - center[1]) * ratio), urgent

    def objective(self, obs, zone_goal, urgent, nearby, ratings, current_weapon):
        me = obs["self"]
        pos = me["pos"]
        zone = obs["zone"]
        if urgent:
            self.goal_id = None
            return zone_goal, None, "zone"
        candidates = []
        hp = me["hp"] / me["max_hp"]
        now = obs["t"]
        vision_range = 125
        can_farm = not self.recovering and len(nearby) < 2 and self.hit_rate < 35 and me["cubes"] < 10

        def add(identity, point, mode, value, actor=None, radius=vision_range):
            d = distance(pos, point)
            if d > radius or distance(point, zone["center"]) > zone["radius"] - 8:
                return
            if mode != "drop" and abs(point[2] - pos[2]) > 5:
                return
            threat = min((distance(point, e["pos"]) for e in nearby), default=150)
            danger = max(0, 38 - threat) * (2.6 if self.recovering else 1.25)
            commitment = 9 if identity == self.goal_id else 0
            score = value + commitment - d - danger
            candidates.append((score, identity, point, mode, actor))

        for key, (kind, item, seen) in self.loot.items():
            if kind == "cubes":
                add(key, item["pos"], "cube", 61 if me["cubes"] < 10 else 30, radius=74)
            elif kind == "pickups":
                value = ratings[item["kind"]]
                if item["kind"] != "gun" and value > current_weapon + (0.8 if current_weapon else 0):
                    add(key, item["pos"], "weapon", 127 if not current_weapon else 42,
                        radius=127 if not current_weapon else 90)
            elif can_farm and now - seen < 5:
                add(key, item["pos"], "farm", 41 + 0.061 * (300 - item.get("hp", 300)), item, radius=87)

        if can_farm:
            for npc in obs["mechs"]:
                if npc["kind"] == "npc":
                    add(npc["id"], npc["pos"], "farm", 65 + 0.07 * (350 - npc.get("hp", 350)), npc, radius=90)
        for drop in obs["drops"]:
            d = distance(pos, drop["pos"])
            if drop["lands_in"] <= d / max(1, me["stats"]["walk_speed"]) + 8 and drop["holder"] in (None, me["id"]):
                add(drop["id"], drop["pos"], "drop", 115 + 60 * (1 - hp) + 18 * drop["progress"],
                    drop, radius=185)

        if candidates:
            score, self.goal_id, point, mode, target = max(candidates, key=lambda item: item[0])
            if score > -19:
                if self.recovering and mode == "farm":
                    return None, None, "idle"
                return point, target, mode
        self.goal_id = None
        if self.recovering:
            return None, None, "idle"
        if distance(pos, zone_goal) > 20:
            return zone_goal, None, "travel"
        if zone["next_radius"] < 55:
            return zone["next_center"], None, "travel"
        if me["cubes"] < 10 and not nearby:
            # Explore in persistent bursts so the forward vision wedge finds loot.
            angle = self.explore_angle + math.floor(now / 9) * 1.49
            point = (pos[0] + 46 * math.cos(angle), pos[1] + 46 * math.sin(angle), pos[2])
            if distance(point, zone["next_center"]) < zone["next_radius"] - 25:
                return point, None, "travel"
            return zone["next_center"], None, "travel"
        return None, None, "idle"


if __name__ == "__main__":
    mf.run(Bot)
