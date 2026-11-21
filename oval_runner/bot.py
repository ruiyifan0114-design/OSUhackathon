"""Choose one match intent, then steer and shoot for it."""
import math

from arena import Arena
from awareness import Awareness
from combat import Combat
from movement import Movement, distance
import mechfight as mf


# Value of one upgrade effect, keyed by the effect path from hello.config.
# Effects are additive here: a two-effect upgrade gets the sum of its parts.
# This is deliberately path-based so new ids or rebalanced descriptions do
# not silently become zero-value choices.
UPGRADE_PATH_VALUE = {
    "mech.hp": 7.2,
    "mech.armor": 6.2,
    "mech.walk_speed": 3.8,
    "mech.boost_speed": 3.8,
    "mech.energy": 3.5,
    "mech.energy_regen": 3.5,
    "mech.boost_drain": 2.7,
    "mech.jet_drain": 2.8,
    "mech.regen_per_s": 5.0,
    "mech.regen_delay": 2.0,
    "mech.turn_rate": 2.4,
    "mech.pitch_rate": 1.8,
    "mech.near_range_bonus": 2.6,
    "mech.vision_bonus": 1.8,
    "mech.hearing": 2.2,
    "weapons.gun.damage": 5.0,
    "weapons.gun.cooldown": 5.0,
    "weapons.gun.speed": 2.0,
    "weapons.gun.range": 1.5,
    "weapons.gun.mag": 3.5,
    "weapons.gun.reload": 3.0,
    "weapons.mortar.splash_radius": 3.0,
    "weapons.mortar.cooldown": 4.5,
    "weapons.mortar.speed": 2.0,
    "weapons.laser.charge": 3.5,
    "weapons.laser.damage": 6.0,
    "weapons.laser.pierce": 5.0,
    "weapons.lightning.chain": 6.0,
    "weapons.lightning.range": 5.0,
    "weapons.lightning.cone_deg": 2.0,
    "weapons.lightning.energy": 4.0,
    "weapons.lightning.damage": 2.0,
}

# Small situational corrections for effects whose value is not captured by
# their stat path alone.
UPGRADE_ID_BONUS = {
    "nanites": 0.8,
    "sensors": 0.6,
    "plating": 0.4,
    "servos": 0.4,
    "jets": 0.3,
    "lightning_chain": 0.8,
    "laser_pierce": 0.5,
}


class Bot:
    name = "Oval Runner IV"

    def __init__(self, hello):
        self.config = hello["config"]
        self.arena = Arena(hello)
        self.combat = Combat(hello, self.arena)
        self.movement = Movement(hello, self.arena)
        self.awareness = Awareness(hello)
        self.upgrade_defs = {item["id"]: item for item in self.config.get("upgrades", ())}
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
        self.laser_cooldown = {}
        self.last_attacker = None

    def update(self, obs):
        now = obs["t"]
        dt = min(2, max(0, now - self.last_time))
        self.hit_rate *= math.exp(-dt / 2.0)
        self.last_time = now
        self.awareness.update(obs)
        for event in obs["events"]:
            kind = event["type"]
            if kind == "damaged" and event["weapon"] != "zone":
                self.hit_rate += event["amount"] / 2.0
                self.damage_taken += event["amount"]
                self.last_bearing = event["bearing"]
                self.last_attacker = event.get("by")
                self.last_damage = now
            elif kind == "hit" and event["target"] < 16:
                self.damage_done += event["amount"]
            elif kind == "kill":
                self.kills += 1
                if self.first_kill is None:
                    self.first_kill = now
            elif kind == "took" and event["kind"] != "gun" and self.first_weapon is None:
                self.first_weapon = now
        for beam in obs["beams"]:
            if beam["kind"] == "laser":
                self.laser_cooldown[beam["by"]] = now - beam["age"] + self.config["weapons"]["laser"]["cooldown"]

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
        nearby = [x for x in enemies if distance(pos, x["pos"]) < 190]
        zone_goal, urgent = self.zone_goal(obs)
        threats = self.threats(obs, nearby)
        alert = self.awareness.alert(now)
        sound_pressure = alert.score(now) if alert is not None else 0.0
        danger = max(self.hit_rate, sum(rate for rate, _, _ in threats),
                     sound_pressure * 0.55)
        exposed = len(threats)
        hp_fraction = me["hp"] / me["max_hp"]
        if zone["next_radius"] == 0 and zone["shrink_end"] - now < 8:
            self.recovering = False
        elif hp_fraction < (0.57 if exposed > 1 else 0.44) or danger * 2.4 > me["hp"]:
            self.recovering = True
        elif hp_fraction >= (0.77 if exposed else 0.70):
            self.recovering = False

        action = {}
        if obs["offer"]:
            action["pick"] = self.choose_upgrade(obs)
        ratings = self.weapon_ratings(obs, enemies)
        rating = ratings[secondary["kind"]] if secondary else 0
        reach = self.config["loot"]["pickup_radius"]
        options = [p for p in obs["pickups"] if p["kind"] != "gun"
                   and ratings[p["kind"]] > rating + (0.8 if secondary else 0)
                   and distance(pos, p["pos"]) < reach and abs(p["pos"][2] - pos[2]) < 4]
        safe_swap = not secondary or danger < 115 and not secondary.get("charge", 0)
        if options and safe_swap:
            action["take"] = max(options, key=lambda p: ratings[p["kind"]])["id"]
            if secondary:
                action["drop"] = 1

        goal, mode, target, ring = self.plan(
            obs, enemies, nearby, threats, danger, zone_goal, urgent,
            ratings, rating, alert,
        )
        if mode in ("fight", "finish", "breakout"):
            self.combat_target = target["id"]
        else:
            self.combat_target = None
        if target is None and self.last_bearing is not None and now - self.last_damage < 1.6:
            scan = self.last_bearing
        elif target is None and alert is not None and alert.age(now) < 2.5:
            # The vision wedge does not cover behind us.  Turn the torso toward
            # the strongest recent sound and let the next observation confirm
            # whether it was a mech, loot, or scenery.
            scan = alert.bearing
        elif goal is not None and distance(pos, goal) > 5:
            scan = math.atan2(goal[1] - pos[1], goal[0] - pos[0]) + 0.50 * math.sin(now * 1.25)
        else:
            scan = me["aim"][0] + 0.31
        escaping = mode in ("zone", "retreat", "breakout")
        self.combat.apply(action, obs, target, scan, allow_new_charge=not escaping)
        if target is None and not escaping and alert is not None:
            self.blind_lightning(action, obs, alert)
        if secondary and secondary["kind"] == "laser" and escaping:
            action["fire"][1] = False
        steering_mode = "retreat" if mode == "breakout" else mode
        self.movement.apply(action, obs, goal, steering_mode, nearby,
                            ring=ring, target=target, alert=alert)
        self.trace(obs, mode, target)
        return action

    def choose_upgrade(self, obs):
        """Pick the offer with the best config-derived marginal value."""
        choices = obs["offer"]["choices"]
        held = {weapon["kind"] for weapon in obs["self"]["weapons"] if weapon}
        return max(
            range(len(choices)),
            key=lambda index: self.upgrade_score(choices[index], held),
        )

    def upgrade_score(self, choice, held):
        definition = self.upgrade_defs.get(choice["id"], choice)
        score = 0.0
        for effect in definition.get("effects", ()):
            score += UPGRADE_PATH_VALUE.get(effect.get("path"), 0.0)
        score += UPGRADE_ID_BONUS.get(choice["id"], 0.0)
        requires = definition.get("requires")
        if requires and requires not in held:
            # The protocol currently offers weapon upgrades only for held
            # weapons, but keep the valuation safe if that changes.
            score *= 0.20
        return score if score > 0 else 1.0

    def blind_lightning(self, action, obs, alert):
        """Use a close sound as a coarse target for the wide lightning cone."""
        me = obs["self"]
        if alert.age(obs["t"]) > 0.75 or alert.distance > 44:
            return
        if alert.kind not in ("gun", "mortar", "laser", "laser_charge",
                              "lightning", "step", "boost", "jet"):
            return
        secondary = me["weapons"][1]
        if secondary is None or secondary["kind"] != "lightning":
            return
        stats = secondary["stats"]
        if (secondary.get("cooldown", 0) > 0.15
                or me["energy"] < stats["energy"] + 18):
            return
        error = abs(math.remainder(alert.bearing - me["aim"][0], math.tau))
        if error > math.radians(stats["cone_deg"]) * 0.36:
            return
        origin = (me["pos"][0], me["pos"][1], me["pos"][2] + me["stats"]["muzzle_height"])
        contact = alert.point(me["pos"])
        contact = (contact[0], contact[1], me["pos"][2] + 4.5)
        if not self.arena.clear(origin, contact):
            return
        action["aim"] = [alert.bearing, 0.0]
        action["fire"][1] = True

    def threats(self, obs, nearby):
        """Rank exposed opponents by sustained pressure and telegraphed burst."""
        me = obs["self"]
        pos = me["pos"]
        body = (pos[0], pos[1], pos[2] + me["stats"]["height"] * 0.6)
        threats = []
        weapons = self.config["weapons"]
        cubes = self.config["cubes"]
        now = obs["t"]
        for foe in nearby[:5]:
            x, y, z = foe["pos"]
            d = distance(pos, foe["pos"])
            if not self.arena.clear((x, y, z + 7), body):
                continue
            yaw = math.atan2(pos[1] - y, pos[0] - x)
            turn = abs(math.remainder(foe["aim"][0] - yaw, math.tau))
            facing = 1.0 if turn < 0.55 else 0.60
            kinds = foe.get("weapons", ())
            rate = max(12, 86 - d * 0.50)
            burst = 0.0
            arc_range = weapons["lightning"]["range"]
            if "lightning" in kinds and d < arc_range * 1.22:
                rate += 154 if d < arc_range else 82
            if "laser" in kinds:
                if foe.get("charging"):
                    burst = weapons["laser"]["damage"] * (0.73 if turn < 0.4 else 0.35)
                elif now >= self.laser_cooldown.get(foe["id"], 0):
                    rate += 58
            if "mortar" in kinds:
                rate += 65 if d < 75 else 28
            scale = (1 + cubes["damage_per_cube"] * min(cubes["stat_cap"], foe.get("cubes", 0)))
            scale *= 1 - me["stats"].get("armor", 0)
            threats.append((rate * facing * scale, burst * scale, foe))
        recent_attacker = self.last_attacker if now - self.last_damage < 1 else None
        return sorted(threats, key=lambda entry: entry[0] + entry[1] +
                      (25 if entry[2]["id"] == recent_attacker else 0), reverse=True)

    def plan(self, obs, enemies, nearby, threats, danger, zone_goal, urgent,
             ratings, rating, alert):
        me = obs["self"]
        pos = me["pos"]
        now = obs["t"]
        zone = obs["zone"]
        secondary = me["weapons"][1]
        hp = me["hp"]
        nearest = distance(pos, enemies[0]["pos"]) if enemies else math.inf
        goal, resource, mode = self.objective(obs, zone_goal, urgent, nearby, ratings, rating)
        exposed = len(threats)

        # Spend the short-horizon combat estimate on nearby, plausible opponents.
        ranked = []
        isolation_by_id = {}
        for foe in enemies[:4]:
            d = distance(pos, foe["pos"])
            if d > 190 or urgent and d > 55:
                break
            estimate = self.combat.estimate(obs, foe)
            if not math.isfinite(estimate.first_shot):
                continue
            stick = 10 if foe["id"] == self.combat_target else 0
            foe_hp = foe.get("hp") or foe.get("max_hp") or 1400
            foe_max = foe.get("max_hp") or 1400
            wounded = 1 - min(1.0, foe_hp / max(1.0, foe_max))
            revenge = 8 if foe["id"] == self.last_attacker and now - self.last_damage < 3 else 0
            isolation = min(
                (distance(foe["pos"], other["pos"]) for other in enemies
                 if other["id"] != foe["id"]),
                default=200.0,
            )
            isolation_by_id[foe["id"]] = isolation
            # The practice squad shares vision and does not fight itself.
            # A target separated from its squad is the only kind of fight
            # that can be converted into a clean kill.
            room_bonus = 9.0 * min(1.0, isolation / 55.0)
            crowd_penalty = 7.0 if isolation < 22 else 0.0
            score = (0.58 * estimate.damage_1s + 165 / (0.75 + estimate.kill_time)
                     - 0.29 * d - 19 * min(2, estimate.first_shot)
                     + 34 * wounded + revenge + stick
                     + room_bonus - crowd_penalty)
            ranked.append((score, foe, estimate))
        _, target, estimate = max(ranked, key=lambda item: item[0], default=(0, None, None))
        if urgent:
            return zone_goal, "zone", target, None

        reserve = me["max_hp"] * (0.09 if obs["alive"] <= 3 else 0.16)
        burst = sum(amount for _, amount, _ in threats)
        rate_by_id = {foe["id"]: rate for rate, _, foe in threats}

        def zone_safe(foe):
            # When the next circle is about to move, do not commit to a kill
            # that lies outside it.  After the move there is still time to
            # take an opportunistic shot without chasing.
            return (
                zone["shrink_end"] - now > 25
                or distance(foe["pos"], zone["next_center"])
                <= max(0, zone["next_radius"] - 8)
            )

        finish = None
        best_finish = -math.inf
        for score, foe, eta in ranked:
            foe_hp = foe.get("hp") or foe.get("max_hp") or 1400
            finish_window = (2.8 if foe_hp <= me["max_hp"] * 0.55
                             or isolation_by_id.get(foe["id"], 0) > 45 else 2.3)
            if eta.kill_time > finish_window or not zone_safe(foe):
                continue
            # Count incoming fire until the kill, then only fire from survivors.
            remaining_rate = max(0, danger - rate_by_id.get(foe["id"], 0))
            loss = danger * eta.kill_time + remaining_rate * 0.55 + burst
            utility = score - 0.43 * loss - 30 * eta.kill_time
            if hp - loss > reserve and utility > best_finish:
                best_finish, finish = utility, (foe, eta)
        if finish:
            foe, eta = finish
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
            sound_clear = (alert is None or alert.age(now) > 1.8
                           or alert.distance > 110)
            if (not contested and sound_clear
                    and (danger * duration < hp * 0.55
                         or exposed == 0 and danger < 20)):
                return goal, "drop", target, (drop["pos"], drop["radius"])
            mode, goal, resource = "travel", zone_goal, None

        own_burst = estimate.damage_1s if estimate else 0
        outmatched = (exposed >= 2 and hp < me["max_hp"] * 0.86
                      and danger + burst * 0.6 > 95 + own_burst * 1.27)
        reloading = (exposed > 0 and nearest < 70 and estimate is not None
                     and estimate.first_shot > 0.65 and hp < danger * 3 + burst
                     and obs["self"]["weapons"][0].get("reload", 0) > 0.45)
        if self.recovering or outmatched or reloading:
            immediate = [foe for _, _, foe in threats] or nearby
            if not immediate and nearest < 170 and obs["t"] - self.last_damage < 3:
                immediate = enemies[:1]
            cover = self.retreat_cover(obs, immediate, zone_goal)
            freshly_hit = obs["t"] - self.last_damage < 1.8
            if cover is None and immediate and (exposed or freshly_hit or not me["in_zone"]):
                cover = self.movement.engage(obs, immediate[0], 70, True, zone_goal)
            if not immediate and self.last_bearing is not None and obs["t"] - self.last_damage < 1.4:
                bearing = self.last_bearing
                cover = (pos[0] - 24 * math.cos(bearing), pos[1] - 24 * math.sin(bearing), pos[2])
            elif not immediate and alert is not None and alert.age(now) < 2.0:
                # No visible attacker, but a close sound still justifies
                # opening distance while the torso scans toward it.
                bearing = alert.bearing
                cover = (pos[0] - 20 * math.cos(bearing), pos[1] - 20 * math.sin(bearing), pos[2])
            # Trading can still win if we're already pinned, but suppress fire once
            # the lane is broken so we can complete the repair delay.
            if (target is not None and nearest < 24 and freshly_hit and exposed
                    and estimate.first_shot <= 0.28 and estimate.damage_1s > 90
                    and (cover is None or distance(pos, cover) > me["stats"]["walk_speed"] * 0.9)
                    and hp < danger * 2.0 + burst):
                return cover, "breakout", target, None
            # Retreat is not the same as switching off the gun.  A nearby
            # pursuer is already committed; fire back while opening distance
            # or using cover.  The escaping flag still forbids a new laser
            # charge, so this does not slow the retreat.
            defensive = None
            visible = [foe for _, _, foe in threats]
            if visible:
                candidate = min(visible, key=lambda foe: distance(pos, foe["pos"]))
                if distance(pos, candidate["pos"]) < 140:
                    defensive = candidate
            return cover, "retreat", defensive, None

        # An unseen close sound is evidence, not a target.  Do not keep farming
        # or walking into it; back off while the scan layer looks toward it.
        unseen_close = (
            alert is not None and not threats and alert.age(now) < 1.8
            and alert.distance < 120 and alert.score(now) > 30
        )
        if unseen_close and not urgent and mode in ("farm", "travel", "idle", "weapon", "cube"):
            bearing = alert.bearing
            away = (pos[0] - 18 * math.cos(bearing),
                    pos[1] - 18 * math.sin(bearing), pos[2])
            if distance(away, zone["center"]) < zone["radius"] - 10:
                return away, "caution", None, None

        if mode == "weapon" and not secondary and nearest > 36:
            return goal, mode, target if estimate and estimate.first_shot < 0.35 else None, None
        if mode == "farm" and (nearest > 90 and danger < 85 or target is None):
            return goal, mode, resource, None
        if (target is not None and distance(pos, target["pos"]) < 155
                and (mode not in ("weapon", "cube") or nearest < 42)):
            d = distance(pos, target["pos"])
            preferred = self.movement.preferred_range(obs, target)
            kill_time = estimate.kill_time if estimate and math.isfinite(estimate.kill_time) else 2.5
            target_hp = target.get("hp") or target.get("max_hp") or 1400
            weak = target_hp <= max(90, me["max_hp"] * 0.34) or kill_time <= 1.10
            close = d <= max(48, preferred + 24)
            late = obs["alive"] <= 4 or zone["next_radius"] <= 70
            incoming = danger * min(kill_time, 2.2) + burst * 0.6
            affordable = hp - incoming > reserve and hp > me["max_hp"] * 0.26
            if (zone_safe(target)
                    and ((affordable and close) or (weak and close)
                         or (late and affordable and d < preferred + 25))):
                return self.movement.engage(obs, target, preferred, False, zone_goal), "fight", target, None
            # Not worth crossing open ground for.  Keep the objective and let
            # the aim layer take only shots that are already available.
            return goal, mode, target if d < 150 else None, None
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
        ratings = {"gun": 0, "mortar": 6.3, "laser": 4.4, "lightning": 6.0}
        # A squad that shares vision often converges in pairs.  Splash and
        # chain damage gain value exactly when two or more targets are close.
        visible = enemies[:6]
        clustered = sum(
            1 for index, foe in enumerate(visible)
            if any(index != other_index and distance(foe["pos"], other["pos"]) <= 8
                   for other_index, other in enumerate(visible))
        )
        if clustered >= 2:
            ratings["mortar"] += min(1.8, 0.6 * clustered)
            ratings["lightning"] += 0.2 * clustered
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
        # Use a conservative effective speed: the local steering has to route
        # around buildings and may have to jump or wait for a clear lane.  A
        # straight-line estimate at full walk speed chronically starts too late.
        speed = me["stats"]["walk_speed"] * 0.48
        required = max(0, d - max(0, radius - 20)) / speed
        urgent = (
            not me["in_zone"]
            or zone_room < (14 if now >= zone["shrink_start"] else 1)
            or required > max(3, zone["shrink_end"] - now - 15)
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
            # Do not chase loot or farm targets that are about to be outside
            # the next circle.  Supply drops are already placed inside it.
            if (mode != "drop" and zone["shrink_end"] - now < 30
                    and distance(point, zone["next_center"]) > zone["next_radius"] - 8):
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
