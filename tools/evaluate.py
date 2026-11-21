"""Summarize recorded Mech Royale JSONL streams (one match per file)."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import mean


def summarize(rows):
    result = {"placement": None, "kills": None, "first_kill_s": None,
              "first_secondary_s": None, "damage_by_weapon": {}, "damage_received": {},
              "repair_hp_observed": 0.0, "kill_survival_5s": {"confirmed": 0, "observed_kills": 0}}
    last_time = None
    last_hp = None
    last_max_hp = None
    kill_times = []
    active_finish = None
    confirmed_finishes = attempts = 0
    has_modes = False
    for record in rows:
        row = record.get("obs", record)
        if row.get("type") == "end":
            result["placement"] = row.get("place")
            result["kills"] = row.get("kills")
        if row.get("type") != "obs":
            continue
        last_time = row["t"]
        own = row["self"]
        events = row.get("events", ())
        if (last_hp is not None and own["max_hp"] == last_max_hp and own["hp"] > last_hp
                and not any(event["type"] == "drop" for event in events)):
            result["repair_hp_observed"] += own["hp"] - last_hp
        last_hp, last_max_hp = own["hp"], own["max_hp"]
        for event in events:
            kind = event["type"]
            if kind in ("hit", "damaged"):
                if kind == "hit" and event["target"] >= 16:
                    continue
                bucket = result["damage_by_weapon" if kind == "hit" else "damage_received"]
                weapon = event["weapon"]
                bucket[weapon] = bucket.get(weapon, 0) + event["amount"]
            elif kind == "kill":
                kill_times.append(last_time)
                if event["target"] == active_finish:
                    confirmed_finishes += 1
                    active_finish = None
            elif kind == "took" and event["kind"] != "gun" and result["first_secondary_s"] is None:
                result["first_secondary_s"] = last_time
        label = record.get("action", {}).get("debug", {}).get("label", "")
        if label:
            mode, _, target = label.split(maxsplit=1)[0].partition(":")
            if target:
                has_modes = True
                current = int(target) if mode == "finish" and target.isdigit() else None
                if current is not None and active_finish != current:
                    attempts += 1
                active_finish = current
    if kill_times:
        result["first_kill_s"] = kill_times[0]
        result["kill_survival_5s"] = {
            "confirmed": sum(last_time - timestamp >= 5 for timestamp in kill_times),
            "observed_kills": len(kill_times),
        }
    place = result["placement"]
    result["placement_points"] = None if place is None else 16 - place
    if has_modes:
        result["finish_conversion"] = {"attempts": attempts, "confirmed_kills": confirmed_finishes}
    return result


def evaluate(paths):
    matches = {}
    for path in paths:
        if path.name in matches:
            raise ValueError(f"Duplicate match filename: {path.name}")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as source:
            matches[path.name] = summarize(json.loads(line) for line in source if line.strip())
    scores = [match["placement_points"] for match in matches.values()
              if match["placement_points"] is not None]
    return {"scored_matches": len(scores), "mean_points": mean(scores) if scores else None,
            "bottom_four": sum(score <= 3 for score in scores), "matches": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        report = evaluate(args.matches)
        if args.baseline:
            baseline = evaluate(args.baseline)
            pairs = []
            for name, match in report["matches"].items():
                ref = baseline["matches"].get(name)
                if ref and match["placement_points"] is not None and ref["placement_points"] is not None:
                    pairs.append(match["placement_points"] - ref["placement_points"])
            report["comparison"] = {"baseline_mean_points": baseline["mean_points"],
                                    "matched_filenames": len(pairs),
                                    "mean_matched_delta": mean(pairs) if pairs else None}
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
