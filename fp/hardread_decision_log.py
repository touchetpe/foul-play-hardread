"""HardRead per-decision logging for self-play datasets.

Enabled when HARDREAD_DECISION_LOG is set to a file path; one JSON line is
appended per decision. Set HARDREAD_LOG_STATES=1 to also store the poke-engine
state string of every sampled world (large).

Records the bot's *own* view only: its public battle state, the sampled worlds
it searched, per-world search statistics and its choice. The opponent's true
team is joined later by the evaluator from the run's team assignments.
"""
import json
import os
import time

_PATH = os.environ.get("HARDREAD_DECISION_LOG")
_LOG_STATES = os.environ.get("HARDREAD_LOG_STATES") == "1"


def enabled():
    return bool(_PATH)


def _pkmn(p):
    if p is None:
        return None
    sr = getattr(p, "speed_range", None)
    return {
        "name": p.name,
        "hp": p.hp,
        "max_hp": p.max_hp,
        "status": p.status,
        "item": p.item,
        "ability": p.ability,
        "tera_type": p.tera_type,
        "terastallized": p.terastallized,
        "moves": [m.name for m in p.moves],
        "boosts": {k: v for k, v in dict(p.boosts).items() if v},
        "impossible_items": sorted(getattr(p, "impossible_items", []) or []),
        "speed_range": [sr.min, None if sr.max == float("inf") else sr.max] if sr else None,
        "nature": getattr(p, "nature", None),
        "evs": list(getattr(p, "evs", []) or []),
    }


def _side(battler):
    return {
        "active": _pkmn(battler.active),
        "reserve": [_pkmn(p) for p in battler.reserve],
        "side_conditions": {k: v for k, v in dict(battler.side_conditions).items() if v},
    }


def _options(options, total):
    return [
        {"m": o.move_choice, "v": o.visits, "s": round(o.total_score / o.visits, 4) if o.visits else None}
        for o in sorted(options, key=lambda o: -o.visits)
    ]


def log_decision(battle, sampled, mcts_results, choice, search_time_ms, states=None):
    if not _PATH:
        return
    worlds = []
    for i, ((b, chance), (res, _chance, _idx)) in enumerate(zip(sampled, mcts_results)):
        opp = [b.opponent.active] + list(b.opponent.reserve)
        w = {
            "chance": chance,
            "total_visits": res.total_visits,
            "opp_sets": [
                {"name": p.name, "item": p.item, "ability": p.ability, "tera_type": p.tera_type,
                 "moves": [m.name for m in p.moves], "nature": getattr(p, "nature", None),
                 "evs": list(getattr(p, "evs", []) or [])}
                for p in opp if p is not None
            ],
            "side_one": _options(res.side_one, res.total_visits),
            "side_two": _options(res.side_two, res.total_visits),
        }
        if _LOG_STATES and states is not None:
            w["state"] = states[i]
        worlds.append(w)
    rec = {
        "t": time.time(),
        "battle_tag": battle.battle_tag,
        "turn": battle.turn,
        "team_preview": battle.team_preview,
        "weather": str(battle.weather) if battle.weather else None,
        "field": str(battle.field) if battle.field else None,
        "trick_room": battle.trick_room,
        "user": _side(battle.user),
        "opponent": _side(battle.opponent),
        "search_time_ms": search_time_ms,
        "worlds": worlds,
        "choice": choice,
    }
    with open(_PATH, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
