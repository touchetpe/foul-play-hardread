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
# v3: Monte Carlo snapshot of Foul Play's own belief (its set sampler, after its in-place
# damage-based pruning) for every visible opponent Pokemon. 0 disables it.
_BELIEF_SAMPLES = int(os.environ.get("HARDREAD_LOG_BELIEF_SAMPLES", "0") or 0)


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
        # v2: every other field Foul Play's set sampler reads, so its belief can be rebuilt offline.
        "level": getattr(p, "level", None),
        "removed_item": getattr(p, "removed_item", None),
        "impossible_abilities": sorted(getattr(p, "impossible_abilities", []) or []),
        "can_have_choice_item": getattr(p, "can_have_choice_item", None),
        "can_terastallize": getattr(p, "can_terastallize", None),
        "original_ability": getattr(p, "original_ability", None),
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


def _belief_snapshot(battle, n):
    """Histogram of n draws of Foul Play's set sampler per visible opponent Pokemon.

    The global random state is saved and restored, so play is unchanged.
    """
    import random
    from copy import deepcopy
    from fp.search.standard_battles import sample_pokemon

    state = random.getstate()
    out = {}
    try:
        opp = [battle.opponent.active] + list(battle.opponent.reserve)
        for p in opp:
            if p is None or p.hp <= 0 or p.name in out:
                continue
            h = {"n": 0, "item": {}, "ability": {}, "tera": {}, "spread": {}, "moves": {}}
            for _ in range(n):
                q = deepcopy(p)
                try:
                    sample_pokemon(q, battle.mode)
                except Exception:
                    continue
                h["n"] += 1
                for key, val in (("item", q.item), ("ability", q.ability), ("tera", q.tera_type),
                                 ("spread", f"{q.nature}|{','.join(str(e) for e in (q.evs or []))}")):
                    val = str(val)
                    h[key][val] = h[key].get(val, 0) + 1
                for m in q.moves:
                    h["moves"][m.name] = h["moves"].get(m.name, 0) + 1
            out[p.name] = h
    finally:
        random.setstate(state)
    return out


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
    belief = None
    if _BELIEF_SAMPLES > 0:
        t_b = time.time()
        belief = _belief_snapshot(battle, _BELIEF_SAMPLES)
        belief = {"samples": _BELIEF_SAMPLES, "ms": round(1000 * (time.time() - t_b)), "pkmn": belief}
    rec = {
        "v": 3 if belief is not None else 2,
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
    if belief is not None:
        rec["fp_belief"] = belief
    with open(_PATH, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
