"""HardRead engine levers for Foul Play's search, all off by default.

With no HARDREAD_LEVER_* variable set, every function here returns None (or its
input unchanged) and Foul Play behaves exactly as upstream. Levers are tuned on
development set A and gated on development set B (PREREGISTRATION.md).

HARDREAD_LEVER_SELECT   final move selection across sampled worlds
    fp      upstream: sample among moves with >= 75% of the top visit share (default)
    argmax  play the move with the highest chance-weighted visit share
    value   among moves with >= 75% of the top visit share, play the highest
            chance-weighted mean search score
HARDREAD_LEVER_WORLDS   worlds vs search time per world, at equal total compute
    fp           upstream (default)
    double-half  twice the worlds at half the time each
    half-double  half the worlds (never fewer than the process count) at twice the time
"""
import logging
import os

logger = logging.getLogger(__name__)

SELECT = os.environ.get("HARDREAD_LEVER_SELECT", "fp")
WORLDS = os.environ.get("HARDREAD_LEVER_WORLDS", "fp")
_SELECT_CHOICES = ("fp", "argmax", "value")
_WORLDS_CHOICES = ("fp", "double-half", "half-double")
if SELECT not in _SELECT_CHOICES:
    raise ValueError(f"HARDREAD_LEVER_SELECT must be one of {_SELECT_CHOICES}, got {SELECT!r}")
if WORLDS not in _WORLDS_CHOICES:
    raise ValueError(f"HARDREAD_LEVER_WORLDS must be one of {_WORLDS_CHOICES}, got {WORLDS!r}")


def describe():
    return {"select": SELECT, "worlds": WORLDS}


def search_params(num_battles, search_time_ms, parallelism):
    """Return (num_battles, search_time_ms) after the worlds lever."""
    if WORLDS == "double-half":
        return num_battles * 2, max(1, search_time_ms // 2)
    if WORLDS == "half-double":
        fewer = max(parallelism, num_battles // 2)
        return fewer, int(search_time_ms * num_battles / fewer)
    return num_battles, search_time_ms


def aggregate(mcts_results):
    """Chance-weighted visit share and mean score per root move across worlds."""
    share, score_num, score_den = {}, {}, {}
    for res, chance, _index in mcts_results:
        total = res.total_visits or 1
        for o in res.side_one:
            share[o.move_choice] = share.get(o.move_choice, 0.0) + chance * o.visits / total
            score_num[o.move_choice] = score_num.get(o.move_choice, 0.0) + chance * o.total_score
            score_den[o.move_choice] = score_den.get(o.move_choice, 0.0) + chance * o.visits
    mean_score = {m: (score_num[m] / score_den[m]) if score_den[m] else 0.0 for m in share}
    return share, mean_score


def select_move(mcts_results):
    """Return the lever's choice, or None to keep Foul Play's own selection."""
    if SELECT == "fp":
        return None
    share, mean_score = aggregate(mcts_results)
    if SELECT == "argmax":
        choice = max(share, key=lambda m: (share[m], mean_score[m]))
    else:  # value
        top = max(share.values())
        candidates = [m for m in share if share[m] >= 0.75 * top]
        choice = max(candidates, key=lambda m: (mean_score[m], share[m]))
    logger.info(f"HardRead lever select={SELECT}: {choice} "
                f"(share {share[choice]:.3f}, score {mean_score[choice]:.3f})")
    return choice
