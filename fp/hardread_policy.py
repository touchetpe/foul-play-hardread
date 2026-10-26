"""HardRead policy hook: let an external policy pick Foul Play's move instead of MCTS.

Off unless HARDREAD_POLICY is set; with it unset Foul Play is unchanged (the hooks in
fp/search/main.py and fp/websocket_client.py are no-ops). Foul Play still handles Showdown,
the battle state and the action formatting; only the choice comes from elsewhere.

HARDREAD_POLICY=llm   ask an OpenAI-compatible chat/completions endpoint (tools, tool_choice).

Observation parity with the SFT data (tools/dataset/extract.py + tools/sft/serialize.py):
  view     built by fp.hardread_decision_log._side/_pkmn on the same battle copy the search
           logs (after team-preview handling), JSON round-tripped like the decision log;
  legal    the MCTS root actions: union over the same sampled worlds (mode.search_params,
           mode.prepare_battles) of poke-engine's root side-one options, sorted, which is
           extract.search_targets()'s "legal";
  history  the battle's protocol lines received so far (captured in websocket_client),
           cut like extract.history_cuts (first decision of turn N ends at "|turn|N", team
           preview at "|teampreview"), then fp.hardread_history.canonical_history;
  turn, k  battle.turn (or 0) and the index of this decision within the turn.

Every failure (serializer error, HTTP error, timeout, no tool call, bad arguments, unknown
choice id) is counted and logged, and the bot then plays a clearly logged FALLBACK: the first
legal action. The arena counts every fallback as an illegal move. Server rejections of a
submitted choice ("|error|[Invalid choice]") are logged as separate events.

Environment:
  HARDREAD_POLICY              "llm" (unset: vanilla Foul Play)
  HARDREAD_LLM_URL             full endpoint URL, e.g. http://127.0.0.1:18045/v1/chat/completions
  HARDREAD_LLM_MODEL           "model" field (default "hardread")
  HARDREAD_LLM_MAX_TOKENS      default 256
  HARDREAD_LLM_TIMEOUT         seconds per request, default 20
  HARDREAD_LLM_TOOL_CHOICE     default "required"
  HARDREAD_LLM_EXTRA           JSON merged into the request body (e.g. chat_template_kwargs)
  HARDREAD_LLM_DEBUG_IDS=1     also send "x_hardread_choice_ids" (mock endpoints only)
  HARDREAD_SERIALIZER          path to serialize.py (default <repo>/tools/sft/serialize.py)
  HARDREAD_POLICY_LOG          JSONL path: one record per decision (+ rejection events)
  HARDREAD_POLICY_LOG_HISTORY=1  include the canonical history and request in each record
"""
import hashlib
import importlib.util
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from fp.hardread_history import canonical_history

logger = logging.getLogger(__name__)

_POLICY = os.environ.get("HARDREAD_POLICY") or None
_LOG_PATH = os.environ.get("HARDREAD_POLICY_LOG")
_LOG_HISTORY = os.environ.get("HARDREAD_POLICY_LOG_HISTORY") == "1"
POLICIES = ("llm",)

_lock = threading.Lock()
_history = defaultdict(list)  # battle_tag -> raw protocol lines received on our channel
_turn_count = defaultdict(lambda: defaultdict(int))  # battle_tag -> turn -> decisions so far
_serializer = None


def enabled():
    return _POLICY is not None


# ---------------------------------------------------------------- history capture
def record(message):
    """Called with every websocket message; keeps the lines of battle rooms."""
    if not message.startswith(">battle-"):
        return
    head, _, body = message.partition("\n")
    tag = head[1:].strip()
    lines = body.split("\n")
    with _lock:
        _history[tag].extend(lines)
    for line in lines:
        if line.startswith("|error|[Invalid choice]"):
            logger.warning("HardRead policy: server rejected our choice: {}".format(line))
            _write({"event": "server_rejection", "t": time.time(), "battle_tag": tag, "line": line})


def history_for_decision(raw, turn, k, team_preview):
    """Raw lines visible at this decision, cut exactly like extract.history_cuts."""
    if team_preview:
        marks = [i for i, l in enumerate(raw) if l.startswith("|teampreview")]
        return raw[: marks[0] + 1] if marks else list(raw)
    if k == 0:
        marks = [i for i, l in enumerate(raw) if l.strip() == "|turn|{}".format(turn)]
        if marks:
            return raw[: marks[-1] + 1]
    return list(raw)


# ---------------------------------------------------------------- legal actions and view
def legal_actions(battle):
    """Sorted union of the MCTS root side-one options over the search's sampled worlds."""
    from poke_engine import monte_carlo_tree_search
    from fp.search.poke_engine_helpers import battle_to_poke_engine_state

    num_battles, _ = battle.mode.search_params(battle)
    legal = set()
    for b, _chance in battle.mode.prepare_battles(battle, num_battles):
        state = battle_to_poke_engine_state(b)
        # The root is populated with every option before any iteration; 1 batch is enough.
        res = monte_carlo_tree_search(state, duration_ms=0, iterations=1, threads=1)
        legal.update(o.move_choice for o in res.side_one)
    return sorted(legal)


def build_view(battle):
    from fp.hardread_decision_log import _side

    view = {
        "user": _side(battle.user),
        "opponent": _side(battle.opponent),
        "weather": str(battle.weather) if battle.weather else None,
        "field": str(battle.field) if battle.field else None,
        "trick_room": battle.trick_room,
    }
    # Same JSON round trip as the decision log (json.dumps(..., default=str)).
    return json.loads(json.dumps(view, default=str))


def _load_serializer():
    global _serializer
    if _serializer is None:
        default = Path(__file__).resolve().parents[2] / "tools" / "sft" / "serialize.py"
        path = Path(os.environ.get("HARDREAD_SERIALIZER") or default)
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("hardread_serialize", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _serializer = mod
        logger.info("HardRead policy: serializer {}".format(path))
    return _serializer


# ---------------------------------------------------------------- LLM call
class PolicyFailure(Exception):
    def __init__(self, kind, detail="", info=None):
        super().__init__("{}: {}".format(kind, detail))
        self.kind = kind
        self.detail = str(detail)[:500]
        self.info = info or {}


def _llm_choose(example, legal):
    import requests

    url = os.environ.get("HARDREAD_LLM_URL")
    if not url:
        raise PolicyFailure("config", "HARDREAD_LLM_URL is not set")
    body = {
        "model": os.environ.get("HARDREAD_LLM_MODEL", "hardread"),
        "messages": example["messages"],
        "tools": example["tools"],
        "tool_choice": os.environ.get("HARDREAD_LLM_TOOL_CHOICE", "required"),
        "temperature": 0,
        "max_tokens": int(os.environ.get("HARDREAD_LLM_MAX_TOKENS", "256")),
    }
    if os.environ.get("HARDREAD_LLM_EXTRA"):
        body.update(json.loads(os.environ["HARDREAD_LLM_EXTRA"]))
    id_map = {str(k): v for k, v in example["id_map"].items()}
    if os.environ.get("HARDREAD_LLM_DEBUG_IDS") == "1":
        body["x_hardread_choice_ids"] = list(id_map)
    timeout = float(os.environ.get("HARDREAD_LLM_TIMEOUT", "20"))
    t0 = time.time()
    try:
        r = requests.post(url, json=body, timeout=timeout)
    except requests.exceptions.Timeout as e:
        raise PolicyFailure("timeout", e)
    except requests.exceptions.RequestException as e:
        raise PolicyFailure("http_error", e)
    info = {"llm_latency_s": round(time.time() - t0, 3)}
    if r.status_code != 200:
        raise PolicyFailure("http_error", "status {}: {}".format(r.status_code, r.text[:300]), info)
    try:
        resp = r.json()
        msg = resp["choices"][0]["message"]
    except Exception as e:
        raise PolicyFailure("bad_response", e)
    usage = resp.get("usage") or {}
    info.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                finish_reason=resp["choices"][0].get("finish_reason"))
    calls = msg.get("tool_calls") or []
    info["n_tool_calls"] = len(calls)
    if not calls:
        raise PolicyFailure("no_tool_call", (msg.get("content") or "")[:300], info)
    fn = calls[0].get("function") or {}
    tools = {t["function"]["name"]: t["function"] for t in example["tools"] if t.get("type") == "function"}
    if fn.get("name") not in tools:
        raise PolicyFailure("wrong_tool", fn.get("name"), info)
    try:
        args = fn.get("arguments")
        args = json.loads(args) if isinstance(args, str) else (args or {})
        required = tools[fn["name"]].get("parameters", {}).get("required") or list(
            tools[fn["name"]].get("parameters", {}).get("properties", {}))
        choice_id = args[required[0]]
    except Exception as e:
        raise PolicyFailure("bad_arguments", "{!r} ({})".format(fn.get("arguments"), e), info)
    info["choice_id"] = choice_id
    action = id_map.get(str(choice_id))
    if action is None or action not in legal:
        raise PolicyFailure("illegal_id", "choice_id={!r}".format(choice_id), info)
    return action, info


# ---------------------------------------------------------------- entry point
def choose(battle):
    """Called by fp.search.main.find_best_move (after team-preview handling) when enabled."""
    t0 = time.time()
    if _POLICY not in POLICIES:
        raise ValueError("HARDREAD_POLICY={!r}: unknown policy (known: {})".format(_POLICY, POLICIES))
    tag = battle.battle_tag
    turn = battle.turn or 0
    with _lock:
        k = _turn_count[tag][turn]
        _turn_count[tag][turn] += 1
        raw = list(_history.get(tag, []))
    player = battle.user.name
    legal = legal_actions(battle)
    t_legal = time.time() - t0
    view = build_view(battle)
    history = canonical_history(history_for_decision(raw, turn, k, battle.team_preview), player)
    rec = {"event": "decision", "t": t0, "policy": _POLICY, "battle_tag": tag, "player": player,
           "turn": turn, "k": k, "team_preview": battle.team_preview, "legal": legal,
           "n_history": len(history),
           "history_sha1": hashlib.sha1("\n".join(history).encode()).hexdigest()[:12],
           "legal_seconds": round(t_legal, 3)}
    action, failure = None, None
    example = None
    try:
        if not legal:
            raise PolicyFailure("no_legal_actions")
        try:
            example = _load_serializer().build_example(
                view, history, legal, player, turn, k, battle.team_preview, task="action")
        except Exception as e:
            raise PolicyFailure("serialize_error", repr(e))
        action, info = _llm_choose(example, legal)
        rec.update(info)
    except PolicyFailure as e:
        failure = e
        rec.update(e.info)
        rec.update(failure=e.kind, failure_detail=e.detail)
    if failure is not None:
        action = legal[0] if legal else None
        rec["fallback"] = True
        logger.warning("HardRead policy FALLBACK ({}): playing first legal action {!r}".format(failure, action))
    else:
        rec["fallback"] = False
    rec["action"] = action
    rec["latency_s"] = round(time.time() - t0, 3)
    if _LOG_HISTORY:
        rec["history"] = history
        rec["view"] = view
        if example is not None:
            rec["messages"] = example.get("messages")
    _write(rec)
    logger.info("HardRead policy: turn {} k {} action {!r} ({}s{})".format(
        turn, k, action, rec["latency_s"], ", FALLBACK " + failure.kind if failure else ""))
    if action is None:
        raise RuntimeError("HardRead policy: no legal action")
    return action


def _write(rec):
    if not _LOG_PATH:
        return
    with _lock:
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
