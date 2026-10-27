"""HardRead belief lever: Foul Play's world sampling guided by the learned belief (tools/belief, v1).

Off unless HARDREAD_LEVER_BELIEF is set. With it unset, `enabled()` is False, nothing here runs (numpy is not even
imported) and prepare_battles is upstream Foul Play, random-number stream included.

HARDREAD_LEVER_BELIEF=resample   importance resampling (lever A). For every visible opponent Pokemon that Foul Play
        samples (the active one and the alive reserves), draw a pool of M candidate sets with Foul Play's OWN sampler
        (sample_pokemon on deep copies, so its damage-pruned TeamDatasets/SmogonSets and every legality check apply),
        score them with the belief network on the same public view, and give each world the set drawn from the pool
        with weight  w_i ∝ p_belief(set_i)^λ / q̂(set_i)^λ,  q̂ = smoothed frequency of that exact set among the M draws
        (sampling-importance-resampling: as M grows the world's set follows p_belief restricted to the sets Foul Play
        can produce; λ = 0 is Foul Play's own distribution). One pool per decision is shared by all its worlds.
HARDREAD_LEVER_BELIEF=direct     lever B: sets drawn from the belief's marginals (item, ability, tera, spread class,
        moves), completed and checked with Foul Play's set logic; Foul Play's sampler is the fallback.

p_belief(set) = p(item) p(ability) p(tera) p(spread class) Π_{hidden moves m in set} p_m/(1-p_m), using only the
attributes still hidden in the view (revealed ones are equal across candidates); categorical probabilities are
floored at 1e-4.

Environment:
  HARDREAD_LEVER_BELIEF     resample | direct (unset: vanilla Foul Play)
  HARDREAD_BELIEF_CKPT      the exported belief (.npz from tools/belief/export_npz.py)
  HARDREAD_BELIEF_M         pool size per Pokemon per decision (default 16; at least the number of worlds)
  HARDREAD_BELIEF_BUDGET_MS time budget for the draws beyond the one-per-world that vanilla makes anyway (default 60;
                            extra draws go round-robin over the Pokemon and stop when it is spent)
  HARDREAD_BELIEF_LAMBDA    tempering λ in [0, 1] (default 1)
  HARDREAD_BELIEF_FP_INPUT  1: also feed the pool's histogram (Foul Play's own draws) to a worlds model, as the
                            S-A "fp_belief as worlds input" variant (needs a checkpoint trained with --worlds 1;
                            HARDREAD_BELIEF_WORLDS_CKPT, if set, is then used instead of HARDREAD_BELIEF_CKPT)
  HARDREAD_BELIEF_LOG       JSONL path: one record per decision (timings, pool sizes, effective sample sizes)

Any failure (missing numpy or checkpoint, model error, unexpected state) is logged and the decision, or the single
Pokemon, falls back to Foul Play's own sampling. numpy: imported lazily; if this venv has none, the same-ABI user
site-packages is appended (the review lane's trick); Azure nodes install numpy in setup_node.sh.
"""
import json
import logging
import math
import os
import random
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

logger = logging.getLogger(__name__)

MODE = os.environ.get("HARDREAD_LEVER_BELIEF") or None
CKPT = os.environ.get("HARDREAD_BELIEF_CKPT") or None
WORLDS_CKPT = os.environ.get("HARDREAD_BELIEF_WORLDS_CKPT") or None
M = int(os.environ.get("HARDREAD_BELIEF_M") or 16)
BUDGET_MS = float(os.environ.get("HARDREAD_BELIEF_BUDGET_MS") or 60)
LAMBDA = float(os.environ.get("HARDREAD_BELIEF_LAMBDA") or 1.0)
FP_INPUT = os.environ.get("HARDREAD_BELIEF_FP_INPUT") == "1"
if FP_INPUT and WORLDS_CKPT:  # the fp-input variant needs a model trained with worlds
    CKPT = WORLDS_CKPT
LOG_PATH = os.environ.get("HARDREAD_BELIEF_LOG") or None
CHOICES = ("resample", "direct")
if MODE is not None and MODE not in CHOICES:
    raise ValueError(f"HARDREAD_LEVER_BELIEF must be one of {CHOICES}, got {MODE!r}")

P_FLOOR = 1e-4
Q_SMOOTH = 0.5
STATS_KEYS = ("decisions", "fallback_decisions", "fallback_pkmn", "direct_fallback_pkmn")
STATS = {k: 0 for k in STATS_KEYS}
_model = None
_model_error = None


def enabled():
    return MODE is not None


def configure(mode=None, model=None, m=None, lam=None, fp_input=None, log_path=None, budget_ms=None):
    """Tests and tools: set the lever in-process (bypasses the environment)."""
    global MODE, _model, M, LAMBDA, FP_INPUT, LOG_PATH, _model_error, BUDGET_MS
    if budget_ms is not None:
        BUDGET_MS = budget_ms
    MODE = mode
    if model is not None:
        _model, _model_error = model, None
    if m is not None:
        M = m
    if lam is not None:
        LAMBDA = lam
    if fp_input is not None:
        FP_INPUT = fp_input
    if log_path is not None:
        LOG_PATH = log_path


# ---------------------------------------------------------------- numpy and the shared feature helpers
_np = None


def _numpy():
    global _np
    if _np is None:
        try:
            import numpy
        except ImportError:
            import site
            extra = [getattr(site, "getusersitepackages", lambda: "")(),
                     str(Path.home() / "Library/Python/{}.{}/lib/python/site-packages".format(*sys.version_info[:2]))]
            for p in extra:
                if p and p not in sys.path:
                    sys.path.append(p)
            import numpy
        _np = numpy
    return _np


_ID = re.compile(r"[^a-z0-9]")
NEUTRAL = {"hardy", "docile", "serious", "bashful", "quirky"}
STATS6 = ("hp", "atk", "def", "spa", "spd", "spe")
NATURES = {
    "lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
    "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
    "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
    "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa"),
    "serious": (None, None),
}
STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")


def toid(s):
    return None if s is None else _ID.sub("", str(s).lower())


def canon_nature(n):
    n = toid(n) or "serious"
    return "serious" if n in NEUTRAL else n


def ev_class(evs):
    evs = [int(v) for v in evs]
    order = sorted(range(6), key=lambda i: (-evs[i], i))
    top = sorted(i for i in order[:2] if evs[i] > 0)
    return "+".join(STATS6[i] for i in top) or "none"


def spread_class(nature, evs):
    return "{}|{}".format(canon_nature(nature), ev_class(evs))


def _dex():
    from fp.data import pokedex
    return pokedex


def dex_entry(name):
    dex = _dex()
    n = toid(name)
    if n in dex:
        return dex[n]
    cands = [k for k in dex if n and n.startswith(k)]
    return dex[max(cands, key=len)] if cands else None


def species_key(name):
    e = dex_entry(name)
    if e and e.get("battleOnly"):
        b = e["battleOnly"]
        return toid(b if isinstance(b, str) else b[0])
    return toid(name)


def speed_stat(name, nature, evs, level=100, iv=31):
    e = dex_entry(name)
    if not e:
        return None
    base = e["baseStats"]["speed"]
    raw = (2 * base + iv + int(evs[5]) // 4) * level // 100 + 5
    plus, minus = NATURES.get(canon_nature(nature), (None, None))
    mult = 1.1 if plus == "spe" else 0.9 if minus == "spe" else 1.0
    return int(raw * mult)


def view_of(p):
    """The decision log's public view of one Pokemon (fp.hardread_decision_log._pkmn), JSON round-tripped."""
    from fp.hardread_decision_log import _pkmn
    return json.loads(json.dumps(_pkmn(p), default=str))


def view_row(view, ctx, turn, team_preview, active, world_dists=None):
    """Everything the network reads about one opponent Pokemon (same fields as a tools/belief example)."""
    return {"view": view, "ctx": ctx, "turn": turn, "team_preview": bool(team_preview), "active": bool(active),
            "species": toid(view["name"]), "skey": species_key(view["name"]), "world_dists": world_dists}


# ---------------------------------------------------------------- the network (numpy port of tools/belief/model.py)
MAXLEN = {"teammates": 6, "own_team": 6, "moves": 6, "imp_items": 24, "imp_abilities": 16}
WORLD_HEADS = ("item", "ability", "tera", "nature", "evclass", "spread", "moves")
CAT_HEADS = ("item", "ability", "tera", "nature", "evclass", "spread")
VOCAB_OF = {"item": "item", "ability": "ability", "tera": "tera", "nature": "nature", "evclass": "evclass",
            "spread": "spread", "moves": "move"}


class BeliefModel:
    def __init__(self, path):
        np = _numpy()
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.W = {k: z[k] for k in z.files if k != "meta"}
        self.itos = self.meta["vocab"]
        self.stoi = {k: {t: i for i, t in enumerate(v)} for k, v in self.itos.items()}
        self.drop = set(self.meta.get("drop_inputs") or [])
        self.worlds = bool(self.meta.get("worlds"))
        self.alpha = float(self.meta.get("world_alpha", 0.05))
        self.neg = float(self.meta.get("neg", -30.0))
        self.choice_ids = list(self.meta["choice_ids"])
        self.legal = self.W["legal_ability"].astype(bool)
        self._erf = np.vectorize(math.erf, otypes=[np.float64])
        self._spe = {}

    def idx(self, kind, token):
        return self.stoi[kind].get(token, 1)

    def _pad(self, ids, n):
        ids = list(ids)[:n]
        return ids + [0] * (n - len(ids))

    def featurize(self, rows):
        np = _numpy()
        B = len(rows)
        f = {k: np.zeros((B, n), np.int64) for k, n in MAXLEN.items()}
        for k in ("species", "own_active", "item_view", "removed_item", "ability_view", "tera_view"):
            f[k] = np.zeros(B, np.int64)
        dense = np.zeros((B, 24), np.float32)
        for i, r in enumerate(rows):
            v, x = r["view"], r["ctx"]
            f["species"][i] = self.idx("species", r["skey"])
            f["teammates"][i] = self._pad([self.idx("species", toid(n)) for n in x.get("teammates") or []], 6)
            f["own_team"][i] = self._pad([self.idx("species", toid(n)) for n in x.get("own_team") or []], 6)
            f["own_active"][i] = self.idx("species", toid(x["own_active"])) if x.get("own_active") else 0
            f["moves"][i] = self._pad([self.idx("move", toid(m)) for m in v.get("moves") or []], 6)
            f["item_view"][i] = self.idx("item", toid(v.get("item")) if v.get("item") else "none")
            f["removed_item"][i] = self.idx("item", toid(v["removed_item"])) if v.get("removed_item") else 0
            f["ability_view"][i] = self.idx("ability", toid(v["ability"])) if v.get("ability") else self.idx("ability", "?")
            f["tera_view"][i] = (self.idx("tera", toid(v["tera_type"])) if v.get("terastallized") and v.get("tera_type")
                                 else self.idx("tera", "?"))
            f["imp_items"][i] = self._pad([self.idx("item", t) for t in v.get("impossible_items") or []], 24)
            f["imp_abilities"][i] = self._pad([self.idx("ability", t) for t in v.get("impossible_abilities") or []], 16)
            sp = r["species"]
            if sp not in self._spe:
                lo_s = speed_stat(sp, "brave", [0] * 6) or 100
                hi_s = speed_stat(sp, "jolly", [0, 0, 0, 0, 0, 252]) or 300
                self._spe[sp] = (lo_s, hi_s)
            lo_s, hi_s = self._spe[sp]
            sr = v.get("speed_range") or [0, None]
            lo = sr[0] or 0
            hi = sr[1]
            span = max(1.0, hi_s - lo_s)
            hp = (v.get("hp") or 0) / (v.get("max_hp") or 1)
            st = v.get("status")
            d = [hp, float(r["active"]), float(bool(v.get("terastallized"))), float(v.get("can_have_choice_item") is not False),
                 lo / 600.0, float(lo > 0), (hi or 0) / 600.0, float(hi is not None),
                 max(-1.0, min(3.0, (lo - lo_s) / span)) if lo > 0 else 0.0,
                 max(-1.0, min(3.0, (hi - lo_s) / span)) if hi is not None else 0.0,
                 min(r["turn"] or 0, 60) / 60.0, float(r["team_preview"]),
                 (x.get("n_opp_fainted") or 0) / 6.0, (x.get("n_own_fainted") or 0) / 6.0,
                 float(r["species"] != r["skey"]), len(v.get("moves") or []) / 4.0, float(bool(v.get("removed_item"))),
                 (v.get("level") or 100) / 100.0] + [float(st == s) for s in STATUSES]
            dense[i, :len(d)] = d
        for k in self.drop:
            if k in f:
                f[k][:] = 0
        f["dense"] = dense
        return f

    def _bag(self, name, idx):
        np = _numpy()
        emb = self.W[name][idx]
        cnt = np.maximum(1, (idx > 0).sum(1)).astype(np.float32)
        return emb.sum(1) / cnt[:, None]

    def _gelu(self, x):
        return 0.5 * x * (1.0 + self._erf(x / math.sqrt(2.0)))

    def forward(self, rows):
        """Per-head log-probabilities [B, V] (moves: log-sigmoid), plus raw move logits under 'moves_logit'."""
        np = _numpy()
        with np.errstate(all="ignore"):  # macOS Accelerate raises spurious FP flags in float64 matmul
            return self._forward(np, rows)

    def _forward(self, np, rows):
        f = self.featurize(rows)
        W = self.W
        sp = f["species"]
        h = np.concatenate([
            W["species.weight"][sp], self._bag("teammates.emb.weight", f["teammates"]), W["own_active.weight"][f["own_active"]],
            self._bag("own_team.emb.weight", f["own_team"]), self._bag("moves.emb.weight", f["moves"]),
            W["item_view.weight"][f["item_view"]], W["removed_item.weight"][f["removed_item"]],
            W["ability_view.weight"][f["ability_view"]], W["tera_view.weight"][f["tera_view"]],
            self._bag("imp_items.emb.weight", f["imp_items"]), self._bag("imp_abilities.emb.weight", f["imp_abilities"]),
            f["dense"]], axis=1).astype(np.float64)
        z = self._gelu(h @ W["trunk.0.weight"].T.astype(np.float64) + W["trunk.0.bias"])
        z = self._gelu(z @ W["trunk.3.weight"].T.astype(np.float64) + W["trunk.3.bias"])
        out = {}
        for k in WORLD_HEADS:
            o = z @ W["heads.{}.weight".format(k)].T + W["heads.{}.bias".format(k)]
            if "prior_" + k in W:
                o = o + W["prior_scale." + k][0] * W["prior_" + k][sp]
            if self.worlds:
                dist = np.zeros_like(o)
                for i, r in enumerate(rows):
                    for t, pr in ((r.get("world_dists") or {}).get(k) or {}).items():
                        dist[i, self.idx(VOCAB_OF[k], t)] += pr
                a = self.alpha
                if k == "moves":
                    dist = np.clip(dist, 0, 1)
                    o = o + W["world_scale.moves"][0] * np.log((dist + a) / (1 - dist + a))
                else:
                    o = o + W["world_scale." + k][0] * np.log(dist + a)
            out[k] = o
        # hard constraints, as tools/belief/model.py constrain()
        li = out["item"]
        for i in range(len(rows)):
            li[i, f["imp_items"][i]] = self.neg
            if f["dense"][i, 3] < 0.5:
                li[i, self.choice_ids] = self.neg
        li[:, 0] = self.neg
        la = np.where(self.legal[sp], out["ability"], self.neg)
        for i in range(len(rows)):
            la[i, f["imp_abilities"][i]] = self.neg
        la[:, 0] = self.neg
        out["ability"] = la
        res = {}
        for k in CAT_HEADS:
            o = out[k]
            m = o.max(1, keepdims=True)
            res[k] = o - m - np.log(np.exp(o - m).sum(1, keepdims=True))
        x = out["moves"]
        res["moves"] = -np.logaddexp(0.0, -x)
        res["moves_logit"] = x
        return res


def _get_model():
    global _model, _model_error
    if _model is None and _model_error is None:
        try:
            if not CKPT:
                raise RuntimeError("HARDREAD_BELIEF_CKPT is not set")
            _model = BeliefModel(CKPT)
            logger.info(f"HardRead belief lever: loaded {CKPT}")
        except Exception as e:  # the lever then falls back on every decision
            _model_error = repr(e)
            logger.warning(f"HardRead belief lever: cannot load the belief model, using Foul Play's sampling: {e!r}")
    if _model is None:
        raise RuntimeError("belief model unavailable: {}".format(_model_error))
    return _model


# ---------------------------------------------------------------- the lever
SET_FIELDS = ("moves", "ability", "item", "nature", "evs", "stats", "max_hp", "hp", "tera_type")


def _set_key(p):
    return (str(p.item), str(p.ability), str(p.tera_type), tuple(sorted(m.name for m in p.moves)),
            str(p.nature), tuple(int(e) for e in (p.evs or [])))


def _hidden(view):
    return {"item": view.get("item") == "unknownitem" and not view.get("removed_item"),
            "ability": view.get("ability") is None, "tera": not view.get("terastallized"),
            "revealed": {toid(m) for m in view.get("moves") or []}}


def _score(model, lp, i, hid, p):
    """log p_belief(set) up to a constant, from row i of the forward pass."""
    np = _numpy()
    floor = math.log(P_FLOOR)
    s = 0.0
    if hid["item"]:
        it = "none" if p.item in (None, "", "None") else toid(p.item)
        s += max(floor, float(lp["item"][i, model.idx("item", it)]))
    if hid["ability"]:
        s += max(floor, float(lp["ability"][i, model.idx("ability", toid(p.ability) if p.ability else "?")]))
    if hid["tera"]:
        s += max(floor, float(lp["tera"][i, model.idx("tera", toid(p.tera_type) if p.tera_type else "?")]))
    s += max(floor, float(lp["spread"][i, model.idx("spread", spread_class(p.nature, p.evs or [0] * 6))]))
    for m in {toid(mv.name) for mv in p.moves} - hid["revealed"]:
        s += float(np.clip(lp["moves_logit"][i, model.idx("move", m)], -12.0, 12.0))
    return s


def _pool_hist(cands):
    """The pool as a fp_belief-style distribution per head (the worlds input of a --worlds model)."""
    n = len(cands)
    d = {k: {} for k in WORLD_HEADS}

    def add(k, t, w):
        d[k][t] = d[k].get(t, 0.0) + w

    for q in cands:
        add("item", "none" if q.item in (None, "", "None") else toid(q.item), 1 / n)
        add("ability", toid(q.ability) if q.ability else "?", 1 / n)
        add("tera", toid(q.tera_type) if q.tera_type else "?", 1 / n)
        evs = list(q.evs or [0] * 6)
        add("nature", canon_nature(q.nature), 1 / n)
        add("evclass", ev_class(evs), 1 / n)
        add("spread", spread_class(q.nature, evs), 1 / n)
        for m in {toid(mv.name) for mv in q.moves}:
            add("moves", m, 1 / n)
    return d


def _slots(battle):
    opp = battle.opponent
    out = []
    if opp.active is not None:
        out.append(("active", opp.active))
    out += [(i, p) for i, p in enumerate(opp.reserve) if p.is_alive()]
    return out


def _ctx(battle):
    opp, user = battle.opponent, battle.user
    opp_all = ([opp.active] if opp.active else []) + list(opp.reserve)
    own_all = ([user.active] if user.active else []) + list(user.reserve)
    return {"n_opp_fainted": sum(1 for p in opp_all if (p.hp or 0) <= 0),
            "n_own_fainted": sum(1 for p in own_all if (p.hp or 0) <= 0),
            "own_active": user.active.name if user.active else None,
            "own_team": [p.name for p in own_all], "names": [p.name for p in opp_all]}


class _same_pokemon_cache:
    """While one Pokemon's M candidates are drawn, every draw starts from an identical deep copy, so Foul Play's
    dataset filters (TeamDatasets.get_all_remaining_sets, SmogonSets.get_all_remaining_trait_combinations) return the
    same lists: compute them once. The lists are only read (sample_pokemon deep-copies the chosen set), the random
    stream is unchanged, and the instance attributes are removed again on exit."""
    NAMES = (("team_datasets", "get_all_remaining_sets"), ("smogon_sets", "get_all_remaining_trait_combinations"))

    def __init__(self, mode, boxes=None):
        self.mode = mode
        self.patched = []
        self.boxes = {} if boxes is None else boxes  # persists across entries for the same Pokemon

    def __enter__(self):
        for holder, name in self.NAMES:
            obj = getattr(self.mode, holder, None)
            if obj is None or name in vars(obj):
                continue
            orig = getattr(obj, name)
            box = self.boxes.setdefault(name, [])

            def cached(pkmn, _orig=orig, _box=box):
                if not _box:
                    _box.append(_orig(pkmn))
                return _box[0]

            setattr(obj, name, cached)
            self.patched.append((obj, name))
        return self

    def __exit__(self, *exc):
        for obj, name in self.patched:
            try:
                delattr(obj, name)
            except AttributeError:
                pass
        return False


def pools(battle, num_battles, sample_fn):
    """Candidate pools and resampling weights for this decision, or None (caller samples like upstream)."""
    if MODE is None:
        return None
    t0 = time.perf_counter()
    STATS["decisions"] += 1
    try:
        model = _get_model()
        slots = _slots(battle)
        boxes = {key: {} for key, _ in slots}
        cands = {key: [] for key, _ in slots}

        cost = {key: 0.0 for key, _ in slots}
        worst = {key: 0.0 for key, _ in slots}

        def draw(key, p):
            t = time.perf_counter()
            with _same_pokemon_cache(battle.mode, boxes[key]):
                q = deepcopy(p)
                sample_fn(q, battle.mode)
            cands[key].append(q)
            dt = time.perf_counter() - t
            cost[key] += dt
            worst[key] = max(worst[key], dt)

        # the draws vanilla Foul Play would make anyway (one per world and Pokemon) ...
        for key, p in slots:
            for _ in range(num_battles):
                draw(key, p)
        # ... then extra draws, round-robin over the Pokemon, up to M each, within the time budget. Foul Play's
        # moveset sampler spins for 10-300 ms per draw on some views (its Smogon move loop with tiny probabilities),
        # so a Pokemon is only drawn again while twice its slowest draw so far fits in the remaining budget.
        t_extra = time.perf_counter()
        budget = BUDGET_MS / 1000.0
        for _ in range(max(0, M - num_battles)):
            for key, p in slots:
                left = budget - (time.perf_counter() - t_extra)
                if left <= 0 or 2 * max(worst[key], cost[key] / max(1, len(cands[key]))) > left:
                    continue
                draw(key, p)
            if time.perf_counter() - t_extra > budget:
                break
        m = max(len(v) for v in cands.values()) if cands else 0
        t1 = time.perf_counter()
        c = _ctx(battle)
        rows, views = [], []
        for key, p in slots:
            v = view_of(p)
            views.append(v)
            ctx = {"teammates": [n for n in c["names"] if n != p.name], "own_active": c["own_active"],
                   "own_team": c["own_team"], "n_opp_fainted": c["n_opp_fainted"], "n_own_fainted": c["n_own_fainted"]}
            wd = _pool_hist(cands[key]) if (FP_INPUT and model.worlds) else None
            rows.append(view_row(v, ctx, battle.turn, battle.team_preview, key == "active", world_dists=wd))
        lp = model.forward(rows)
        t2 = time.perf_counter()
        out = {"mode": MODE, "slots": {}, "m": m}
        ess = {}
        for i, (key, p) in enumerate(slots):
            hid = _hidden(views[i])
            lst = cands[key]
            keys = [_set_key(q) for q in lst]
            count = {}
            for k in keys:
                count[k] = count.get(k, 0) + 1
            u = len(count)
            n = len(lst)
            if MODE == "direct":
                choices = _direct(model, lp, i, hid, p, lst, battle.mode, max(M, num_battles))
                out["slots"][key] = (choices, [1.0] * len(choices))
                ess[str(key)] = len(choices)
                continue
            logw = []
            for q, k in zip(lst, keys):
                qhat = (count[k] + Q_SMOOTH) / (n + Q_SMOOTH * u)
                logw.append(LAMBDA * (_score(model, lp, i, hid, q) - math.log(qhat)))
            mx = max(logw)
            w = [math.exp(x - mx) for x in logw]
            tot = sum(w)
            w = [x / tot for x in w]
            out["slots"][key] = (lst, w)
            ess[str(key)] = round(1.0 / sum(x * x for x in w), 2)
        t3 = time.perf_counter()
        out["timing_ms"] = {"draw": round(1000 * (t1 - t0), 2), "extra_draws": round(1000 * (t1 - t_extra), 2),
                            "infer": round(1000 * (t2 - t1), 2),
                            "weights": round(1000 * (t3 - t2), 2), "total": round(1000 * (t3 - t0), 2)}
        if LOG_PATH:
            try:
                with open(LOG_PATH, "a") as fh:
                    fh.write(json.dumps({"t": time.time(), "battle_tag": battle.battle_tag, "turn": battle.turn,
                                         "team_preview": battle.team_preview, "mode": MODE, "worlds": num_battles,
                                         "m": m, "pkmn": len(slots), "ess": ess,
                                         "draws": {str(k): len(v) for k, v in cands.items()}, "stats": dict(STATS),
                                         "unique": {str(k): len(set(_set_key(q) for q in v)) for k, v in cands.items()},
                                         "timing_ms": out["timing_ms"]}) + "\n")
            except OSError:
                pass
        return out
    except Exception as e:
        STATS["fallback_decisions"] += 1
        logger.warning(f"HardRead belief lever failed, using Foul Play's own sampling for this decision: {e!r}")
        return None


def apply_world(battle_copy, pool, sample_fn, mode):
    """Give one world's opponent Pokemon their sets drawn from the pools (Foul Play's sampler on any failure)."""
    opp = battle_copy.opponent
    for key, (lst, w) in pool["slots"].items():
        target = opp.active if key == "active" else opp.reserve[key]
        try:
            src = random.choices(lst, weights=w)[0]
            if src.name != target.name:
                raise RuntimeError("pool/world mismatch: {} vs {}".format(src.name, target.name))
            for attr in SET_FIELDS:
                setattr(target, attr, deepcopy(getattr(src, attr)))
        except Exception as e:
            STATS["fallback_pkmn"] += 1
            logger.warning(f"HardRead belief lever: {e!r}; sampling {target.name} with Foul Play's sampler")
            sample_fn(target, mode)


# ---------------------------------------------------------------- lever B (direct sampling)
def _direct(model, lp, i, hid, p, pool, mode, m, tries=8):
    """m sets drawn from the belief marginals and completed/checked with Foul Play's set logic.

    item, ability, tera and the spread class come from the belief (if hidden); the concrete nature/EVs are taken from a
    pool candidate of that class (else the belief's class is dropped); hidden moves are drawn without replacement in
    proportion to the belief's P(move in set) among moves Foul Play's pool contains (so they are legal for the species).
    A draw is kept if Foul Play's own checks accept it (PokemonSet.set_makes_sense on the view's Pokemon and
    PredictedPokemonSet.set_makes_logical_sense); otherwise a pool candidate is used."""
    np = _numpy()
    from fp.data.sets import PokemonMoveset, PokemonSet, PredictedPokemonSet
    from fp.search.helpers import populate_pkmn_from_set

    def draw(head, vocab, allowed=None):
        row = np.exp(lp[head][i])
        ids = [j for j in range(2, len(row)) if allowed is None or model.itos[vocab][j] in allowed]
        if not ids:
            return None
        pr = row[ids]
        if pr.sum() <= 0:
            return None
        return model.itos[vocab][random.choices(ids, weights=pr.tolist())[0]]

    pool_moves = sorted({toid(mv.name) for q in pool for mv in q.moves} - hid["revealed"])
    by_class = {}
    for q in pool:
        by_class.setdefault(spread_class(q.nature, q.evs or [0] * 6), []).append(q)
    out = []
    for _ in range(m):
        made = None
        for _t in range(tries):
            base = random.choice(pool)
            item = draw("item", "item") if hid["item"] else base.item
            ability = draw("ability", "ability") if hid["ability"] else base.ability
            tera = draw("tera", "tera") if hid["tera"] else base.tera_type
            cls = draw("spread", "spread", allowed=set(by_class))
            sb = random.choice(by_class[cls]) if cls in by_class else base
            known = [mv.name for mv in p.moves]
            moves = list(known)
            cand = [mm for mm in pool_moves if mm not in moves]
            while len(moves) < 4 and cand:
                pr = [math.exp(float(np.clip(lp["moves_logit"][i, model.idx("move", mm)], -12, 12))) for mm in cand]
                pick = random.choices(cand, weights=pr)[0]
                moves.append(pick)
                cand.remove(pick)
            if not item or not ability:
                continue
            pset = PokemonSet(ability=ability, item=item, nature=sb.nature, evs=tuple(sb.evs), count=1,
                              tera_type=tera if tera != "?" else None)
            pred = PredictedPokemonSet(pkmn_set=pset, pkmn_moveset=PokemonMoveset(moves=tuple(moves)))
            try:
                ok = pset.set_makes_sense(p) and pred.set_makes_logical_sense()
            except Exception:
                ok = False
            if not ok:
                continue
            q = deepcopy(p)
            populate_pkmn_from_set(q, pred, source="hardread-belief-direct")
            made = q
            break
        if made is None:
            STATS["direct_fallback_pkmn"] += 1
            made = random.choice(pool)
        out.append(made)
    return out


def describe():
    return {"mode": MODE, "ckpt": CKPT, "m": M, "lambda": LAMBDA, "fp_input": FP_INPUT, "budget_ms": BUDGET_MS,
            "stats": dict(STATS)}
