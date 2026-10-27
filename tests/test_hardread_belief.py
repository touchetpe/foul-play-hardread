"""HardRead belief lever (fp/hardread_belief.py): off by default, shifts world sets toward the belief when on."""
import random

import pytest

from fp import hardread_belief as HB
from fp.battle.helpers import maximum_ev
from fp.battle.state import Pokemon
from fp.search.standard_battles import prepare_battles, sample_pokemon
from tests.test_search_standard_battles import make_predicted_set, make_standard_battle

try:
    np = HB._numpy()
except ImportError:  # the lever needs numpy; vanilla tests do not
    np = None


def make_battle():
    battle = make_standard_battle()
    battle.opponent.active = Pokemon("gengar", 100)
    battle.opponent.reserve = [Pokemon("azelf", 100)]
    common = dict(ability="cursedbody", nature="timid", evs=(0, 0, 0, maximum_ev(), 4, maximum_ev()))
    battle.mode.team_datasets.pkmn_sets = {
        "gengar": [
            make_predicted_set(["shadowball", "sludgebomb", "focusblast", "trick"], item="choicescarf", **common),
            make_predicted_set(["shadowball", "sludgebomb", "focusblast", "nastyplot"], item="lifeorb", **common),
        ],
        "azelf": [
            make_predicted_set(["psychic", "stealthrock", "explosion", "knockoff"], ability="levitate",
                               item="focussash", nature="jolly", evs=(0, maximum_ev(), 0, 0, 4, maximum_ev())),
        ],
    }
    return battle


class StubBelief:
    """A belief that puts 95% on Life Orb for every Pokemon's item, Cursed Body / Levitate as abilities, flat elsewhere."""
    worlds = False
    itos = {
        "item": ["<pad>", "<unk>", "lifeorb", "choicescarf", "focussash"],
        "ability": ["<pad>", "<unk>", "cursedbody", "levitate"],
        "tera": ["<pad>", "<unk>", "ghost", "psychic"],
        "spread": ["<pad>", "<unk>", "timid|spa+spe", "jolly|atk+spe"],
        "move": ["<pad>", "<unk>", "trick", "nastyplot"],
    }

    def __init__(self, fail=False):
        self.fail = fail
        self.stoi = {k: {t: i for i, t in enumerate(v)} for k, v in self.itos.items()}

    def idx(self, kind, token):
        return self.stoi[kind].get(token, 1)

    def forward(self, rows):
        if self.fail:
            raise RuntimeError("stub failure")
        B = len(rows)
        item = np.log(np.tile([1e-9, 1e-9, 0.95, 0.05, 1e-3], (B, 1)))
        ability = np.log(np.tile([1e-9, 1e-9, 0.99, 0.01], (B, 1)))
        flat = np.log(np.tile([1e-9, 1e-9, 0.5, 0.5], (B, 1)))
        return {"item": item, "ability": ability, "tera": flat, "nature": flat, "evclass": flat, "spread": flat,
                "moves": flat, "moves_logit": np.zeros((B, 4))}


@pytest.fixture(autouse=True)
def _reset():
    yield
    HB.configure(mode=None)
    HB._model = None
    HB._model_error = None


def lifeorb_share(n_decisions, worlds=4):
    hits = total = 0
    for _ in range(n_decisions):
        for b, _chance in prepare_battles(make_battle(), worlds):
            hits += b.opponent.active.item == "lifeorb"
            total += 1
    return hits / total


def test_off_by_default_and_never_called(monkeypatch):
    assert not HB.enabled()
    monkeypatch.setattr(HB, "pools", lambda *a, **k: pytest.fail("belief lever called while off"))
    random.seed(1)
    prepare_battles(make_battle(), 3)


def test_off_matches_upstream_random_stream():
    # upstream prepare_battles' sampling loop, written out: same draws in the same order
    random.seed(7)
    ours = [(b.opponent.active.item, tuple(m.name for m in b.opponent.active.moves))
            for b, _ in prepare_battles(make_battle(), 4)]
    state_ours = random.getstate()
    random.seed(7)
    battle = make_battle()
    ref = []
    from copy import deepcopy
    for _ in range(4):
        bc = deepcopy(battle)
        sample_pokemon(bc.opponent.active, battle.mode)
        for p in filter(lambda x: x.is_alive(), bc.opponent.reserve):
            sample_pokemon(p, battle.mode)
        bc.opponent.lock_moves()
        ref.append((bc.opponent.active.item, tuple(m.name for m in bc.opponent.active.moves)))
    assert ours == ref
    assert random.getstate() == state_ours


@pytest.mark.skipif(np is None, reason="numpy unavailable")
def test_resample_shifts_world_sets_toward_the_belief():
    random.seed(0)
    vanilla = lifeorb_share(60)
    HB.configure(mode="resample", model=StubBelief(), m=16, lam=1.0)
    random.seed(0)
    lever = lifeorb_share(60)
    assert 0.3 < vanilla < 0.7
    assert lever > 0.85
    # the reserve Pokemon keeps its only legal set, and legality is Foul Play's
    for b, _ in prepare_battles(make_battle(), 4):
        assert b.opponent.reserve[0].item == "focussash"
        assert len(b.opponent.active.moves) == 4
        assert b.opponent.active.ability == "cursedbody"


@pytest.mark.skipif(np is None, reason="numpy unavailable")
def test_lambda_zero_keeps_foul_plays_distribution():
    HB.configure(mode="resample", model=StubBelief(), m=16, lam=0.0)
    random.seed(0)
    assert 0.3 < lifeorb_share(60) < 0.7


@pytest.mark.skipif(np is None, reason="numpy unavailable")
def test_model_failure_falls_back_to_foul_play():
    HB.configure(mode="resample", model=StubBelief(fail=True), m=16, lam=1.0)
    before = HB.STATS["fallback_decisions"]
    random.seed(0)
    sampled = prepare_battles(make_battle(), 3)
    assert HB.STATS["fallback_decisions"] == before + 1
    for b, _ in sampled:
        assert b.opponent.active.item in ("lifeorb", "choicescarf")
        assert len(b.opponent.active.moves) == 4


@pytest.mark.skipif(np is None, reason="numpy unavailable")
def test_original_battle_untouched():
    HB.configure(mode="resample", model=StubBelief(), m=8, lam=1.0)
    battle = make_battle()
    random.seed(0)
    prepare_battles(battle, 2)
    assert battle.opponent.active.item == "unknownitem"
    assert battle.opponent.active.moves == []


@pytest.mark.skipif(np is None, reason="numpy unavailable")
def test_direct_mode_draws_belief_items_with_legal_sets():
    HB.configure(mode="direct", model=StubBelief(), m=16, lam=1.0)
    random.seed(0)
    share = lifeorb_share(30)
    assert share > 0.8
    for b, _ in prepare_battles(make_battle(), 4):
        assert len(b.opponent.active.moves) == 4
        assert b.opponent.active.item in ("lifeorb", "choicescarf")


def test_pool_cache_does_not_change_draws():
    from copy import deepcopy
    battle = make_battle()
    p = battle.opponent.active

    def draws():
        out = []
        for _ in range(12):
            q = deepcopy(p)
            sample_pokemon(q, battle.mode)
            out.append((q.item, q.ability, tuple(m.name for m in q.moves), q.nature, tuple(q.evs)))
        return out

    random.seed(5)
    plain = draws()
    random.seed(5)
    with HB._same_pokemon_cache(battle.mode):
        cached = draws()
    assert plain == cached
    # the instance attributes are gone again
    assert "get_all_remaining_sets" not in vars(battle.mode.team_datasets)
    assert "get_all_remaining_trait_combinations" not in vars(battle.mode.smogon_sets)
