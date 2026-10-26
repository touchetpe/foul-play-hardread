"""HardRead oracle injection for value-of-information experiments.

When enabled, each sampled search world replaces an opponent Pokemon's sampled
set with its TRUE set with probability p, independently per Pokemon and per
world. The evaluator-only team file must never be given to a production bot.

Environment variables (all optional; the oracle is off unless TEAM is set):
  HARDREAD_ORACLE_TEAM    path to the opponent's team in Showdown export format
  HARDREAD_ORACLE_P       probability in [0, 1] (default 1.0)
  HARDREAD_ORACLE_FIELDS  comma list from {all,item,ability,tera,moves,spread}
                          (default all)
  HARDREAD_ORACLE_SEED    optional RNG seed for reproducibility
"""
import logging
import os
import random

from fp.teams.team_converter import export_to_dict

logger = logging.getLogger(__name__)

_VALID_FIELDS = {"item", "ability", "tera", "moves", "spread"}
_EV_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


class Oracle:
    def __init__(self, team_path, p=1.0, fields=("all",), seed=None):
        with open(team_path) as f:
            team = export_to_dict(f.read())
        self.truth = {pkmn["species"]: pkmn for pkmn in team}
        self.p = float(p)
        fields = {f.strip() for f in fields if f.strip()}
        self.fields = _VALID_FIELDS if (not fields or "all" in fields) else fields
        unknown = self.fields - _VALID_FIELDS
        if unknown:
            raise ValueError(f"Unknown oracle fields: {sorted(unknown)}")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("HARDREAD_ORACLE_P must be in [0, 1]")
        self.rng = random.Random(seed)
        self.applied = 0
        self.skipped = 0
        self.missing = 0

    def apply(self, pkmn):
        """Overwrite selected fields of a sampled opponent Pokemon with the truth."""
        true_set = self.truth.get(pkmn.name) or self.truth.get(getattr(pkmn, "base_name", None))
        if true_set is None:
            self.missing += 1
            logger.warning(f"Oracle: no true set for {pkmn.name}")
            return False
        if self.rng.random() >= self.p:
            self.skipped += 1
            return False

        if "moves" in self.fields and true_set["moves"]:
            known = {mv.name: mv.current_pp for mv in pkmn.moves}
            pkmn.moves = []
            for mv in true_set["moves"]:
                pkmn.add_move(mv)
            for mv in pkmn.moves:
                if mv.name in known:
                    mv.current_pp = known[mv.name]
        if "item" in self.fields and true_set["item"] and pkmn.removed_item is None:
            pkmn.item = true_set["item"]
        if "ability" in self.fields and true_set["ability"]:
            pkmn.ability = true_set["ability"]
        if "tera" in self.fields and true_set["tera_type"] and not pkmn.terastallized:
            pkmn.tera_type = true_set["tera_type"]
        if "spread" in self.fields and true_set["nature"]:
            evs = ",".join(str(int(true_set["evs"].get(s) or 0)) for s in _EV_ORDER)
            pkmn.set_spread(true_set["nature"], evs)

        self.applied += 1
        logger.info(
            f"Oracle applied to {pkmn.name}: fields={sorted(self.fields)} "
            f"item={pkmn.item} ability={pkmn.ability} tera={pkmn.tera_type} "
            f"moves={[m.name for m in pkmn.moves]}"
        )
        return True


_ORACLE = None
_LOADED = False


def get_oracle():
    global _ORACLE, _LOADED
    if not _LOADED:
        _LOADED = True
        path = os.environ.get("HARDREAD_ORACLE_TEAM")
        if path:
            seed = os.environ.get("HARDREAD_ORACLE_SEED")
            _ORACLE = Oracle(
                path,
                p=os.environ.get("HARDREAD_ORACLE_P", "1.0"),
                fields=os.environ.get("HARDREAD_ORACLE_FIELDS", "all").split(","),
                seed=int(seed) if seed else None,
            )
            logger.info(
                f"Oracle enabled: p={_ORACLE.p} fields={sorted(_ORACLE.fields)} team={path}"
            )
    return _ORACLE
