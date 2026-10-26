"""Canonical public battle history, shared by HardRead training and serving.

Pure Python, no Foul Play imports, so the SFT serializer can import it too.

Two sources must give the same history for the same decision:
  * training: the server's saved battle log (logs/*.log.json "log"), cut at history_end by
    tools/dataset/extract.py. That log is the *omniscient* channel (Showdown's "replay log
    (exact damage)"): it shows BOTH sides' exact HP, e.g. "|-damage|p2a: Blissey|239/656".
  * serving: the protocol lines the bot's client receives on its player channel. Its own HP
    is exact; the opponent's is the HP Percentage Mod value, e.g. "|-damage|p2a: Blissey|37/100".

canonical_history(lines, player) maps both to the same list:
  1. drops lines that only exist on one side or carry no game information: client/chat/room
     lines (|init|, |title|, |j|, |c|, |raw|, |request|, |error|, ...), timestamps (|t:|),
     blank separators (|), timer notices (|inactive|, |inactiveoff|) and server hints (|-hint|);
  2. rewrites the opponent's exact HP "hp/maxhp" to Showdown's public percentage
     (sim/pokemon.ts getHealth: ceil(100*hp/maxhp), 99 if not full); idempotent on "pct/100";
  3. anonymises |player| lines to "|player|pN" (usernames and avatars differ per run).
Never feed the omniscient log to a model without this: it leaks the opponent's exact HP.
"""
import math
import re

DROP = frozenset((
    "", "t:", "init", "title", "j", "J", "join", "l", "L", "leave", "n", "N", "name",
    "c", "chat", "c:", "raw", "html", "uhtml", "uhtmlchange", "request", "sentchoice",
    "error", "bigerror", "deinit", "noinit", "expire", "badge", "rated", "tempnotify",
    "tempnotifyoff", "notify", "hidelines", "unlink", "debug", "seed", "split", "choice",
    "timer", "inactive", "inactiveoff", "-hint", "pm", "popup", "updatesearch",
    "updatechallenges", "queryresponse", "formats", "challstr", "updateuser", "customgroups",
))
HP_LINES = frozenset(("switch", "drag", "replace", "detailschange", "-damage", "-heal", "-sethp"))
_IDENT = re.compile(r"^(p[1-4])[a-z]?: ")
_HP = re.compile(r"^(\d+)/(\d+)(\s.*)?$")


def public_hp(hp, maxhp):
    """Showdown's HP Percentage Mod value for the non-owning side (sim/pokemon.ts getHealth)."""
    if hp <= 0:
        return 0
    pct = math.ceil(100 * hp / maxhp)
    if pct == 100 and hp < maxhp:
        pct = 99
    return pct


def line_type(line):
    if not line.startswith("|"):
        return ""
    return line.split("|", 2)[1]


def canonical_line(line, player):
    """Canonical form of one protocol line from `player`'s point of view, or None to drop it."""
    typ = line_type(line)
    if typ in DROP:
        return None
    if typ == "player":
        parts = line.split("|")
        return "|player|" + parts[2] if len(parts) > 2 else line
    if typ in HP_LINES:
        parts = line.split("|")
        side = None
        for i in range(2, len(parts)):
            m = _IDENT.match(parts[i])
            if m:
                side = m.group(1)
                continue
            h = _HP.match(parts[i])
            if h and side is not None and side != player:
                hp, maxhp, rest = int(h.group(1)), int(h.group(2)), h.group(3) or ""
                if maxhp != 100:
                    parts[i] = f"{public_hp(hp, maxhp)}/100{rest}"
        return "|".join(parts)
    return line


def canonical_history(lines, player):
    """Canonical public history (list of str) for `player` ("p1"/"p2")."""
    out = []
    for line in lines:
        c = canonical_line(line, player)
        if c is not None:
            out.append(c)
    return out
