"""Consistency checker for docs/_consistency_map.yaml.

Reads the central fact -> source -> consumers map and, for each fact of
type "single", checks each consumer *only where it actually talks about
the fact's topic* (a regex, matched case-insensitively). Within the text
window around each mention, the consumer is consistent if it matches at
least one of the "accept" patterns (equivalent correct forms are all
fine -- the check validates the fact, not one specific wording) and does
NOT match any of the "wrong" patterns (an explicit known-incorrect form,
when one is worth naming, e.g. a stale count).

A consumer that never mentions the topic at all is not checked -- it is
simply not a consumer of that fact at that point in the document, not a
violation.

Facts of type "multi" are skipped on purpose: the name legitimately has
more than one value depending on measurement scope (e.g. calibration
bias on the full test set vs. on half of it, for the leak-free
calibration experiment) and are not comparable as a single expected
value.

Two severities, two outcomes:
    high -> printed under "[HIGH]" and, in --gate mode, makes the script
            exit 1 (fails a CI job).
    low  -> printed under "[LOW]" and never fails the run; these are
            known, accepted, non-blocking gaps (e.g. a historical note
            in a roadmap doc that was never updated after a later fact
            changed).

Modes:
    --report  (default) : print findings, always exit 0.
    --gate               : print findings, exit 1 if any HIGH finding exists.

Not wired into CI yet -- run manually, or add a step that calls this with
--gate once the map has proven itself over a few real edits.
"""
import re
import sys
import pathlib

import yaml

MAP = pathlib.Path("docs/_consistency_map.yaml")
CONTEXT_CHARS = 150  # characters of context on each side of a topic mention


def load():
    return yaml.safe_load(MAP.read_text(encoding="utf-8"))["facts"]


def check():
    high, low = [], []
    for f in load():
        if f.get("type") == "multi":
            continue
        topic = f.get("topic")
        accept = f.get("accept", [])
        wrong = f.get("wrong", [])
        if not topic:
            continue
        for c in f.get("consumers", []):
            p = pathlib.Path(c)
            if not p.exists():
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(topic, txt, re.IGNORECASE):
                window = txt[max(0, m.start() - CONTEXT_CHARS): m.end() + CONTEXT_CHARS]
                is_wrong = any(re.search(w, window, re.IGNORECASE) for w in wrong)
                is_right = any(re.search(a, window, re.IGNORECASE) for a in accept)
                if is_wrong or not is_right:
                    target = high if f.get("severity") == "high" else low
                    target.append(f"{f['name']}: {c} — mentions the topic but value diverges")
                    break  # one flagged mention per (fact, consumer) is enough
    return high, low


if __name__ == "__main__":
    high, low = check()
    if high:
        print("[HIGH — quebra consistencia, vira task]")
        for h in high:
            print("  ", h)
    if low:
        print("[LOW — acumulador]")
        for l in low:
            print("  ", l)
    if not high and not low:
        print("consistencia OK")

    gate = "--gate" in sys.argv
    sys.exit(1 if (gate and high) else 0)
