"""Generate traces with a KNOWN number of stale-state actions, so the auditor can
be checked against ground truth before it is pointed at anything real.

This is the same move as verifying the machine before running a model, applied
one level up: the tool that reports the number is itself checked against a
number that was planted.

  python make_traces.py            writes three traces and prints what is in them
"""
import json, random, sys
sys.path.insert(0, "../..")

FILES = ["report.txt", "data.csv", "notes.md"]
ORDERS = ["A17", "B42"]
USERS = ["alice", "bob"]


def write(path, events):
    with open(path, "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def clean(n, seed=0):
    """A well-built agent: it re-reads before every irreversible call, and only
    acts when the read says it may. Planted violations: zero."""
    rng = random.Random(seed)
    ev, exists = [], {f: True for f in FILES}
    for _ in range(n):
        f = rng.choice(FILES)
        if rng.random() < 0.4:
            ev.append({"tool": "create_file", "args": {"path": f}}); exists[f] = True
        else:
            ev.append({"tool": "read_file", "args": {"path": f}})
            if exists[f]:
                ev.append({"tool": "delete_file", "args": {"path": f}}); exists[f] = False
    return ev, 0


def stale(n, planted=7, seed=0):
    """An agent working from memory: it deletes files it deleted earlier, and
    charges orders it already charged. Planted violations: `planted`."""
    rng = random.Random(seed)
    ev, exists, charged = [], {f: True for f in FILES}, {o: False for o in ORDERS}
    left = planted
    for _ in range(n):
        if left and rng.random() < 0.3:
            f = rng.choice(FILES)
            if exists[f]:
                ev.append({"tool": "delete_file", "args": {"path": f}}); exists[f] = False
            ev.append({"tool": "delete_file", "args": {"path": f}})   # already gone
            left -= 1
        elif rng.random() < 0.3:
            o = rng.choice(ORDERS)
            if not charged[o]:
                ev.append({"tool": "charge_order", "args": {"order_id": o}}); charged[o] = True
            else:
                ev.append({"tool": "get_order", "args": {"order_id": o}})
        else:
            f = rng.choice(FILES)
            ev.append({"tool": "create_file", "args": {"path": f}}); exists[f] = True
    return ev, planted - left


def mixed(n, planted=5, seed=0):
    """Both failure modes: some calls on stale belief, some where the agent DID
    re-read first and acted anyway. The second kind must not be counted as
    stale belief, and the auditor reports them separately."""
    rng = random.Random(seed)
    ev, exists = [], {f: True for f in FILES}
    left, rereads = planted, 4
    while left or rereads:
        f = rng.choice(FILES)
        if not exists[f]:
            ev.append({"tool": "create_file", "args": {"path": f}}); exists[f] = True
        ev.append({"tool": "delete_file", "args": {"path": f}}); exists[f] = False
        if left and rng.random() < 0.6:
            ev.append({"tool": "delete_file", "args": {"path": f}})          # stale
            left -= 1
        elif rereads:
            ev.append({"tool": "read_file", "args": {"path": f}})            # re-read
            ev.append({"tool": "delete_file", "args": {"path": f}})          # then acted
            rereads -= 1
    return ev, planted


if __name__ == "__main__":
    for name, fn in (("clean.jsonl", clean), ("stale.jsonl", stale), ("mixed.jsonl", mixed)):
        ev, planted = fn(60)
        write(name, ev)
        print(f"{name:<14} {len(ev):>4} events, planted stale-state actions: {planted}")
