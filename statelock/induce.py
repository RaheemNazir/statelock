"""Induce an exact automaton from (string, label) pairs, by search.

The learner never sees a state. It needs an oracle it can query for the label of
a string it chooses -- `label_fn(word) -> int` -- which is the same access a
person has when they can run the system they are modelling on inputs of their
choosing.

Construction: a cover set holding one representative prefix per discovered
state, closed under the alphabet; a suffix set grown by consistency repair when
two prefixes share a row but their extensions do not. Both corrections were
forced by verification failures, not chosen for elegance:
  * sampling prefixes does not close the table (64k sampled prefixes still left
    transitions undefined, and the verifier refused);
  * a fixed suffix set under-separates (K = 149 against a minimal 152, with no
    unresolved transitions -- a wrong merge that looks nearly right).
"""
import numpy as np

__all__ = ["induce"]


def induce(alphabet_size, label_fn, suffixes=None, max_rounds=200, max_states=100000):
    """Return (delta, out, start, info).

    alphabet_size: number of input symbols, which are 0..alphabet_size-1.
    label_fn: callable taking a tuple of symbols, returning an int label.
    suffixes: optional starting suffix set (list of tuples). Defaults to the
        empty string plus every single symbol; more are added as needed.
    """
    V = alphabet_size
    S = list(suffixes) if suffixes else [()] + [(a,) for a in range(V)]
    P = [()]
    cache = {}

    def lab(w):
        if w not in cache: cache[w] = int(label_fn(w))
        return cache[w]

    def row(p): return tuple(lab(p + s) for s in S)

    rounds = added = 0
    reps = {}; sid = {}; cover = {}
    while rounds < max_rounds:
        rounds += 1
        e1 = {p + (a,) for p in P for a in range(V)}
        e2 = {q + (b,) for q in e1 for b in range(V)}
        cand = sorted(set(P) | e1 | e2, key=lambda w: (len(w), w))
        rows = {p: row(p) for p in cand}
        reps = {}; sid = {}; cover = {}
        for p in cand:
            r = rows[p]
            if r not in reps:
                reps[r] = len(reps); cover[reps[r]] = p
            sid[p] = reps[r]
            if len(reps) > max_states:
                raise RuntimeError("state count exceeded max_states; the target "
                                   "may not be finite-state at this resolution")
        bad = None
        bystate = {}
        for p in cand: bystate.setdefault(sid[p], []).append(p)
        for _, ps in bystate.items():
            if len(ps) < 2: continue
            p0 = ps[0]
            for q in ps[1:]:
                for a in range(V):
                    u, w = p0 + (a,), q + (a,)
                    if u in rows and w in rows and rows[u] != rows[w]:
                        j = next(i for i in range(len(S)) if rows[u][i] != rows[w][i])
                        bad = (a,) + S[j]; break
                if bad: break
            if bad: break
        if bad is not None and bad not in S:
            S.append(bad); added += 1; continue
        newP = sorted(set(cover.values()), key=lambda w: (len(w), w))
        if set(newP) == set(P): break
        P = newP

    K = len(reps)
    delta = np.zeros((V, K), np.int64); unresolved = 0
    for i in range(K):
        p = cover[i]
        for a in range(V):
            q = p + (a,)
            if q in sid: delta[a, i] = sid[q]
            else: unresolved += 1
    out = np.array([lab(cover[i]) for i in range(K)], np.int64)
    info = {"states": K, "suffixes": len(S), "suffixes_added": added,
            "rounds": rounds, "unresolved_transitions": unresolved,
            "queries": len(cache)}
    return delta, out, sid[()], info
