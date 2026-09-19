"""Exact equivalence checking. This is the load-bearing function of the whole
library: it decides whether two automata agree on EVERY string, by breadth-first
search over reachable pairs of states, with no length cutoff.

Accuracy on sampled strings cannot substitute. In the work this library comes
from, a bounded enumeration to length 11 reported 6/6 correct and disagreed with
true equivalence on 22 of 48 deliberately corrupted targets.
"""
from collections import deque
import numpy as np

__all__ = ["equivalent", "reachable", "minimal_size"]


def equivalent(delta_a, out_a, delta_b, out_b, start_a=0, start_b=0):
    """True if the two DFAs produce the same output on every input string.

    delta_*: int array (alphabet_size, n_states); delta[x, s] is the next state.
    out_*:   int array (n_states,); the output symbol emitted in that state.
    """
    delta_a, out_a = np.asarray(delta_a), np.asarray(out_a)
    delta_b, out_b = np.asarray(delta_b), np.asarray(out_b)
    A = delta_a.shape[0]
    if delta_b.shape[0] != A:
        raise ValueError("alphabet sizes differ")
    seen = {(int(start_a), int(start_b))}
    q = deque(seen)
    while q:
        a, b = q.popleft()
        for x in range(A):
            a2, b2 = int(delta_a[x, a]), int(delta_b[x, b])
            if int(out_a[a2]) != int(out_b[b2]):
                return False
            if (a2, b2) not in seen:
                seen.add((a2, b2)); q.append((a2, b2))
    return True


def reachable(delta, start=0):
    """Indices of states reachable from `start`."""
    delta = np.asarray(delta); A = delta.shape[0]
    seen = {int(start)}; q = deque(seen)
    while q:
        s = q.popleft()
        for x in range(A):
            t = int(delta[x, s])
            if t not in seen:
                seen.add(t); q.append(t)
    return sorted(seen)


def minimal_size(delta, out):
    """Number of states in the minimal equivalent DFA (Moore refinement)."""
    delta, out = np.asarray(delta), np.asarray(out)
    K = delta.shape[1]; part = out.copy()
    while True:
        sig = [tuple([int(part[i])] + [int(part[delta[x, i]]) for x in range(delta.shape[0])])
               for i in range(K)]
        m = {}; new = np.empty(K, np.int64)
        for i, s in enumerate(sig):
            if s not in m: m[s] = len(m)
            new[i] = m[s]
        if len(m) == len(set(part.tolist())):
            return len(set(part.tolist()))
        part = new
