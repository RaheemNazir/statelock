"""NL-CONNECTIVITY with the entity count as a parameter, so difficulty can be
graded: minimal state count grows fast with N."""
import numpy as np
def build(N):
    V = N + 3; LINKS, DOT, RESET = N, N+1, N+2
    def canon(p):
        m = {}; out = []
        for x in p:
            if x not in m: m[x] = len(m)
            out.append(m[x])
        return tuple(out)
    def step(s, tok):
        p, pend = s
        if tok in (LINKS, DOT): return (p, None if tok == DOT else pend)
        if tok == RESET: return (canon(tuple(range(N))), None)
        if pend is None: return (p, tok)
        a, b = p[pend], p[tok]
        return (canon(tuple(a if x == b else x for x in p)), None)
    s0 = (canon(tuple(range(N))), None); idx = {s0: 0}; order = [s0]; i = 0
    while i < len(order):
        s = order[i]; i += 1
        for t in range(V):
            s2 = step(s, t)
            if s2 not in idx: idx[s2] = len(order); order.append(s2)
    K = len(order); T = np.zeros((V, K), np.int64)
    for s, si in idx.items():
        for t in range(V): T[t, si] = idx[step(s, t)]
    lab = np.array([1 if s[0][0] == s[0][1] else 0 for s in order], np.int64)
    return dict(N=N, V=V, K=K, T=T, lab=lab, s0=0, LINKS=LINKS, DOT=DOT, RESET=RESET)

def minimal_count(T, lab):
    K = T.shape[1]; part = lab.copy()
    while True:
        sig = [tuple([int(part[i])] + [int(part[T[a, i]]) for a in range(T.shape[0])])
               for i in range(K)]
        m = {}; new = np.empty(K, np.int64)
        for i, s in enumerate(sig):
            if s not in m: m[s] = len(m)
            new[i] = m[s]
        if len(m) == len(set(part.tolist())): return len(set(part.tolist()))
        part = new

def gen_prose(M, n, L, rng, p_reset=0.12):
    N, LINKS, DOT, RESET = M['N'], M['LINKS'], M['DOT'], M['RESET']
    X = np.empty((n, L), np.int64)
    for i in range(n):
        w = []
        while len(w) < L:
            if rng.random() < p_reset: w += [RESET, DOT]
            else:
                w += [int(rng.integers(0, N)), LINKS, int(rng.integers(0, N)), DOT]
        X[i] = w[:L]
    return X

def scan(M, X):
    T, lab = M['T'], M['lab']
    n, L = X.shape; s = np.full(n, M['s0'], np.int64); st = np.empty((n, L), np.int64)
    for t in range(L): s = T[X[:, t], s]; st[:, t] = s
    return st, lab[st]

if __name__ == "__main__":
    for N in (3, 4, 5):
        M = build(N)
        print({"entities": N, "vocab": M['V'], "raw_states": M['K'],
               "minimal_states": minimal_count(M['T'], M['lab'])})
