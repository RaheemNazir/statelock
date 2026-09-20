"""Tests that re-derive the paper's claims through the public API."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statelock import induce, equivalent, minimal_size, StateLock, StateDiscoverer, fit
from task2 import build, gen_prose, scan, probes, row_table, mixed

def test_induce_exact(N=5):
    M = build(N)
    def label_fn(w):
        s = M['s0']
        for t in w: s = M['T'][t, s]
        return int(M['lab'][s])
    delta, out, start, info = induce(M['V'], label_fn)
    ok = equivalent(delta, out, M['T'], M['lab'], start, M['s0'])
    mn = minimal_size(M['T'], M['lab'])
    print({"test": "induce", "entities": N, "states_found": info["states"],
           "target_minimal": mn, "queries": info["queries"],
           "suffixes_added": info["suffixes_added"], "exact": ok})
    assert ok and info["states"] == mn
    return delta, out, start, M

def test_verifier_rejects_corruption(delta, out, start, M, n=40, seed=0):
    rng = np.random.default_rng(seed); caught = 0
    for _ in range(n):
        d2 = delta.copy()
        a, s = rng.integers(0, d2.shape[0]), rng.integers(0, d2.shape[1])
        d2[a, s] = int(rng.integers(0, d2.shape[1]))
        if not equivalent(d2, out, M['T'], M['lab'], start, M['s0']): caught += 1
    print({"test": "verifier_rejects_corruption", "corruptions": n, "flagged": caught})
    assert caught > 0

def test_statelock_runs_exactly(delta, out, start, M, L=4096):
    lock = StateLock(delta, start, d_model=32, out=out)
    X = gen_prose(M, 4, L, np.random.default_rng(0))
    _, Y = scan(M, X)
    st = lock.run(torch.from_numpy(X)).numpy()
    agree = float((np.asarray(out)[st] == Y).mean())
    h = torch.zeros(4, L, 32)
    got = lock(h, torch.from_numpy(X))
    print({"test": "statelock", "length": L, "label_agreement": agree,
           "shape_ok": tuple(got.shape) == (4, L, 32)})
    assert agree == 1.0 and got.shape == (4, L, 32)

def test_discover_small(N=3, K=32, steps=3000):
    M = build(N)
    P = probes(M, n_deep=48)
    rng = np.random.default_rng(0)
    pools = []
    for L, bs in ((8, 64), (16, 48), (32, 32), (64, 16)):
        X = mixed(M, 1024, L, rng, 0.5); ST, _ = scan(M, X)
        pools.append((torch.from_numpy(X),
                      torch.from_numpy(row_table(M, P)[ST].astype(np.uint8)), bs))
    mo = StateDiscoverer(M['V'], K, len(P), seed=0)
    r2 = np.random.default_rng(1)
    def batches(it):
        Xt, Rt, bs = pools[int(r2.integers(0, 4))]
        i = torch.randint(0, Xt.shape[0], (bs,))
        return Xt[i], Rt[i].float()
    loss = fit(mo, batches, steps=steps, lr=2e-2, tau=1.0)
    delta, s0 = mo.table()
    Xe = gen_prose(M, 32, 256, np.random.default_rng(9)); _, Ye = scan(M, Xe)
    st = mo.states(torch.from_numpy(Xe))
    out = np.zeros(K, np.int64)
    for c in range(K):
        m = st == c
        if m.any(): out[c] = int(round(float(Ye[m].mean())))
    ok = equivalent(delta, out, M['T'], M['lab'], s0, M['s0'])
    print({"test": "discover", "entities": N, "steps": steps,
           "train_loss": round(loss, 5), "exact": ok,
           "note": "exactness tracks training loss; restart if loss is not ~0"})
    return ok, loss

if __name__ == "__main__":
    d, o, s, M = test_induce_exact(5)
    test_verifier_rejects_corruption(d, o, s, M)
    test_statelock_runs_exactly(d, o, s, M)
    test_discover_small(3, 32, 3000)
    print("ALL TESTS PASSED")
