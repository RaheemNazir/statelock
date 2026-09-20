"""Self-test for step2 v4 (real weights cannot be downloaded here)."""
import random, sys
sys.path.insert(0, '/home/claude/s2')
import step2_real_model as S

rng = random.Random(0)
cases = S.balanced_cases(16, 40, rng)
ys = sum(1 for _, a, _, _ in cases if a == "yes")
print({"check": "balance", "yes": ys, "no": len(cases) - ys})
assert ys == 40 and len(cases) == 80

p, a, st, wr = cases[0]
pl = S.build_prompt(p, st, wr, "plain")
ws = S.build_prompt(p, st, wr, "with_state")
wg = S.build_prompt(p, st, wr, "wrong_state")
print({"check": "conditions", "plain_has_state": "Known groups" in pl,
       "with_state_has_state": "Known groups" in ws,
       "wrong_state_has_state": "Known groups" in wg,
       "wrong_differs_from_true": ws != wg})
assert "Known groups" not in pl and "Known groups" in ws and "Known groups" in wg

print({"check": "auc", "perfect": S.auc([1,2,3,4],[0,0,1,1]),
       "reversed": S.auc([4,3,2,1],[0,0,1,1]), "tied": S.auc([1,1,1,1],[0,0,1,1])})
assert S.auc([1,2,3,4],[0,0,1,1]) == 1.0 and S.auc([1,1,1,1],[0,0,1,1]) == 0.5

truth = {S.build_prompt(p, s, w, "with_state"): (a == "yes") for p, a, s, w in cases}

def run(fake, name):
    scores = [fake(S.build_prompt(p, s, w, "with_state")) for p, a, s, w in cases]
    labels = [1 if a == "yes" else 0 for _, a, _, _ in cases]
    lo, hi = S.bootstrap_ci(scores, labels, n_boot=300)
    r = {"fake": name, "auc": round(S.auc(scores, labels), 4), "ci95": [lo, hi]}
    print(r); return r

import random as _r
_rr = _r.Random(1)
r1 = run(lambda t: 5.0, "always_yes_no_info")
r2 = run(lambda t: 5.0 + (1.0 if truth.get(t) else -1.0), "always_yes_but_informative")
r3 = run(lambda t: (1.0 if truth.get(t) else -1.0) + _rr.gauss(0, 1.0), "noisy_informative")

assert r1["auc"] == 0.5 and r1["ci95"][0] == 0.5 == r1["ci95"][1], "no-info must have a degenerate interval at 0.5"
assert r2["auc"] == 1.0
assert r3["ci95"][0] > 0.5, "an informative model must have an interval clear of chance"
print("SELFTEST PASSED")
