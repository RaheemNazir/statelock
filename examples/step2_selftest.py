"""Test step2 v3 logic with fakes (real weights cannot be downloaded here)."""
import random, sys
sys.path.insert(0, '/home/claude/s2')
import step2_real_model as S

rng = random.Random(0)
cases = S.balanced_cases(16, 40, rng)
ys = sum(1 for _, a, _ in cases if a == "yes")
print({"check": "balance", "yes": ys, "no": len(cases) - ys})
assert ys == 40 and len(cases) == 80

p, a, st = cases[0]
prompt = S.build_prompt(p, st, True)
print({"check": "prompt", "shots": prompt.count("Answer:") - 1,
       "state_included": "Known groups" in prompt})
assert prompt.count("Answer:") - 1 == 4 and "Known groups" in prompt
assert "Known groups" not in S.build_prompt(p, st, False)

# AUC sanity
print({"check": "auc_perfect", "auc": S.auc([1, 2, 3, 4], [0, 0, 1, 1])})
print({"check": "auc_reversed", "auc": S.auc([4, 3, 2, 1], [0, 0, 1, 1])})
print({"check": "auc_all_tied", "auc": S.auc([1, 1, 1, 1], [0, 0, 1, 1])})
assert S.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
assert S.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
assert S.auc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5

truth = {S.build_prompt(p, s, True): (a == "yes") for p, a, s in cases}

def run(fake, name):
    scores, labels = [], []
    for passage, ans, state in cases:
        scores.append(fake(S.build_prompt(passage, state, True)))
        labels.append(1 if ans == "yes" else 0)
    r = {"fake": name, "auc": round(S.auc(scores, labels), 4),
         "acc_at_median": round(S.acc_at_median(scores, labels), 4),
         "yes_rate_at_zero": round(sum(s > 0 for s in scores)/len(scores), 4)}
    print(r); return r

import random as _r
_rr = _r.Random(1)
r1 = run(lambda t: 5.0, "always_yes_no_info")
r2 = run(lambda t: 5.0 + (1.0 if truth.get(t) else -1.0), "always_yes_but_informative")
r3 = run(lambda t: (1.0 if truth.get(t) else -1.0) + _rr.gauss(0, 1.0), "noisy_informative")

assert r1["auc"] == 0.5, "a constant score must read as no information"
assert r2["auc"] == 1.0 and r2["yes_rate_at_zero"] == 1.0, \
    "signal must be visible even when the model answers yes to everything"
assert 0.6 < r3["auc"] < 1.0
print("SELFTEST PASSED")
