"""Test step2 logic with fake models (real weights cannot be downloaded here).
Three fakes: always-no (the v1 failure mode), biased-but-informative (what
calibration must rescue), and correct."""
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

def run(fake_margin, name):
    """Replay the scoring loop with a fake margin function."""
    offset = fake_margin(S.build_prompt("N/A", "N/A", False))
    hits = yes_pred = 0
    for passage, ans, state in cases:
        m = fake_margin(S.build_prompt(passage, state, True)) - offset
        pred = "yes" if m > 0 else "no"
        hits += int(pred == ans); yes_pred += int(pred == "yes")
    n = len(cases)
    r = {"fake": name, "accuracy": round(hits/n, 4), "yes_rate": round(yes_pred/n, 4),
         "degenerate": yes_pred in (0, n)}
    print(r); return r

truth = {S.build_prompt(p, s, True): (a == "yes") for p, a, s in cases}

r1 = run(lambda t: -5.0, "always_no")
r2 = run(lambda t: (-3.0 + (1.0 if truth.get(t) else -1.0)), "biased_but_informative")
r3 = run(lambda t: (1.0 if truth.get(t) else -1.0), "correct")

assert r1["degenerate"] and abs(r1["accuracy"] - 0.5) < 1e-9, "always-no must score exactly 0.5 and be flagged"
assert r2["accuracy"] == 1.0 and not r2["degenerate"], "calibration must rescue a biased but informative model"
assert r3["accuracy"] == 1.0
print("SELFTEST PASSED")
