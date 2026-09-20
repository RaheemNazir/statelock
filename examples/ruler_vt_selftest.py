"""Self-test the variable-tracking harness with fake models."""
import random, re, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ruler_vt as R

rng = random.Random(0)
ctx, q, gold, state, wrong = R.make_sample(2000, rng)

# the gold set must actually be derivable from the context
assigns = re.findall(r"VAR (X\d+) = (\w+)", ctx)
val = re.search(r"value (\d+)", q).group(1)
direct = {a for a, b in assigns if b == val}
resolved = set(direct)
for _ in range(10):
    for a, b in assigns:
        if b in resolved: resolved.add(a)
print({"check": "gold_derivable", "gold": gold, "recomputed": sorted(resolved),
       "match": sorted(resolved) == gold})
assert sorted(resolved) == gold, "the gold answer must follow from the context"

print({"check": "length", "target_tokens": 2000, "words": len(ctx.split())})
assert 1000 < len(ctx.split()) < 2200

pl = R.build_prompt(ctx, q, state, wrong, "plain")
ws = R.build_prompt(ctx, q, state, wrong, "with_state")
wg = R.build_prompt(ctx, q, state, wrong, "wrong_state")
print({"check": "conditions", "plain_leaks": "Resolved" in pl,
       "with_state_has": "Resolved" in ws, "wrong_differs": ws != wg,
       "wrong_same_shape": wg.count("Resolved") == ws.count("Resolved")})
assert "Resolved" not in pl and "Resolved" in ws and ws != wg

# scoring behaves
print({"check": "score_perfect", "s": R.score(", ".join(gold), gold)})
print({"check": "score_empty", "s": R.score("I don't know.", gold)})
print({"check": "score_wrongset", "s": R.score("X99, X98", gold)})
assert R.score(", ".join(gold), gold) == (1.0, 1.0, 1.0)
assert R.score("I don't know.", gold)[0] == 0.0

# a fake that copies the state sentence should score 1.0 on with_state and
# near 0 on wrong_state -- which is what makes the control meaningful
def fake_copy(prompt):
    m = re.search(r"Resolved bindings for \d+: ([^.]*)\.", prompt)
    return m.group(1) if m else ""
for cond, expect in (("with_state", 1.0), ("wrong_state", 0.0)):
    p = R.build_prompt(ctx, q, state, wrong, cond)
    r, _, _ = R.score(fake_copy(p), gold)
    print({"fake": "copies_state", "condition": cond, "recall": r})
    assert abs(r - expect) < 1e-9
print("SELFTEST PASSED")
