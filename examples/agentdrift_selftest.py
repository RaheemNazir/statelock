"""Self-test AgentDrift without any model."""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentdrift as A  # sits beside this file

for sf in A.SURFACES:
    delta, order, idx = A.build_machine(sf)
    ok, checked = A.verify_machine(sf, delta, order, idx)
    print({"surface": sf["name"], "states": len(order), "events": len(sf["events"]),
           "transitions_checked": checked, "exhaustive_agreement": ok})
    assert ok

sf = A.SURFACES[0]; machine = A.build_machine(sf)
rng = random.Random(0)
cases = A.balanced_cases(sf, 20, 20, rng, machine)
yes = sum(1 for c in cases if c["answer"])
print({"check": "balance", "yes": yes, "no": len(cases) - yes})
assert yes == 20 and len(cases) == 40

c = cases[0]
pl = A.build_prompt(c, "plain"); ws = A.build_prompt(c, "with_state")
wg = A.build_prompt(c, "wrong_state")
print({"check": "conditions", "plain_leaks_state": "Current files" in pl,
       "with_state_has": "Current files" in ws,
       "wrong_differs": ws != wg, "steps_in_transcript": pl.count("step ")})
assert "Current files" not in pl and "Current files" in ws and ws != wg

# the label must follow from the transcript, recomputed independently
delta, order, idx = machine
s = sf["start"]()
for line in c["transcript"].split("\n"):
    body = line.split(": ", 1)[1]
    op = body.split("tool.")[1].split("_file")[0]
    path = body.split("path='")[1].split("'")[0]
    s = sf["step"](s, (op, path))
print({"check": "label_derivable", "recomputed_matches": sf["state_line"](s) == c["state_line"]})
assert sf["state_line"](s) == c["state_line"]

# metrics behave
labels = [c["answer"] and 1 or 0 for c in cases]
print({"check": "auc_perfect", "v": A.auc([1 if l else 0 for l in labels], labels)})
assert A.auc([1 if l else 0 for l in labels], labels) == 1.0
assert A.auc([0.0] * len(labels), labels) == 0.5
always_yes = [1.0] * len(labels)
print({"check": "always_yes", "auc": A.auc(always_yes, labels),
       "unsafe_action_rate": A.unsafe_rate(always_yes, labels)})
assert A.unsafe_rate(always_yes, labels) == 1.0
always_no = [-1.0] * len(labels)
assert A.unsafe_rate(always_no, labels) == 0.0
print("SELFTEST PASSED")
