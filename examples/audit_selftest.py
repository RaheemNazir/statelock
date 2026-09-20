"""Check the auditor against traces whose violation count was planted.

If this does not print ALL CHECKS PASSED, the number the auditor reports on a
real trace means nothing.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# walk up until the folder holding the statelock package is found
d = HERE
for _ in range(4):
    if os.path.isdir(os.path.join(d, "statelock")):
        sys.path.insert(0, d); break
    d = os.path.dirname(d)
sys.path.insert(0, HERE)

from statelock.audit import load_spec, build_machines, verify_spec, read_trace, replay
import make_traces as mt

SPEC = os.path.join(HERE, "example_spec.yaml")


def run(name, events, expect_stale, expect_reread=None):
    spec = load_spec(SPEC)
    machines = build_machines(spec)
    ok, checked = verify_spec(spec, machines)
    assert ok, "spec does not verify"
    mt.write(name, events)
    findings, _ = replay(spec, machines, read_trace(name))
    stale = sum(1 for f in findings if not f["re_read_first"])
    reread = sum(1 for f in findings if f["re_read_first"])
    print(f"{os.path.basename(name):<14} pairs checked {checked:>3}  stale {stale} (planted {expect_stale})"
          f"  re-read {reread}" + (f" (expected {expect_reread})" if expect_reread is not None else ""))
    assert stale == expect_stale, f"{name}: reported {stale}, planted {expect_stale}"
    if expect_reread is not None:
        assert reread == expect_reread, f"{name}: re-read {reread}, expected {expect_reread}"


def main():
    ev, planted = mt.clean(60);  run(os.path.join(HERE, "clean.jsonl"), ev, planted, 0)
    ev, planted = mt.stale(60);  run(os.path.join(HERE, "stale.jsonl"), ev, planted, 0)
    ev, planted = mt.mixed(60);  run(os.path.join(HERE, "mixed.jsonl"), ev, planted, 4)
    for seed in range(1, 6):     # the count must hold across sessions, not one lucky seed
        ev, planted = mt.stale(80, planted=9, seed=seed)
        run(os.path.join(HERE, f"stale_s{seed}.jsonl"), ev, planted, 0)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
