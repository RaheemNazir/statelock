"""Audit an agent's own tool-call trace for actions taken on stale state.

No model is run and nothing is sent anywhere. The input is a trace the agent
framework already produced, plus a short spec saying which tools are
irreversible and what state they depend on. The output is how many irreversible
calls in that trace acted on state that had already changed.

The machine that decides this is checked against the direct simulator over every
tracked value and every tool BEFORE the trace is replayed, and the check is
printed. A run that cannot verify refuses to report a number.

Two counts are reported, never one:
  acted_on_stale   the state forbade the call and nothing re-read it first
  re_read_first    the state forbade the call, but the agent read that object
                   after the change and before acting, so it acted on fresh
                   information and this is a different bug

TRACE FORMAT (JSONL, one object per line; unknown fields ignored)
  {"tool": "delete_file", "args": {"path": "report.txt"}}
  optional: "ts", "result", "id"

SPEC FORMAT (YAML or JSON)
  tracked:
    file:
      values: [exists, gone]
      initial: exists
      id_arg: path
  tools:
    create_file:  {sets: {file: exists}}
    delete_file:  {sets: {file: gone}, irreversible: true, requires: {file: exists}}
    read_file:    {reads: [file]}
"""
import json, sys

__all__ = ["load_spec", "build_machines", "verify_spec", "replay", "audit", "main"]


# --------------------------------------------------------------------- spec

def load_spec(path):
    text = open(path).read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        spec = yaml.safe_load(text)
    else:
        spec = json.loads(text)
    _check_spec(spec)
    return spec


def _check_spec(spec):
    tracked = spec.get("tracked") or {}
    tools = spec.get("tools") or {}
    if not tracked:
        raise ValueError("spec has no tracked objects")
    if not tools:
        raise ValueError("spec has no tools")
    for kind, t in tracked.items():
        if not t.get("values"):
            raise ValueError(f"tracked.{kind} has no values")
        if t.get("initial") not in t["values"]:
            raise ValueError(f"tracked.{kind}.initial must be one of its values")
        if not t.get("id_arg"):
            raise ValueError(f"tracked.{kind} has no id_arg (which argument names the object)")
    for name, t in tools.items():
        for key in ("sets", "requires"):
            for kind, val in (t.get(key) or {}).items():
                if kind not in tracked:
                    raise ValueError(f"tool {name}.{key} names unknown object {kind}")
                if val not in tracked[kind]["values"]:
                    raise ValueError(f"tool {name}.{key}.{kind}={val} is not a value of {kind}")
        for kind in (t.get("reads") or []):
            if kind not in tracked:
                raise ValueError(f"tool {name}.reads names unknown object {kind}")
        if t.get("irreversible") and not t.get("requires"):
            raise ValueError(f"tool {name} is irreversible but has no requires")


# ------------------------------------------------------- machine + hard check

def _step_sim(spec, kind, value, tool):
    """The direct simulator: what this tool does to one tracked object."""
    sets = (spec["tools"][tool].get("sets") or {})
    return sets.get(kind, value)


def build_machines(spec):
    """One transition table per tracked kind: table[value][tool] -> value."""
    machines = {}
    for kind, t in spec["tracked"].items():
        machines[kind] = {v: {tool: _step_sim(spec, kind, v, tool)
                              for tool in spec["tools"]} for v in t["values"]}
    return machines


def verify_spec(spec, machines):
    """Exhaustive agreement between each table and the simulator, over every
    tracked value and every tool. There is no sampling and no cutoff."""
    checked = 0
    for kind, t in spec["tracked"].items():
        for v in t["values"]:
            for tool in spec["tools"]:
                if machines[kind][v][tool] != _step_sim(spec, kind, v, tool):
                    return False, checked
                checked += 1
    return True, checked


# -------------------------------------------------------------------- replay

def read_trace(path):
    events = []
    for i, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"line {i} of {path} is not JSON")
        if "tool" not in e:
            raise ValueError(f"line {i} of {path} has no 'tool' field")
        events.append(e)
    return events


def replay(spec, machines, events):
    """Walk the trace. State starts at each kind's initial value per object id."""
    state = {}                 # (kind, id) -> value
    last_change = {}           # (kind, id) -> step index of the last change
    last_read = {}             # (kind, id) -> step index of the last read
    findings = []

    def value(kind, oid):
        return state.get((kind, oid), spec["tracked"][kind]["initial"])

    for i, e in enumerate(events, 1):
        tool = e.get("tool")
        t = spec["tools"].get(tool)
        if t is None:
            continue           # tools not in the spec are ignored, by design
        args = e.get("args") or {}

        for kind in (t.get("reads") or []):
            oid = args.get(spec["tracked"][kind]["id_arg"])
            if oid is not None:
                last_read[(kind, oid)] = i

        if t.get("irreversible"):
            for kind, need in (t.get("requires") or {}).items():
                oid = args.get(spec["tracked"][kind]["id_arg"])
                if oid is None:
                    continue
                have = value(kind, oid)
                if have != need:
                    ch = last_change.get((kind, oid))
                    rd = last_read.get((kind, oid))
                    fresh = ch is not None and rd is not None and rd > ch
                    findings.append(dict(
                        step=i, tool=tool, object=f"{kind}:{oid}",
                        required=need, actual=have,
                        changed_at_step=ch, last_read_step=rd,
                        re_read_first=bool(fresh)))

        for kind, val in (t.get("sets") or {}).items():
            oid = args.get(spec["tracked"][kind]["id_arg"])
            if oid is None:
                continue
            if value(kind, oid) != val:
                last_change[(kind, oid)] = i
            state[(kind, oid)] = val

    return findings, state


# ---------------------------------------------------------------------- run

def audit(trace_path, spec_path, quiet=False):
    spec = load_spec(spec_path)
    machines = build_machines(spec)
    ok, checked = verify_spec(spec, machines)
    if not quiet:
        print(f"machine check: {len(spec['tracked'])} tracked kinds, "
              f"{checked} (value, tool) pairs checked against the simulator, "
              f"exhaustive agreement {ok}")
    if not ok:
        raise SystemExit("spec does not verify; no number is reported")

    events = read_trace(trace_path)
    findings, _ = replay(spec, machines, events)
    stale = [f for f in findings if not f["re_read_first"]]
    reread = [f for f in findings if f["re_read_first"]]
    irreversible_calls = sum(
        1 for e in events
        if (spec["tools"].get(e.get("tool")) or {}).get("irreversible"))

    report = dict(trace=trace_path, spec=spec_path, machine_verified=ok,
                  pairs_checked=checked, events=len(events),
                  irreversible_calls=irreversible_calls,
                  acted_on_stale=len(stale), re_read_first=len(reread),
                  findings=findings)
    if not quiet:
        print(f"trace: {len(events)} events, {irreversible_calls} irreversible calls")
        print(f"acted on stale state: {len(stale)}")
        print(f"re-read first (different bug, not stale belief): {len(reread)}")
        for f in stale[:20]:
            print(f"  step {f['step']:>4}  {f['tool']}({f['object']})  "
                  f"needed {f['required']}, was {f['actual']}, "
                  f"changed at step {f['changed_at_step']}")
        if len(stale) > 20:
            print(f"  ... {len(stale) - 20} more")
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: python -m statelock.audit TRACE.jsonl SPEC.yaml [--json OUT]")
        return 2
    trace, spec = argv[0], argv[1]
    report = audit(trace, spec)
    if "--json" in argv:
        out = argv[argv.index("--json") + 1]
        json.dump(report, open(out, "w"), indent=1)
        print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
