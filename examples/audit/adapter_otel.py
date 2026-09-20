"""Turn an OpenTelemetry GenAI span export into the trace format the auditor reads.

Step six of the build, and the only part that touches anybody else's format.
OpenTelemetry is the right first adapter because several agent frameworks export
it, so one adapter covers several systems.

  python adapter_otel.py spans.json > trace.jsonl

WHAT IT ASSUMES. Spans with a name or attribute identifying a tool call, and the
arguments as JSON in an attribute. Exporters differ, so the two field names are
at the top and are meant to be edited to match yours. Run it, look at the first
ten lines of output, and confirm the tool names and argument keys are the ones
your spec uses. This adapter is not verified by anything; the auditor is. A
wrong adapter gives a wrong number with a verified machine behind it.
"""
import json, sys

TOOL_KEYS = ["gen_ai.tool.name", "tool.name", "function.name"]
ARG_KEYS = ["gen_ai.tool.input", "tool.arguments", "function.arguments", "input"]


def attr(span, keys):
    a = span.get("attributes") or {}
    if isinstance(a, list):                      # some exporters emit a key/value list
        a = {d.get("key"): d.get("value") for d in a}
    for k in keys:
        if k in a:
            v = a[k]
            if isinstance(v, dict) and "stringValue" in v:
                v = v["stringValue"]
            return v
    return None


def convert(spans):
    out = []
    for s in spans:
        tool = attr(s, TOOL_KEYS) or (s.get("name") if s.get("name", "").startswith("tool.") else None)
        if not tool:
            continue
        tool = tool.split(".")[-1]
        raw = attr(s, ARG_KEYS)
        args = raw if isinstance(raw, dict) else {}
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
        out.append({"tool": tool, "args": args, "ts": s.get("startTimeUnixNano") or s.get("start_time")})
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python adapter_otel.py SPANS.json > trace.jsonl", file=sys.stderr)
        return 2
    d = json.load(open(sys.argv[1]))
    spans = d if isinstance(d, list) else d.get("spans") or d.get("resourceSpans") or []
    if spans and isinstance(spans[0], dict) and "scopeSpans" in spans[0]:
        spans = [s for r in spans for sc in r.get("scopeSpans", []) for s in sc.get("spans", [])]
    rows = convert(spans)
    for r in rows:
        print(json.dumps(r))
    print(f"{len(rows)} tool calls converted", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
