"""STEP 2, v2 -- does a real pretrained model track entity state over length,
and does supplying the verified state help?

WHY v1 MEASURED NOTHING. Its numbers had accuracy exactly equal to the majority
baseline in every row, and the two conditions identical to four decimals. That
is the signature of a model answering the same token every time: the score was
just the class balance. Three faults, fixed here.

  1. UNBALANCED. Questions were drawn at random, so the majority class moved
     around and a constant answer could score 0.57. Cases are now balanced 50/50
     by construction, so a constant answer scores exactly 0.50 and cannot hide.
  2. NO FORMAT. A base model given a bare question has no reason to emit yes or
     no at all. Four worked examples now precede the question.
  3. UNCALIBRATED. Base models carry a large prior toward one of two answer
     tokens. The same prompt with the content removed is now scored first, and
     that bias is subtracted (contextual calibration). Without this, a model
     that knows the answer can still answer "no" every time.

REPORTED, so a degenerate run is visible rather than silent:
  accuracy, yes_rate (0.5 means it is actually deciding), mean margin, and the
  calibration offset. If yes_rate is 0.0 or 1.0, the run is degenerate and the
  accuracy is meaningless whatever it says.

Usage:  python step2_real_model.py            (default model list)
        python step2_real_model.py MODEL_ID   (one model)
"""
import json, random, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODELS = ["EleutherAI/pythia-160m", "Qwen/Qwen2.5-0.5B-Instruct"]
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin"]
LENGTHS = [8, 16, 32, 64, 128]
N_PER_CLASS = 40            # 40 yes + 40 no at each length
SEED = 0


def make_case(n_sent, rng):
    """One passage. Returns (passage, answer, state_sentence)."""
    parent = list(range(len(NAMES)))

    def find(a):
        while parent[a] != a: a = parent[a]
        return a

    sents = []
    for _ in range(n_sent):
        if rng.random() < 0.12:
            parent = list(range(len(NAMES)))
            sents.append("Everyone parts ways.")
            continue
        a, b = rng.randrange(len(NAMES)), rng.randrange(len(NAMES))
        parent[find(a)] = find(b)
        sents.append(f"{NAMES[a]} knows {NAMES[b]}.")
    ans = "yes" if find(0) == find(1) else "no"
    group = [NAMES[i] for i in range(len(NAMES)) if find(i) == find(0)]
    state = "Known groups: " + ", ".join(group) + " are connected."
    return " ".join(sents), ans, state


def balanced_cases(n_sent, n_per_class, rng, tries=20000):
    """Exactly n_per_class yes and n_per_class no, so a constant answer is 0.50."""
    buckets = {"yes": [], "no": []}
    for _ in range(tries):
        passage, ans, state = make_case(n_sent, rng)
        if len(buckets[ans]) < n_per_class:
            buckets[ans].append((passage, ans, state))
        if len(buckets["yes"]) == n_per_class and len(buckets["no"]) == n_per_class:
            break
    cases = buckets["yes"] + buckets["no"]
    rng.shuffle(cases)
    return cases


QUESTION = ("Question: are Alice and Bob connected, directly or indirectly? "
            "Answer with yes or no.\nAnswer:")

FEWSHOT = (
    "Alice knows Bob.\n" + QUESTION + " yes\n\n"
    "Carol knows Dave.\n" + QUESTION + " no\n\n"
    "Alice knows Carol. Carol knows Bob.\n" + QUESTION + " yes\n\n"
    "Everyone parts ways. Dave knows Erin.\n" + QUESTION + " no\n\n"
)


def build_prompt(passage, state, use_state):
    body = passage + ((" " + state) if use_state else "")
    return FEWSHOT + body + "\n" + QUESTION


@torch.no_grad()
def margin(model, tok, text, yes_id, no_id, device, max_len=3000):
    """log p(yes) - log p(no) for the next token."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    logits = model(**ids).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    return float(lp[yes_id] - lp[no_id])


def main():
    models = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_MODELS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for model_id in models:
        print(f"\n=== {model_id} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        yes_id = tok(" yes", add_special_tokens=False).input_ids[-1]
        no_id = tok(" no", add_special_tokens=False).input_ids[-1]

        # contextual calibration: the same prompt shape with no content
        null_prompt = build_prompt("N/A", "N/A", False)
        offset = margin(model, tok, null_prompt, yes_id, no_id, device)
        print(f"calibration offset {offset:+.3f}", flush=True)

        for L in LENGTHS:
            for use_state in (False, True):
                rng = random.Random(SEED + L)
                cases = balanced_cases(L, N_PER_CLASS, rng)
                hits = yes_pred = 0
                margins = []
                for passage, ans, state in cases:
                    m = margin(model, tok, build_prompt(passage, state, use_state),
                               yes_id, no_id, device) - offset
                    pred = "yes" if m > 0 else "no"
                    hits += int(pred == ans); yes_pred += int(pred == "yes")
                    margins.append(m)
                n = len(cases)
                rec = {"model": model_id, "sentences": L,
                       "condition": "with_state" if use_state else "plain",
                       "accuracy": round(hits / n, 4),
                       "yes_rate": round(yes_pred / n, 4),
                       "mean_margin": round(sum(margins) / n, 4),
                       "calibration_offset": round(offset, 4),
                       "n": n, "balanced_baseline": 0.5,
                       "degenerate": yes_pred in (0, n)}
                print(rec, flush=True); out.append(rec)
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    json.dump(out, open("step2_results.json", "w"), indent=1)
    print("\nwritten step2_results.json")
    deg = [r for r in out if r["degenerate"]]
    if deg:
        print(f"WARNING: {len(deg)} of {len(out)} rows are degenerate (the model "
              f"answered the same token every time). Their accuracy means nothing.")


if __name__ == "__main__":
    main()
