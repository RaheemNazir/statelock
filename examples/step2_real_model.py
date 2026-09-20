"""STEP 2, v3 -- does a real pretrained model track entity state over length,
and does the verified state help?

WHY v1 AND v2 MEASURED NOTHING, AND WHAT CHANGED.
  v1 scored accuracy against an unbalanced set, so a model answering one word
  every time scored 0.5667 and looked like a result.
  v2 balanced the set and subtracted a calibration offset, so the same model
  scored exactly 0.50 and was flagged degenerate. Correct, but still no signal:
  both models answered "yes" to all 80 questions, at every length.

The mistake both versions share is THRESHOLDING. Turning a continuous score into
yes or no throws away everything except which side of an arbitrary line it fell,
and a fixed prior moves every item to the same side. The question we actually
care about is different: does the model's score carry ANY information about the
answer? That is a ranking question, and ranking does not care about bias.

So the metric is ROC AUC over the yes-minus-no log-probability margin:
  0.50 means the score is uninformative, whatever the model answers
  1.00 means the score separates connected from unconnected perfectly
A model that says yes to everything but gives connected pairs a higher score
still scores above 0.50, and that is real signal that thresholding hides.

Also reported: accuracy at the median split (what you could get by calibrating
the threshold per condition), yes_rate for transparency, and a paired
plain-versus-with_state comparison, which is the actual experiment.

Usage:  python step2_real_model.py            (default model list)
        python step2_real_model.py MODEL_ID   (one model)
"""
import json, random, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODELS = ["EleutherAI/pythia-160m", "Qwen/Qwen2.5-0.5B-Instruct"]
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin"]
LENGTHS = [8, 16, 32, 64, 128]
N_PER_CLASS = 40
SEED = 0


def make_case(n_sent, rng):
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


def auc(scores, labels):
    """ROC AUC by rank, ties averaged. No sklearn dependency.
    labels: 1 for yes, 0 for no."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): ranks[k] = avg
        i = j + 1
    pos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    n_pos = sum(l for _, l in pairs); n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    return (pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def acc_at_median(scores, labels):
    """Accuracy with the threshold placed at the median score, which is the best
    a per-condition calibration could do without peeking at labels."""
    s = sorted(scores); mid = s[len(s) // 2]
    return sum(int((sc > mid) == (l == 1)) for sc, l in zip(scores, labels)) / len(scores)


@torch.no_grad()
def margin(model, tok, text, yes_id, no_id, device, max_len=3000):
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

        per_length = {}
        for L in LENGTHS:
            for use_state in (False, True):
                rng = random.Random(SEED + L)
                cases = balanced_cases(L, N_PER_CLASS, rng)
                scores, labels = [], []
                for passage, ans, state in cases:
                    scores.append(margin(model, tok,
                                         build_prompt(passage, state, use_state),
                                         yes_id, no_id, device))
                    labels.append(1 if ans == "yes" else 0)
                a = auc(scores, labels)
                rec = {"model": model_id, "sentences": L,
                       "condition": "with_state" if use_state else "plain",
                       "auc": round(a, 4),
                       "acc_at_median": round(acc_at_median(scores, labels), 4),
                       "yes_rate_at_zero": round(sum(s > 0 for s in scores)/len(scores), 4),
                       "mean_margin": round(sum(scores)/len(scores), 4),
                       "n": len(cases), "chance_auc": 0.5}
                per_length.setdefault(L, {})[rec["condition"]] = a
                print(rec, flush=True); out.append(rec)
        print("  paired, with_state minus plain, by length:", flush=True)
        for L in LENGTHS:
            d = per_length[L]["with_state"] - per_length[L]["plain"]
            print(f"    {L:4d} sentences:  {d:+.4f} AUC", flush=True)
        del model
        if device == "cuda": torch.cuda.empty_cache()

    json.dump(out, open("step2_results.json", "w"), indent=1)
    print("\nwritten step2_results.json")
    best = max((r["auc"] for r in out), default=0.5)
    if best < 0.55:
        print("READING: every AUC is at or near 0.50. These models carry no usable "
              "information about the answer at any length, so this task is beyond "
              "them rather than being lost over length. That is a finding about "
              "model scale, not about statelock, and it should be reported as such.")


if __name__ == "__main__":
    main()
