"""STEP 2, v4 -- does a real pretrained model track entity state over length,
and does the verified state help?

v3 FOUND A SIGNAL. On Qwen2.5-0.5B-Instruct, AUC fell from 0.78 to 0.61 as the
passage grew, the supplied state held it at 0.91-1.00, and the gap widened with
length. Two things could still explain that away, so v4 tests both.

  WRONG-STATE CONTROL. The state arrives as an extra English sentence. Maybe any
  extra sentence helps, by reminding the model of the names or the format. So a
  third condition supplies a state sentence with the SAME SHAPE and the WRONG
  CONTENT: a group drawn at random rather than the true one. If wrong_state
  helps as much as with_state, the gain is presentation, not information. This
  is the same role the shuffled-machine control plays in the paper.

  SEEDS. v3 ran one seed. Three separate effects in this line of work crossed
  conventional significance at small n and vanished at larger n, so three seeds
  with per-seed numbers, and a bootstrap interval on every AUC.

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
SEEDS = [0, 1, 2]
BOOTSTRAP = 2000
CONDITIONS = ["plain", "with_state", "wrong_state"]


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
    # the control: same sentence shape, content drawn at random
    k = rng.randint(1, len(NAMES))
    wrong = "Known groups: " + ", ".join(rng.sample(NAMES, k)) + " are connected."
    return " ".join(sents), ans, state, wrong


def balanced_cases(n_sent, n_per_class, rng, tries=20000):
    buckets = {"yes": [], "no": []}
    for _ in range(tries):
        passage, ans, state, wrong = make_case(n_sent, rng)
        if len(buckets[ans]) < n_per_class:
            buckets[ans].append((passage, ans, state, wrong))
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


def build_prompt(passage, state, wrong, condition):
    extra = {"plain": "", "with_state": " " + state, "wrong_state": " " + wrong}[condition]
    return FEWSHOT + passage + extra + "\n" + QUESTION


def bootstrap_ci(scores, labels, n_boot=BOOTSTRAP, seed=0):
    """Percentile interval on AUC, resampling cases with replacement."""
    import random as _r
    rr = _r.Random(seed); n = len(scores); vals = []
    idx = list(range(n))
    for _ in range(n_boot):
        pick = [idx[rr.randrange(n)] for _ in range(n)]
        s = [scores[i] for i in pick]; l = [labels[i] for i in pick]
        if 0 < sum(l) < n: vals.append(auc(s, l))
    vals.sort()
    if not vals: return (0.5, 0.5)
    return (round(vals[int(0.025*len(vals))], 4), round(vals[int(0.975*len(vals))], 4))


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

        table = {}
        for seed in SEEDS:
            for L in LENGTHS:
                rng = random.Random(1000 * seed + L)
                cases = balanced_cases(L, N_PER_CLASS, rng)
                for cond in CONDITIONS:
                    scores, labels = [], []
                    for passage, ans, state, wrong in cases:
                        scores.append(margin(model, tok,
                                             build_prompt(passage, state, wrong, cond),
                                             yes_id, no_id, device))
                        labels.append(1 if ans == "yes" else 0)
                    a = auc(scores, labels)
                    lo, hi = bootstrap_ci(scores, labels, seed=seed)
                    rec = {"model": model_id, "seed": seed, "sentences": L,
                           "condition": cond, "auc": round(a, 4),
                           "auc_ci95": [lo, hi],
                           "acc_at_median": round(acc_at_median(scores, labels), 4),
                           "yes_rate_at_zero": round(sum(x > 0 for x in scores)/len(scores), 4),
                           "mean_margin": round(sum(scores)/len(scores), 4),
                           "n": len(cases), "chance_auc": 0.5}
                    table.setdefault((L, cond), []).append(a)
                    print(rec, flush=True); out.append(rec)

        print("\n  AUC by length, mean over seeds:", flush=True)
        print(f"  {'len':>5} {'plain':>18} {'with_state':>18} {'wrong_state':>18}", flush=True)
        for L in LENGTHS:
            cells = []
            for cond in CONDITIONS:
                v = table[(L, cond)]
                cells.append(f"{sum(v)/len(v):.3f} ({min(v):.2f}-{max(v):.2f})")
            print(f"  {L:>5} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}", flush=True)
        print("\n  gaps over plain, mean over seeds:", flush=True)
        for L in LENGTHS:
            p = sum(table[(L, "plain")]) / len(table[(L, "plain")])
            w = sum(table[(L, "with_state")]) / len(table[(L, "with_state")])
            x = sum(table[(L, "wrong_state")]) / len(table[(L, "wrong_state")])
            print(f"    {L:>4}:  with_state {w-p:+.3f}   wrong_state {x-p:+.3f}", flush=True)
        print("\n  READING: with_state must beat plain AND wrong_state. If "
              "wrong_state gains as much, the effect is the extra sentence, not "
              "the information it carries.", flush=True)
        del model
        if device == "cuda": torch.cuda.empty_cache()

    json.dump(out, open("step2_results.json", "w"), indent=1)
    print("\nwritten step2_results.json")


if __name__ == "__main__":
    main()
