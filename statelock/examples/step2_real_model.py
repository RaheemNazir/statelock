"""STEP 2 -- does a REAL pretrained language model lose the state over length,
and does supplying the machine's state help?

NOT RUN HERE: this session cannot download model weights (huggingface.co is
blocked), so this script is untested. Run it on Colab with a T4 GPU. If it
errors, send me the error text and I will fix it.

What it does:
  * writes the connectivity task as ENGLISH sentences ("Alice knows Bob.")
  * asks a small open model, by scoring the words " yes" and " no", whether two
    named people are connected, after passages of growing length
  * repeats the same questions with one extra sentence supplying the machine's
    answer state ("Currently connected: Alice, Bob, Carol.")
  * reports accuracy per length for both conditions, plus the constant baseline

Install first:   !pip install transformers torch accelerate
Then:            !python step2_real_model.py
It writes step2_results.json -- send me that file.
"""
import json, itertools, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "EleutherAI/pythia-160m"      # swap for "gpt2" or "EleutherAI/pythia-410m"
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin"]
LENGTHS = [8, 16, 32, 64, 128]        # sentences per passage
N_Q = 60                              # questions per length
SEEDS = [0, 1, 2]

def make_case(n_sent, rng):
    """Returns (passage, question, answer, state_sentence)."""
    parent = list(range(len(NAMES)))
    def find(a):
        while parent[a] != a: a = parent[a]
        return a
    sents = []
    for _ in range(n_sent):
        if rng.random() < 0.12:
            parent = list(range(len(NAMES))); sents.append("Everyone parts ways.")
            continue
        a, b = rng.randrange(len(NAMES)), rng.randrange(len(NAMES))
        ra, rb = find(a), find(b); parent[ra] = rb
        sents.append(f"{NAMES[a]} knows {NAMES[b]}.")
    ans = "yes" if find(0) == find(1) else "no"
    group = [NAMES[i] for i in range(len(NAMES)) if find(i) == find(0)]
    state = "Currently in the same group as Alice: " + ", ".join(group) + "."
    q = f"Question: are {NAMES[0]} and {NAMES[1]} connected, directly or indirectly? Answer:"
    return " ".join(sents), q, ans, state

@torch.no_grad()
def ask(model, tok, text, device):
    """Score ' yes' vs ' no' as the next token."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    logits = model(**ids).logits[0, -1]
    y = tok(" yes").input_ids[-1]; n = tok(" no").input_ids[-1]
    return "yes" if logits[y] > logits[n] else "no"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
    out = []
    for L in LENGTHS:
        for cond in ("plain", "with_state"):
            hits = 0; total = 0; yes = 0
            for s in SEEDS:
                rng = random.Random(1000*s + L)
                for _ in range(N_Q):
                    passage, q, ans, state = make_case(L, rng)
                    text = passage + (" " + state if cond == "with_state" else "") + " " + q
                    pred = ask(model, tok, text, device)
                    hits += int(pred == ans); total += 1; yes += int(ans == "yes")
            rec = {"model": MODEL, "sentences": L, "condition": cond,
                   "accuracy": round(hits/total, 4), "n": total,
                   "constant_baseline": round(max(yes/total, 1-yes/total), 4)}
            print(rec, flush=True); out.append(rec)
    json.dump(out, open("step2_results.json", "w"), indent=1)
    print("written step2_results.json")

if __name__ == "__main__":
    main()
