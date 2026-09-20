"""VARIABLE TRACKING, the RULER-style long-context benchmark, with and without
a verified tracker.

WHY THIS TASK AND NOT A NEW ONE. Adoption does not follow a fresh demo on a
fresh dataset; it follows a number on a benchmark people already run. Variable
tracking is part of RULER (Hsieh et al., 2024) and is reported by most
long-context evaluations. Its core is a binding tracker: chains of assignments
buried in filler text, then "which variables hold this value". That core is
finite-state over the assignment events, which is exactly what statelock
verifies, so the comparison is honest rather than contrived.

  plain        the context alone
  with_state   plus the resolved binding set, computed by the tracker
  wrong_state  plus a resolved-looking set of the SAME SHAPE, wrong contents

The wrong_state arm is the control that decides whether any gain is information
or presentation. Scoring is set recall and exact-set-match over the generated
answer, which is RULER's own measure, not a log-probability proxy.

Run:  python ruler_vt.py                 default models
      python ruler_vt.py MODEL_ID        one model
"""
import json, random, re, sys
import torch

LENGTHS_TOKENS = [1000, 2000, 4000, 8000]
N_CHAINS = 4           # chains per sample, one of which is queried
CHAIN_DEPTH = 4        # assignments per chain
N_PER_CELL = 20
SEEDS = [0, 1, 2]
DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
          "There and back again. ")


def make_sample(n_tokens_target, rng, words_per_token=0.75):
    """A RULER-style variable tracking sample plus its verified state."""
    names = [f"X{i}" for i in range(1, 60)]
    rng.shuffle(names)
    chains, k = [], 0
    for _ in range(N_CHAINS):
        val = rng.randint(10000, 99999)
        chain = [names[k]]; k += 1
        for _ in range(CHAIN_DEPTH - 1):
            chain.append(names[k]); k += 1
        chains.append((val, chain))

    lines = []
    for val, chain in chains:
        lines.append(f"VAR {chain[0]} = {val}")
        for a, b in zip(chain, chain[1:]):
            lines.append(f"VAR {b} = {a}")
    rng.shuffle(lines)                      # order is irrelevant to the answer

    # pad with filler to the requested length, interleaving the assignments
    n_words_target = int(n_tokens_target * words_per_token)
    body, used = [], 0
    filler_words = FILLER.split()
    per_gap = max(1, (n_words_target - sum(len(l.split()) for l in lines)) //
                  (len(lines) + 1))
    for line in lines:
        for _ in range(per_gap):
            body.append(filler_words[used % len(filler_words)]); used += 1
        body.append(line + ".")
    context = " ".join(body)

    val, chain = chains[rng.randrange(len(chains))]
    gold = sorted(chain)
    question = (f"\n\nQuestion: find all variables that are assigned the value "
                f"{val}, directly or through another variable. "
                f"Answer with variable names separated by commas.\nAnswer:")
    state = ("Resolved bindings for " + str(val) + ": " + ", ".join(gold) + ".")
    other = [c for v, c in chains if v != val]
    wrong_set = sorted(rng.choice(other)) if other else gold
    wrong = ("Resolved bindings for " + str(val) + ": " + ", ".join(wrong_set) + ".")
    return context, question, gold, state, wrong


def build_prompt(context, question, state, wrong, condition):
    extra = {"plain": "", "with_state": "\n" + state, "wrong_state": "\n" + wrong}[condition]
    return context + extra + question


def score(answer, gold):
    """RULER's measure: recall of the gold variables, plus exact set match."""
    found = set(re.findall(r"X\d+", answer))
    g = set(gold)
    recall = len(found & g) / len(g)
    precision = len(found & g) / max(len(found), 1)
    return recall, precision, float(found == g)


@torch.no_grad()
def answer_of(model, tok, prompt, device, max_new=40):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=16000).to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    models = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_MODELS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for model_id in models:
        print(f"\n=== {model_id} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
        table = {}
        for L in LENGTHS_TOKENS:
            for cond in ("plain", "with_state", "wrong_state"):
                rec_all, exact_all = [], []
                for seed in SEEDS:
                    rng = random.Random(1000 * seed + L)
                    for _ in range(N_PER_CELL):
                        ctx, q, gold, st, wr = make_sample(L, rng)
                        a = answer_of(model, tok, build_prompt(ctx, q, st, wr, cond), device)
                        r, p, e = score(a, gold)
                        rec_all.append(r); exact_all.append(e)
                rec = {"model": model_id, "context_tokens": L, "condition": cond,
                       "recall": round(sum(rec_all)/len(rec_all), 4),
                       "exact_set_match": round(sum(exact_all)/len(exact_all), 4),
                       "n": len(rec_all), "seeds": len(SEEDS)}
                table[(L, cond)] = rec["recall"]
                print(rec, flush=True); out.append(rec)
        print("\n  recall by context length:", flush=True)
        print(f"  {'tokens':>8} {'plain':>10} {'with_state':>12} {'wrong_state':>12}", flush=True)
        for L in LENGTHS_TOKENS:
            print(f"  {L:>8} {table[(L,'plain')]:>10.3f} {table[(L,'with_state')]:>12.3f} "
                  f"{table[(L,'wrong_state')]:>12.3f}", flush=True)
        del model
        if device == "cuda": torch.cuda.empty_cache()
    json.dump(out, open("ruler_vt_results.json", "w"), indent=1)
    print("\nwritten ruler_vt_results.json")


if __name__ == "__main__":
    main()
