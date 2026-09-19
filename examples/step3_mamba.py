"""STEP 3 -- the same task on a state-space model instead of a transformer.

NOT RUN HERE (no GPU, no weight downloads). Run on Colab with a T4 GPU.
Install first:  !pip install mamba-ssm causal-conv1d torch numpy
                (if mamba-ssm fails to build, use the fallback flag below)
Then:           !python step3_mamba.py
Send me step3_results.json.

Same NL-CONNECTIVITY task and the same three arms as the transformer ladder:
  none      -- model alone
  machine   -- exact machine state added to the input embedding
  induced   -- the machine induced from labels, if induced_nl.npz is uploaded
The question is whether a state-space model also decays with length. If it does
not, the finding is about attention and the paper must say so.
"""
import json, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, ".")
from nltask import gen_prose, scan, K_TRUE, V          # upload nltask.py too

USE_FALLBACK_GRU = False     # set True if mamba-ssm will not install

if not USE_FALLBACK_GRU:
    from mamba_ssm import Mamba

class Core(nn.Module):
    def __init__(self, d, nl):
        super().__init__()
        if USE_FALLBACK_GRU:
            self.net = nn.GRU(d, d, num_layers=nl, batch_first=True)
        else:
            self.net = nn.ModuleList([Mamba(d_model=d, d_state=16, d_conv=4, expand=2)
                                      for _ in range(nl)])
    def forward(self, h):
        if USE_FALLBACK_GRU: return self.net(h)[0]
        for b in self.net: h = b(h)
        return h

class Net(nn.Module):
    def __init__(self, d=256, nl=4, embed=True, nstate=K_TRUE):
        super().__init__()
        self.embed = embed
        self.tok = nn.Embedding(V, d); self.core = Core(d, nl)
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, 2)
        if embed: self.mach = nn.Embedding(nstate, d)
    def forward(self, x, st=None):
        h = self.tok(x)
        if self.embed: h = h + self.mach(st)
        return self.head(self.ln(self.core(h)))

def data(arm, n, L, rng, dev):
    X = gen_prose(n, L, rng); _, Y = scan(X)
    ST = np.zeros_like(X) if arm == "none" else scan(X)[0]
    t = lambda a: torch.from_numpy(a).to(dev)
    return t(X), t(Y), t(ST)

def run(arm, seed, steps=1500, bs=16, L=128, lr=1e-3, d=256, nl=4):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X, Y, ST = data(arm, 4096, L, rng, dev)
    m = Net(d, nl, embed=(arm != "none")).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lf = nn.CrossEntropyLoss()
    for _ in range(steps):
        i = torch.randint(0, X.shape[0], (bs,), device=dev)
        loss = lf(m(X[i], ST[i]).reshape(-1, 2), Y[i].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    m.eval(); acc = {}
    with torch.no_grad():
        for Lt in (128, 256, 512, 1024, 2048, 8192):
            Xe, Ye, STe = data(arm, 16, Lt, np.random.default_rng(seed+777), dev)
            acc[str(Lt)] = round(float((m(Xe, STe).argmax(-1) == Ye).float().mean()), 4)
    return acc

if __name__ == "__main__":
    out = []
    for arm in ("none", "machine"):
        for seed in range(3):
            acc = run(arm, seed)
            rec = {"core": "gru" if USE_FALLBACK_GRU else "mamba",
                   "arm": arm, "seed": seed, "acc": acc}
            print(rec, flush=True); out.append(rec)
    json.dump(out, open("step3_results.json", "w"), indent=1)
    print("written step3_results.json")
