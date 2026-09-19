"""Discover the automaton by gradient descent, with no oracle-driven search.

The state is a learned transition table run as a hard argmax scan with a
straight-through gradient. What makes it work, and what four earlier attempts
lacked, is the target: the state is trained to predict its MYHILL-NERODE ROW --
the labels the sequence would receive under a bundle of probe continuations --
rather than only the current label. Predicting the current label gives the
optimiser no reason to separate two states that agree now and differ later, and
a partition that is coarser than the target's cannot be repaired afterwards,
because merging only coarsens.

Three settings matter and were each fixed by measurement, not taste:
  * train on strings that COVER the alphabet, not only in-distribution ones.
    With grammatical data alone, loss reached 0.0 and every held-out label was
    right, and exact equivalence still failed, because the table was
    unconstrained on strings the generator never emits.
  * interleave sequence lengths rather than staging them, or the short gradient
    path is forgotten.
  * hold the straight-through temperature constant. The forward pass is a hard
    argmax at any temperature, so annealing only sharpens the surrogate until
    the gradient vanishes.

Exactness tracks the training loss: runs reaching zero verified exact, runs that
stalled did not. So `fit` reports the loss and you restart on that, without
needing an answer key.
"""
import numpy as np
import torch
import torch.nn as nn

__all__ = ["StateDiscoverer", "fit"]


class _Scan(nn.Module):
    def __init__(self, V, K, d, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(V, K, K, generator=g) * 0.5)
        self.s0 = nn.Parameter(torch.randn(K, generator=g) * 0.5)
        self.emb = nn.Embedding(K, d)
        with torch.no_grad():
            self.emb.weight.normal_(0, 1.0, generator=g)
        self.V, self.K = V, K

    def forward(self, x, tau=1.0):
        B, L = x.shape
        p = torch.softmax(self.s0 / tau, -1).expand(B, self.K)
        hard = torch.zeros_like(p).scatter_(1, p.argmax(-1, keepdim=True), 1.0)
        st = hard + p - p.detach()
        outs = []
        for t in range(L):
            logits = torch.einsum('bk,bkj->bj', st, self.W[x[:, t]])
            p = torch.softmax(logits / tau, -1)
            hard = torch.zeros_like(p).scatter_(1, p.argmax(-1, keepdim=True), 1.0)
            st = hard + p - p.detach()
            outs.append(st)
        return torch.stack(outs, 1) @ self.emb.weight

    def table(self):
        return (self.W.argmax(-1).detach().cpu().numpy().astype(np.int64),
                int(self.s0.argmax().item()))


class StateDiscoverer(nn.Module):
    """A discrete state layer that learns its own transition table."""

    def __init__(self, alphabet_size, n_slots, n_probes, d=64, seed=0):
        super().__init__()
        self.scan = _Scan(alphabet_size, n_slots, d, seed)
        self.row = nn.Linear(d, n_probes)

    def forward(self, x, tau=1.0):
        """Returns per-position row logits."""
        return self.row(self.scan(x, tau))

    def states(self, x):
        """The discovered state index at each position (no gradient)."""
        delta, s0 = self.scan.table()
        x = x.cpu().numpy()
        s = np.full(x.shape[0], s0, np.int64)
        out = np.empty(x.shape, np.int64)
        for t in range(x.shape[1]):
            s = delta[x[:, t], s]; out[:, t] = s
        return out

    def table(self):
        return self.scan.table()


def fit(model, batches, steps=40000, lr=2e-2, tau=1.0, log_every=0):
    """Train on an iterable-returning callable `batches(step) -> (x, rows)`.

    x:    LongTensor (B, L) of symbols
    rows: FloatTensor (B, L, n_probes) of 0/1 probe labels
    Returns the smoothed final loss. Exactness tracks this: restart if it does
    not approach zero.
    """
    head = [p for n, p in model.named_parameters() if not n.startswith("scan.W")
            and not n.startswith("scan.s0")]
    opt = torch.optim.AdamW([{"params": [model.scan.W, model.scan.s0], "lr": lr},
                             {"params": head, "lr": lr / 3}], weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr / 3], total_steps=steps)
    lf = nn.BCEWithLogitsLoss(); ema = None
    for it in range(steps):
        x, rows = batches(it)
        loss = lf(model(x, tau), rows)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        v = float(loss.detach())
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        if log_every and it % log_every == 0:
            print(f"step {it} loss {ema:.5f}", flush=True)
    return ema
