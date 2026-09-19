"""Attach a verified machine to a model.

`StateLock` turns a transition table into an embedding lookup over machine
states and adds it to a hidden-state tensor. The table is fixed: it is never
trained, so whatever was verified about it stays true inside the network.

Two ways to use it:
  * directly, adding the state embedding to your own hidden states;
  * `attach(hf_model, lock, input_ids_fn)` installs a forward hook on a Hugging
    Face model's embedding layer, so the state is added to the token embeddings
    with no change to the model's code.
"""
import numpy as np
import torch
import torch.nn as nn

__all__ = ["StateLock", "attach"]


class StateLock(nn.Module):
    def __init__(self, delta, start=0, d_model=768, out=None):
        super().__init__()
        delta = np.asarray(delta, dtype=np.int64)
        self.register_buffer("delta", torch.from_numpy(delta))
        self.start = int(start)
        self.n_states = delta.shape[1]
        self.emb = nn.Embedding(self.n_states, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.register_buffer("out", torch.from_numpy(np.asarray(out, dtype=np.int64))
                             if out is not None else torch.zeros(0, dtype=torch.long))

    @torch.no_grad()
    def run(self, x):
        """State index at every position. Exact, O(length), never trained."""
        d = self.delta.cpu().numpy(); xi = x.cpu().numpy()
        s = np.full(xi.shape[0], self.start, np.int64)
        st = np.empty(xi.shape, np.int64)
        for t in range(xi.shape[1]):
            s = d[xi[:, t], s]; st[:, t] = s
        return torch.from_numpy(st).to(x.device)

    def forward(self, hidden, x):
        """hidden: (B, L, d). x: (B, L) symbols. Returns hidden + state embedding."""
        return hidden + self.emb(self.run(x))


def attach(hf_model, lock, symbol_fn=None):
    """Add the machine's state to a Hugging Face model's token embeddings.

    symbol_fn maps input_ids to machine symbols; omit it when the model's token
    ids already are the machine's alphabet. Returns a handle; call
    handle.remove() to detach.
    """
    emb = hf_model.get_input_embeddings()
    state = {"ids": None}

    def pre_hook(_module, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        state["ids"] = ids
        return None

    def hook(_module, args, output):
        ids = args[0] if args else state["ids"]
        if ids is None:
            return output
        sym = ids if symbol_fn is None else symbol_fn(ids)
        return output + lock.emb(lock.run(sym)).to(output.dtype)

    hf_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    return emb.register_forward_hook(hook)
