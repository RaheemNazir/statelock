# statelock

Transformers lose track of state as inputs get longer. On a prose connectivity
task, a model goes from 0.92 accuracy at its training length to 0.50 at 16x
length, and growing it from 1.3M to 88M parameters does not help: the largest
model is the best in distribution and the worst at length.

`statelock` gives the model a finite machine to consult, and the machine is
**checked exhaustively** rather than measured on a test set. With one attached,
the same model holds 1.0000 at 16x length.

```python
from statelock import induce, equivalent, StateLock

delta, out, start, info = induce(alphabet_size=8, label_fn=my_system)   # find it
assert equivalent(delta, out, ref_delta, ref_out, start, ref_start)      # check it
lock = StateLock(delta, start, d_model=768)                              # attach it
hidden = lock(hidden, symbols)            # inside your model's forward
```

That is the whole interface.

## Install

```
pip install statelock
```

## The three things it does

**Induce.** Work the machine out from labelled strings. No state supervision, no
hints. You supply `label_fn(word) -> int`, the label your system gives a string
you choose. On a 152-state target it recovers exactly 152 states, equal to the
minimal machine, and passes exact equivalence.

**Discover.** Or let a network find the machine during ordinary training, with
no search at all:

```python
from statelock import StateDiscoverer, fit
model = StateDiscoverer(alphabet_size=6, n_slots=32, n_probes=64)
loss = fit(model, batches, steps=3000)     # exactness tracks this loss
delta, start = model.table()
```

The state is trained to predict its Myhill-Nerode row, the labels a bundle of
probe continuations would produce, not just the current label. Predicting the
current label alone gives the optimiser no reason to separate two states that
agree now and differ later, and four earlier attempts failed for exactly that
reason. Exactness tracks the training loss: a run that reaches zero verifies,
one that stalls does not, so you restart on the loss without needing an answer
key.

**Verify.** `equivalent(...)` decides agreement on *every* string by BFS over
reachable state pairs, with no length cutoff. This is not optional rigour. A
bounded enumeration to length 11 once reported 6/6 correct here and disagreed
with true equivalence on 22 of 48 corrupted targets.

## With Hugging Face

```python
from statelock import StateLock, attach
lock = StateLock(delta, start, d_model=model.config.hidden_size)
handle = attach(model, lock, symbol_fn=my_tokens_to_symbols)   # forward hook
```

No change to the model's code. `handle.remove()` detaches.

## What it is for, and what it is not for

Use it where the task has a discrete core with a state count in the hundreds:
entity and relation tracking across a document, protocol and permission state,
bracket and scope structure, small formal languages, board and game state.

It will not help where there is no such core, and it does not make a model
better at anything except keeping that state. The machine is a component you
attach, and attaching it is a design decision about your task.

## Measured

Prose connectivity, dense per-position labels, trained at length 128:

| model | arm | L=128 | L=2048 (16x) |
|---|---|---|---|
| 1.3M | none | 0.9155 | 0.5848 |
| 1.3M | shuffled machine | 0.8984 | 0.6033 |
| 1.3M | statelock | 1.0000 | 0.9998 |
| 27M | none | 0.9099 | 0.6191 |
| 27M | statelock | 1.0000 | 1.0000 |
| 88M | none | 0.9167 | 0.5028 |
| 88M | statelock | 1.0000 | 1.0000 |

Constant-predictor baseline 0.5204. The shuffled-machine row is the control that
matters: same parameters, same embedding, random transitions, no gain.

Discovery (the network finds the machine itself, then it is verified):

| target minimal states | slots offered | exact | loss on exact runs |
|---|---|---|---|
| 11 | 32 | 2/3 seeds | 0.0 |
| 38 | 128 | 1/3 seeds | 0.0 |

## Reproduce

```
python tests/test_all.py
```

Induction to 152 states, 40 corruptions all flagged by the verifier, exact
running at length 4096, and a discovery run that verifies. Runs on CPU.

## Licence

MIT.
