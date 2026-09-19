"""The smallest complete use: a system you can query, a machine, a check.

The 'system' here is a bracket-depth tracker -- is the current nesting depth
zero -- but it is a black box to the learner, which only calls label_fn.
"""
from statelock import induce, equivalent, StateLock

OPEN, CLOSE, OTHER = 0, 1, 2

def my_system(word, cap=8):
    d = 0
    for t in word:
        d = min(d + 1, cap) if t == OPEN else max(d - 1, 0) if t == CLOSE else d
    return int(d == 0)

delta, out, start, info = induce(alphabet_size=3, label_fn=my_system)
print(info)                                   # states found, queries used
print("self-consistent:", equivalent(delta, out, delta, out, start, start))

lock = StateLock(delta, start, d_model=64)    # ready to add to a model
print("states:", lock.n_states)
