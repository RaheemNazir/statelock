"""statelock -- verified finite machines inside neural networks.

Three things, in the order you would use them:
  induce(...)      work the machine out from labelled strings, by search
  StateDiscoverer  or let a network find it by gradient descent
  equivalent(...)  check the result against a reference by exhaustive BFS
  StateLock        attach the verified machine to your model

Everything here is decided by exact equivalence, not by held-out accuracy.
"""
from .verify import equivalent, reachable, minimal_size
from .induce import induce
from .discover import StateDiscoverer, fit
from .layer import StateLock, attach

__version__ = "0.1.0"
__all__ = ["induce", "equivalent", "reachable", "minimal_size",
           "StateDiscoverer", "fit", "StateLock", "attach"]
