# Verified finite machines inside transformers: induction from I/O pairs, composition, embedding, and scale

## Abstract

Transformers with finite depth cannot track the state of a bounded automaton
over arbitrary input length; the failure is visible empirically as a collapse of
accuracy with sequence length, and is consistent with the TC0/NC1 picture of
what constant-depth architectures can compute. One response is to give the
network a finite machine it can consult. That raises three questions this work
answers with executable checks rather than accuracy curves: can the machine be
obtained from input/output pairs alone, with no state supervision and no
membership oracle; do independently obtained machines compose without loss; and
does a machine embedded inside a real transformer keep its exactness as the
transformer is scaled.

We report: (i) exact induction of the minimal automaton, verified by product-BFS
equivalence rather than by held-out accuracy, for MOD-m families to m = 64 and
for S3/S4/S5 observables, from label sequences only; (ii) exact composition of
two independently induced machines into their Cartesian product, verified over
1368 instances with a negative control that fails; (iii) an end-to-end hybrid
layer on continuous inputs with zero measured drift under 2-, 4- and 8-fold
composition against a capacity-matched continuous baseline that drifts;
(iv) a real pretrained model (Qwen2.5-0.5B-Instruct) whose entity-tracking AUC
falls from 0.80 to 0.58 as passages grow while the supplied verified state holds
it at 1.000, with a same-shape wrong-state control that does not help and hurts;
(v) the same pipeline on a prose task whose control is not at chance -- a transformer without the machine reaches 0.92 in distribution and
0.50-0.62 at 16x length at every size from 1.3M to 88M, while the induced
machine holds 1.0000, and a shuffled-machine control scores like no machine;
(vi) a scaling ladder to 88.35M parameters -- GPT-2-small's architecture,
trained from scratch on two CPU cores -- in which the embedded machine's
16x-length accuracy *rises* with model size (0.9620 at 1.35M to 0.9882 at
88.35M) while the identical architecture without the machine stays at chance
(0.503) at every size; and (vii) a training recipe under which the network DISCOVERS a verified-exact
machine itself, overturning four earlier negative attempts, with the discovered
machine giving 0.9997 at 16x length against 0.6451 for the same architecture
without it; and (viii) the bridge: the 120-state machine the 88M model consults is itself induced
from label sequences and verified, and is interchangeable with the given table
(0.9882 at 16x length, both).

We do not claim a general method for learning automata from arbitrary data. The
induction result holds where a coverage-directed sample of the observation
table can be collected, and we report the coverage level below which it fails.

## 1. The bottleneck

State tracking -- maintaining and querying a running finite state across a long
input -- is the load-bearing primitive in the tasks transformers handle worst:
long-horizon symbolic bookkeeping, iterated permutation composition, and
anything whose answer depends on the whole prefix rather than a summary of it.
The S5 word problem is the standard hard case: the group is non-solvable, the
problem is NC1-complete (Barrington), and a constant-depth architecture cannot
solve it at unbounded length.

The empirical form of the bottleneck, reproduced here at every model size we
trained: a transformer trained at length 128 on S5 state tracking reaches
training loss 0.685 and test accuracy 0.5095 at length 2048, against a
constant-predictor baseline of 0.503. Scaling from 1.33M to 88.26M parameters
does not move it (0.5106 -> 0.5095). The failure is not a capacity failure.

## 2. What is verified, and how

Accuracy on held-out strings cannot establish that a learned automaton is the
target: a wrong automaton can agree on every string a sampler emits. Every
correctness claim below is decided by **product BFS over the reachable pairs of
learned and true states**, with no string-length cutoff:

```python
def equivalent(delta, out, T, lab, e, A, start=0):
    seen = {(int(start), int(e))}; q = deque(seen)
    while q:
        a, b = q.popleft()
        for x in range(A):
            a2, b2 = int(delta[x, a]), int(T[x, b])
            if int(out[a2]) != int(lab[b2]): return False
            if (a2, b2) not in seen: seen.add((a2, b2)); q.append((a2, b2))
    return True
```

This replaced a bounded enumeration to length 11 which had reported 6/6 correct;
on 48 deliberate corruptions of a 12-state target the bounded check disagreed
with true equivalence on 22. Those earlier passes were false positives. The
correction is recorded because it changes what the rest of this work means: all
numbers reported here postdate it.

## 3. Induction from I/O pairs alone

Setting: the learner sees sequences of input symbols and, per position, one
observation bit. No states, no hints, no membership oracle, no counterexamples.
It knows the alphabet size and a bound on the number of states. Exact
identification from such data is NP-hard in general (Gold 1978) and hard under
cryptographic assumptions in the representation-independent sense (Kearns &
Valiant), so what follows is a statement about a collectable regime, not about
the worst case.

Three components, each forced by a measured failure of the previous version:

1. **Coverage-directed sampling.** Random and run-shaped samplers left 36.7% of
   observation-table *cells* unobserved at m = 12 while prefix coverage read
   1.0. The fix is to emit concatenations of the table's own prefix/suffix set,
   `p + s`, which uses only the alphabet and the state bound.
2. **Compatibility merging, not row identity.** Requiring identical rows treats
   every gap as evidence of a distinct state; K reached 180 for a 48-state
   target. Merging rows that agree wherever both are observed (the RPNI
   criterion) recovers the exact count.
3. **Majority voting on transitions.** Several prefixes share a state and their
   observed successors disagree; last-write-wins silently picked one, leaving
   transitions wrong while K was already exact.

Result, 3 seeds each, exact equivalence:

| m | 5 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|----|----|----|----|----|----|
| exact | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| K found | 5 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |

Eight further instances across S3 (6 raw / 3 minimal), S4 (24 raw / 4 and 2
minimal), S5 (120 raw / 5 minimal) and MOD-40 (80 raw / 40 minimal): all 3/3.

Head to head at MOD-8 under the one exact criterion, 3 seeds: a Gumbel
relaxation annealed as in the DNAR line 0/3; gradient-free annealing over the
table 0/3; this method 3/3 in 2-8 s.

Perceptual variant (inputs are noisy vectors, symbol identity unknown): exact at
m = 3 and m = 5 (3/3) and m = 8 (2/3), using KMeans initialisation, Viterbi
decoding against the observed labels, alphabet anchoring across rounds, and
margin-weighted per-cell voting.

## 4. Composition

Two machines induced independently from separate I/O streams compose into their
Cartesian product:

```python
T[a*K2 + b, z] = int(T1[a, z1]) * K2 + int(T2[b, z2])
```

Verified over 1368 graph instances: three independently induced REACH machines
composed with an induced PARITY machine compute the target ODD exactly. A MOD3
negative control fails at the first instance, which is what makes the positive
result informative.

State counts are reported after canonical pruning (reachability + minimisation),
because raw state counts overstate: true REACH is 3 states, 2 effective; true
PARITY is 5 states, 3 effective.

## 5. Embedding in a differentiable stack

A hybrid layer takes continuous inputs, selects the machine's initial state by a
per-instance competition across candidate nodes (softmax over nodes with a
straight-through argmax, exactly one selection per instance), and runs the
verified machine. Gradients reach the perception module through a full-rank
interpolation of the two initial-state embeddings whose *value* is exactly zero
(`soft_h - soft_h.detach()`, checked with `allclose`), so the forward pass is
the exact machine and the backward pass is not blocked.

Nine runs (3 noise levels x 3 seeds) against a capacity-matched continuous
baseline with 2.8x more parameters:

| arm | n | held-out acc | drift at N=8 |
|-----|---|--------------|--------------|
| discrete (this) | 9 | 0.9962 | 0.000000 |
| continuous baseline | 9 | 0.5441 | 0.823868 |

Drift is exactly zero in all nine discrete runs at N = 2, 4 and 8. At the
highest noise the residual error is entirely source identification, a perception
limit, and drift stays zero there too. An earlier version of this layer lost
4/9 seeds to initialisation; four restarts selected on *training* loss took it
to 9/9, and that selection rule is part of the method.

## 6. Scale

S5 state tracking, trained at L = 128, tested to L = 2048, readout balanced so a
constant predictor scores 0.503. Two CPU cores, from scratch, 1200 steps,
OneCycle. `machine` means the verified transition table is embedded as an
additive term in the residual stream; it is fixed and never trained.

| params | architecture | machine | train loss | acc L=128 | acc L=2048 (16x) |
|--------|--------------|---------|-----------|-----------|------------------|
| 1.35M | d=128 nl=4 | yes | 0.0001 | 1.0000 | 0.9620 |
| 5.84M | d=256 nl=6 | yes | 0.0000 | 1.0000 | 0.9641 |
| 15.85M | d=384 nl=8 | yes | 0.0000 | 1.0000 | 0.9623 |
| 27.42M | d=512 nl=8 | yes | 0.0000 | 1.0000 | 0.9756 |
| 88.35M | d=768 nl=12 | yes | 0.0000 | 1.0000 | 0.9882 |
| 1.33M | d=128 nl=4 | no | 0.6728 | 0.5106 | 0.5030 |
| 27.36M | d=512 nl=8 | no | 0.6868 | 0.5144 | 0.5105 |
| 88.26M | d=768 nl=12 | no | 0.6852 | 0.5095 | 0.5101 |

The load-bearing observation is the *direction*: with the machine, the
length-generalisation gap narrows as the model grows (0.9620 -> 0.9882); without
it, size does nothing. The 88.35M figure is GPT-2-small's exact architecture and
is reported for reference, not as evidence of large-scale behaviour. The
embedded machine itself is O(L) and was checked exactly at L = 2048, 16384 and
131072; the 2048 ceiling is attention's O(L^2) cost on two cores.

## 7. The bridge

Sections 3 and 6 use the same kind of object but not the same instance: the
transformer consulted a machine that was *given*, while the induction result was
demonstrated on separately constructed targets. The bridge closes that: the
120-state S5 machine the transformer consults is induced from label sequences
only, verified by product BFS, and then embedded.

**Induction.** The learner sees token sequences over the 120-symbol alphabet
and, per position, the one-bit observation. The observation table is built with
prefixes of length <= 1 and suffixes of length 1; transitions are read off by
matching the row of `p + (a,)` against the representative rows, which is why the
data must extend to length-3 words `p + a + s` -- an earlier version left
transitions at zero and product BFS correctly refused equivalence. The strings
collected are exactly cover-set concatenations, chosen from the alphabet size and
the state bound alone.

| cells observed | seeds | K found | true states | exact equivalence |
|----------------|-------|---------|-------------|-------------------|
| 1.00 | 1 | 120 | 120 | **yes** |
| 0.50 | 3 | 111-114 | 120 | no (0/3) |
| 0.25 | 3 | 58-60 | 120 | no (0/3) |
| 0.10 | 3 | 20-21 | 120 | no (0/3) |

The S5 observable has no redundancy at this suffix depth: every cell is
load-bearing, and thinning the table costs exactness immediately rather than
gradually. That is the boundary, stated rather than hidden -- the method needs to
be able to choose its strings.

**Readout agreement, no training involved.** Running the induced machine and the
true machine on the same random inputs and comparing their output bits:

| L | 2048 | 16384 | 131072 |
|---|------|-------|--------|
| agreement | 1.0000 | 1.0000 | 1.0000 |

**Embedded.** The transformer is retrained with the induced machine's states
injected in place of the true ones. Induced state ids are a relabelling of the
true ones; the embedding table is indexed by induced id, and the training labels
are the task's own.

| params | machine | acc L=128 | acc L=2048 (16x) |
|--------|---------|-----------|------------------|
| 1.35M | given | 1.0000 | 0.9620 (3 seeds) |
| 1.35M | **induced** | 1.0000 | **0.9617** (0.9708 / 0.9471 / 0.9673) |
| 88.35M | given | 1.0000 | 0.9882 |
| 88.35M | **induced** | 1.0000 | **0.9882** |

The induced machine is interchangeable with the given one at both ends of the
ladder, to the resolution of these runs. The chain is therefore closed: label
sequences -> verified minimal automaton -> embedded in an 88.35M transformer ->
16x length generalisation that improves with scale, with a same-size control at
chance.

## 7b. A second task, with a non-trivial control

The binding limit on Section 6 is the task: S5 is a group word problem, and its
control sits at chance, so the comparison is trivially won. NL-CONNECTIVITY
replaces it. Prose over an 8-word vocabulary ('<entity> links <entity> .', with
'reset .' mixed in); dense label per position: are entity 0 and entity 1
currently in the same component. Nothing is pre-parsed -- the machine runs on the
token stream and the parse state is part of it. The core is undirected
connectivity, which as union-find is order-independent and exact in one pass, so
it is a genuine DFA: 312 raw states, 152 minimal. Constant predictor 0.5204; a
'did the pair ever appear' heuristic 0.5327.

**Induction.** From labels alone, verified by product BFS: K = 152, equal to the
target's minimal count, exact, with 49 suffixes and no oracle. Two corrections
were forced by refused verifications. Sampling prefixes does not close the table
-- 64,456 sampled prefixes still left 16-32 transitions undefined, because a
state discovered only by an extension has no extensions of its own -- so the
cover set is instead one representative prefix per discovered state, closed
iteratively under the alphabet (149 prefixes rather than 64k). And a fixed
suffix set under-separates: two sentences' worth gave K = 149 against minimal
152, with zero unresolved transitions, a wrong merge a state-count check would
have called nearly right; the suffix set is therefore grown by consistency
repair (15 suffixes added). Unlike Section 7, this induction does **not** need a
complete observation table, which is the more general regime.

**Trained**, L = 128 -> 2048 (16x), one seed per rung except where noted:

| params | arm | L=128 | L=2048 (16x) |
|--------|-----|-------|--------------|
| 1.32M | none (control) | 0.9155 | 0.5848 (3 seeds) |
| 1.36M | shuffled machine (negative control) | 0.8984 | 0.6033 (3 seeds) |
| 1.36M | given machine, 312 states | 0.9999 | 0.9416 (3 seeds) |
| 1.36M | **induced machine, 152 states** | 1.0000 | **0.9998** (3 seeds) |
| 5.78M | none | 0.8989 | 0.6077 |
| 5.86M | given | 1.0000 | 0.9592 |
| 5.86M | **induced** | 1.0000 | **1.0000** |
| 27.3M | none | 0.9099 | 0.6191 |
| 27.5M | given | 1.0000 | 0.9766 |
| 27.5M | **induced** | 1.0000 | **1.0000** |
| 88.2M | none | 0.9167 | 0.5028 |
| 88.4M | given | 1.0000 | 0.9896 |
| 88.4M | **induced** | 1.0000 | **1.0000** |

Three readings. The control is not at chance: it learns the task in distribution
(0.90-0.92) and decays with length, and scale does not repair that -- 0.5848 at
1.32M, 0.6191 at 27.3M, 0.5028 at 88.2M, with the largest control the best
in-distribution and the worst at length. The shuffled-machine arm scores like no
machine, so the gain is the machine's information rather than the embedding's
parameters. And the induced machine beats the given one, because induction
returns the minimal machine: 152 Myhill-Nerode classes instead of 312 raw
states, so the readout is a smaller lookup. Minimality is load-bearing, which we
did not predict.

What this does not show: with the exact state supplied, the label is a function
of that state, so the model's remaining job is a lookup, and the interest lies
entirely in the control, which must compute the state itself and cannot hold it
over length. The machine is still induced separately and injected, not
discovered during training. The prose is templated rather than natural text,
there are five entities, and there is still only one model class.

## 7c. The discovery step: the network finds its own machine

Sections 3-7b induce the machine by search and inject it. The stronger claim is
that ordinary training finds it. Four routes failed to do that, and their
failures specified the fix.

**The failures.** Clustering a trained control's hidden vectors aligns with the
true state classes at 0.29 (training length) and 0.16 (16x), and a table read
off those clusters is not equivalent. A discrete state slot trained on labels is
0/3 exact at three state counts, and 0/16 with the gradient arm tuned as hard as
the method (bare slot, 8 restarts selected on training loss, K set to the exact
minimal count). Prefixes at the same true state get identical observation rows
from the control only 21-25% of the time, so its behaviour does not refine the
target's abstraction; and merging its clusters lands below the minimal state
count every time (0/6), since merging only coarsens. The condition a working
recipe must meet is therefore: produce a partition that *refines* the target's.

**Two changes meet it.** First, the objective rather than the optimiser: the
state slot predicts, from its state alone, the labels the sequence would receive
under a bundle of probe continuations -- its Myhill-Nerode row -- so separation
is what the loss rewards. A row entry is the label of prefix+suffix, the same
quantity the induction pipeline collects, not extra supervision. Second, the
data must cover the strings the verifier explores: with grammatical prose alone,
training loss reached 0.0 and the readout reproduced every held-out label while
product BFS still refused, because behaviour on strings the generator never
emits was unconstrained. Half free token strings, half prose fixes it. Two
smaller corrections followed from measurements: interleave sequence lengths
rather than staging them (a staged curriculum forgets the short gradient path),
and hold the straight-through temperature constant at 1.0 rather than annealing
(the forward pass is a hard argmax at any temperature, so annealing only
sharpens the surrogate until the gradient vanishes: final loss 0.081 annealed
against 0.042 constant).

**Result**, product-BFS equivalence, 3 seeds per rung:

| target minimal states | slots offered | exact | training loss on the exact runs |
|---|---|---|---|
| 11 | 32 | 2/3 | 0.0 |
| 38 | 128 | 1/3 | 0.0 |

Exactness tracks the training objective: every run reaching loss 0.0 verified
exact and every run that stalled did not, at both scales. The recipe therefore
knows when to restart without consulting a verifier, which is the property that
matters where no true machine exists to check against.

**The chain with nothing supplied.** Embedding the discovered 38-state machine
in the transformer, trained at L = 128 and tested at L = 2048:

| arm | L=128 | L=2048 (16x) |
|-----|-------|--------------|
| discovered machine | 1.0000 | 0.9997 (3 seeds) |
| none | 0.9197 | 0.6451 (3 seeds) |

This overturns the negative result above on these scales: gradient descent on
labels alone can land on a verified-exact machine when the objective asks for
future-label separation and the data covers the alphabet. It does not show the
recipe scales further, and the probe bundle remains a design choice with the
same coverage question that bounds the induction pipeline. An environment that
will not answer arbitrary probe queries is untested and is the honest frontier.

## 7d. A real pretrained model, with the state supplied in the prompt

Every result above trains from scratch on a synthetic vocabulary. This section
asks the question a reader asks first: does a pretrained language model show the
same degradation over length, and does the verified state recover it?

Entities are connected by English sentences ("Alice knows Bob."), with a reset
sentence; the question is whether two named people are connected. Cases are
balanced 50/50, four worked examples precede each question, and the score is the
yes-minus-no log-probability margin measured by ROC AUC, so a model with a fixed
answer bias is judged on whether it RANKS connected above unconnected rather
than on which word it emits. Three conditions, three seeds, 80 cases per cell.
The third condition is the control that matters: a state sentence of identical
shape carrying a randomly chosen group.

Qwen2.5-0.5B-Instruct, mean over three seeds:

| sentences | plain | with_state | wrong_state |
|-----------|-------|------------|-------------|
| 8 | 0.796 | 0.974 | 0.569 |
| 16 | 0.804 | 0.986 | 0.598 |
| 32 | 0.669 | 0.913 | 0.565 |
| 64 | 0.734 | 0.971 | 0.537 |
| 128 | 0.581 | **1.000** | 0.421 |

`with_state` beats `plain` in 15 of 15 paired cells, smallest gain +0.141;
`wrong_state` beats it in 0 of 15 and is consistently worse, by 0.10 to 0.23.
Plain falls from 0.800 at 8-16 sentences to 0.658 at 64-128 while with_state
holds (0.980 to 0.985), so the gap widens with length exactly as it does on the
from-scratch ladder. The model's confidence collapses on the way: mean margin
0.128 plain against 0.554 with_state at 128 sentences.

Pythia-160m shows no consistent effect (with_state ahead in 9 of 15 cells,
wrong_state in 6 of 15). It does not track the task well enough at any length
for a state to rescue, which bounds the claim to models that can do the task at
all.

Two things this does and does not settle. It settles that the degradation is not
an artefact of training from scratch on a toy vocabulary, and that the benefit
comes from the information in the state rather than from the presence of an
extra sentence. It does not test induction or discovery here, and the state
arrives in the prompt rather than in the residual stream, which is the cheapest
interface available and also the weakest.

A note on measurement, because two earlier versions of this experiment were
wrong in instructive ways. Scoring accuracy on an unbalanced set gave 0.5667 for
a model that answered one word to all 180 questions, which is the class balance
wearing the costume of a result. Balancing the set and calibrating the threshold
made that failure visible (exactly 0.50, flagged) but still could not see a model
that ranks correctly while answering one word, because thresholding discards
ranking. AUC is what made the signal measurable, and the wrong-state control is
what made it attributable.

## 8. Limits

- Induction requires a coverage-directed sample of the observation table. With
  the table's cells randomly thinned, exactness fails and the induced state
  count degrades gracefully rather than erring loudly (see Section 7 table).
- Section 7d uses a real pretrained model on English prose; the rest is
  synthetic, and the prose in Section 7b is templated. No result
  here is evidence about language modelling, and in both cases the machine is
  induced separately and injected in Sections 3-7b. Section 7c removes that for
  the 11- and 38-state targets, where the network discovers a verified-exact
  machine itself; at 152 states this is untested at the time of writing.
- Only one model class is tested. A state-space or hybrid-attention baseline is
  needed before the comparison can be called architecture-independent.
- The 2048 test ceiling is a compute limit, not a property of the method.
- Several effects we measured crossed conventional significance at n = 3-24 and
  vanished at larger n. Three separate claims died that way during this work
  (a normalisation variant, an initialisation-collapse predictor, and a
  quantisation effect). We report it because the surveyed literature in this
  area commonly reports three seeds.

## 9. Related work, with verification status

- Barrington (1986), bounded-width branching programs and NC1-completeness of
  S5 word problems. [VERIFIED -- standard result]
- Liu et al. (2023), shortcut solutions to automata learning by transformers;
  Merrill et al. (2024), "The Illusion of State in State-Space Models".
  [UNVERIFIED -- cited from memory of the literature, not re-fetched here]
- Gold (1978), NP-completeness of minimum-state identification; Pitt & Warmuth;
  Kearns & Valiant, representation-independent hardness. [VERIFIED -- standard]
- Angluin (1987) L*, and RPNI/EDSM state merging. The merging rule in Section 3
  is RPNI's compatibility criterion, not a new criterion. [VERIFIED -- standard]
- Rodionov & Prokhorenkova, DNAR (ICML 2025), arXiv 2402.11628: finite
  predefined states with hard attention; the no-hint variant reports graph
  accuracy 0 at 64 nodes on BFS. [VERIFIED -- id checked against the paper as
  read during this work]
- GraphFSA (M, Z, T) model class. [UNVERIFIED]

A note on method: the prior-art searches during this work were run on the
mechanisms already in hand rather than on the bottleneck itself, which delayed
finding DNAR by several stages. Searching the bottleneck first is the cheaper
order.

## 10. What would change the picture

In rough order of value: a task with a discrete computational core that someone
outside this line of work already cares about (modular arithmetic embedded in
natural-language chains, graph reachability from text queries, small formal
language parsing); a second model class as a baseline; induction under
distribution shift, where the coverage-directed sampler cannot choose its own
strings; and the machine being *found* by the network rather than supplied.
