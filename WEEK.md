THE WEEK
========
Seven days. The goal is not to prove a paradigm shift, which is something other
people do to your work over years. The goal is to put the strongest possible
seed in the ground and remove every reason for someone not to pick it up.

Your professor named the two things actually under your control: a name people
can say, and code people can adopt in five lines. Both are built. This week is
about shipping them and putting them in front of the right people.


DAY 1 - SHIP THE PACKAGE
------------------------
The library exists, is tested, and builds. It is called statelock.

 1. Make a GitHub account if you do not have one. Create a public repository
    called statelock.
 2. Upload the statelock folder exactly as it is.
 3. Make a PyPI account at pypi.org. Then, in a terminal or in Colab:
      pip install twine
      twine upload dist/*
    It will ask for your PyPI token. After that, anyone in the world can run
    pip install statelock.
 4. Check it worked: in a fresh Colab notebook, run
      !pip install statelock
      then paste the five lines from the README and run them.

Why this is day 1: nothing else you do this week matters if the code is not one
command away. Attention spread partly because it was twenty lines. Every step
between a reader and a working example loses most of the readers.


DAY 2 - REPRODUCE ON A GPU
--------------------------
 1. Open colab.research.google.com, new notebook, Runtime, Change runtime type,
    T4 GPU.
 2. Run:
      !pip install statelock
      !git clone https://github.com/YOURNAME/statelock
      !python statelock/tests/test_all.py
    Expect four lines: induction finds 152 states and exact True, forty out of
    forty corruptions flagged, the machine exact at length 4096, and a discovery
    run that verifies.
 3. Screenshot that output. It goes in the README, under a heading that says
    "reproduced on a T4".
 4. Send me anything that differs.


DAY 3 - THE EXPERIMENT THAT DECIDES THE CLAIM
---------------------------------------------
Run step2_real_model.py on Colab. It asks whether a real pretrained model, one
that already knows English, loses the state over a long passage and whether
supplying it helps.

This is the single most important thing in the week. Everything so far uses
models trained from scratch on invented sentences. If a real model shows the same
weakness, the claim is about language models. If not, the claim is about small
from-scratch models, and the paper has to say so.

I could not test that script here, because this machine cannot download models.
If it errors, send me the red text and I will fix it the same day.


DAY 4 - WRITE THE PAPER DOWN
----------------------------
PAPER.md is already written and covers everything. Turn it into a PDF and put it
on arXiv.

 1. Paste PAPER.md into Overleaf with any article template, or export it to PDF
    from a markdown editor. It does not need to be beautiful. It needs to exist.
 2. arXiv, cs.LG. First submission needs an endorsement, which any professor in
    the field can give in two minutes.
 3. Add the Day 3 result before submitting if it is in, and say plainly which
    way it went.


DAY 5 - ONE PAGE THAT MAKES ADOPTION OBVIOUS
--------------------------------------------
Write a single page on the repository front page answering, in this order:

 1. What breaks. A model at 0.92 in distribution and 0.50 at 16x length, and
    scaling it makes that worse, not better.
 2. What fixes it. Five lines of code.
 3. Why you can trust it. Not a test-set number. Exhaustive equivalence, with
    the verifier catching every one of forty corruptions.
 4. Where it applies. Entity tracking across a document, permission and protocol
    state, scope and bracket structure, small formal languages, game state.
 5. Where it does not. No discrete core, no benefit.

Keep it under a page. The point is that someone deciding in ninety seconds can
decide yes.


DAY 6 - PUT IT IN FRONT OF PEOPLE WHO CAN USE IT
------------------------------------------------
Send five short emails. Not a mailing list, five individuals whose own work is
about state tracking, length generalisation, or neural algorithmic reasoning.
Find them by looking at who wrote the papers cited in the related-work section,
and use the contact address on their most recent paper.

Template, keep it this short:

  Subject: verified state machines inside transformers, pip install statelock

  Dear <name>,

  Your work on <specific thing> is why I am writing. I have a small result you
  might find useful or might want to break.

  A transformer on a prose entity-tracking task holds 0.92 in distribution and
  falls to 0.50 at 16x length, and going from 1.3M to 88M parameters makes the
  length number worse. Attaching a finite machine whose correctness is checked
  by exhaustive equivalence, rather than by a test set, holds it at 1.0000. The
  machine can be induced from labelled strings, and in the smaller cases the
  network finds it during training and the found machine passes the same check.

  It is pip install statelock and a five-line example. Paper: <arXiv link>.

  If it is wrong I would rather know quickly. The one file to attack is
  tests/test_all.py.

  <your name>

Then post the same thing publicly once, with the numbers and the install line,
on whichever platform the people you want actually read.


DAY 7 - MAKE THE NEXT PERSON'S JOB EASY
---------------------------------------
Open three issues on your own repository, titled as work you want done:

 1. "Raise the discovery success rate above one in three at 38 states."
 2. "Discovery at 152 states and beyond."
 3. "Discovery where probe queries are unavailable."

Label them help wanted. This sounds trivial. It is how a repository stops being
a paper attachment and starts being a project. The third issue is the real open
problem and stating it honestly is more attractive to a serious person than
claiming it is solved.


WHAT THIS WEEK CANNOT DO
------------------------
It cannot make the work a paradigm shift. That happens when a second group uses
statelock for a problem you never touched, and a third group cites the second.
You cannot schedule that. What you can do is make it as easy as possible and put
it where those people are, which is what days 1, 5, 6 and 7 are for.

What you can honestly say at the end of the week, if the days go well:

  "A transformer loses entity state over length and scaling makes it worse. A
  verified finite machine fixes it, the machine can be induced or discovered,
  correctness is exhaustive rather than statistical, and it is one pip install."

That sentence is true today, checkable by anyone in ten minutes, and is the
thing that gets picked up. It is not a claim about being a paradigm shift, and
claiming that in the email would cost you the reader you are writing to.


ON THE NAME
-----------
statelock is a working name, chosen because it is short, sayable, and unclaimed
on PyPI. Others considered: verified state layer, too descriptive to spread;
Myhill layer, accurate and meaningful to the subfield but opaque outside it;
state anchor, also fine. If you prefer one of those, change it on day 1 and not
after, because the name is worth almost nothing on day 1 and a lot on day 30.
