# Defending the paper: the three claims reviewers will press

Read this after working through `REPRODUCTION_CHECKLIST.md`. The point is not to
memorise answers but to make sure each of you can explain the mechanism from
first principles, because a reviewer who suspects an author does not understand
their own central claim will say so.

---

## Claim 1 — "The reported PTQ collapse was a quantizer defect, not a property of PTQ"

This is the paper's most consequential and most attackable claim, because it
reverses a published position.

**The mechanism, in one paragraph.** Symmetric INT8 quantization maps a tensor's
range onto `[−128, 127]` using a single scale and no zero point. That is correct
only when the tensor is roughly symmetric about zero. The legacy implementation
computed `scale = (max − min)/255` and then rounded `x/scale` into
`[−128, 127]`. For a strictly positive tensor, `(max − min)` describes a range
that sits entirely on one side of zero, so the scale is roughly twice what a
symmetric quantizer needs, and every value maps into the upper half of the grid
and saturates. BatchNorm's γ is strictly positive and clusters near 1.0, so it is
exactly the worst case: γ with mean 1.0093 collapses to a **single value**,
0.2752 — 27% of true scale. Every channel is then scaled identically and wrongly,
which destroys the network. Folding γ into the preceding convolution before
quantizing, which is standard practice, removes the tensor entirely and the
failure disappears.

**Anticipated challenge:** *"You are attacking a strawman implementation."*

Your answer: the legacy condition is not a strawman, it is a faithful
reproduction of the code that produced the previously published number, retained
as a control and run on identical frozen splits. The validated conditions use
`torch.ao.quantization` with per-channel symmetric weights and BN folding under
two backends. The comparison is like-for-like on 20 seeds.

**Anticipated challenge:** *"Maybe QAT still helps on larger models."*

Concede this immediately. The claim is scoped to models of this size on tabular
data. Say so plainly; do not defend a broader claim than you made.

---

## Claim 2 — "Our architecture offers no accuracy advantage over logistic regression"

**Anticipated challenge:** *"Then why publish the architecture at all?"*

The honest answer, which is in the paper: what the network provides is a
fixed-shape integer graph with a known operator set and known size. A
scikit-learn model has none of those properties, and they are what matter on a
target with tens of kilobytes of Flash and no floating-point unit. The
contribution is deployability, characterised against an exported artifact, plus
two negative results the field should have.

Do not attempt to rescue an accuracy claim under review. The paired tests
(p = 0.625 on PIMA, p = 0.002 against you on CKD) are in the paper; retreating
from them after a reviewer pushes would be far worse than stating them up front.

**Anticipated challenge:** *"UCI-CKD results are meaningless, every model is at
ceiling."*

Agree — the paper says exactly this. It is included because it is a standard
benchmark, and the paper states explicitly that it cannot discriminate between
methods.

---

## Claim 3 — "Parameter-count footprints understate real Flash by 4.7×"

**The mechanism.** The flatbuffer stores far more than weights: per-tensor and
per-channel quantization parameters, the operator graph, tensor metadata, and
buffer offsets. At 4,294 parameters those fixed costs dominate. The paper reports
both the stripped file (20,056 bytes) and the unstripped one, so the reader can
see how much is tensor names alone.

**Anticipated challenge:** *"You never ran it on hardware, so this is not a
TinyML result."*

This is fair and the paper concedes it in three places, including a bolded
sentence in Sect. 5.4 and a limitation. If you can borrow any Cortex-M board
before submission, measured latency and arena usage would convert this from a
static characterisation into a deployment result and would materially strengthen
the paper. If not, do not overclaim.

---

## Two weaknesses to raise before a reviewer does

**The synthetic cohort uses a 5,000-row subsample** while the real cohorts use
their full data. All three now use 10 repeats of stratified five-fold CV, so the
protocol is consistent. The subsample is defensible and stated in the caption:
this cohort's labels are near-deterministic in HbA1c, so additional rows carry no
additional information. Be ready to say that rather than appearing to have
subsampled for convenience.

**Fairness is restricted to age.** PIMA is all female, UCI-CKD records no sex,
and the only cohort with usable sex data is synthetic. The paper says this, but
be ready for the follow-up: the most consequential fairness question for a
screening tool deployed in low-resource settings is ethnicity, and none of these
cohorts record it.

---

## If asked why the earlier version was wrong

Answer directly and without defensiveness. The cover letter lists all eight
corrections. The strongest position is that you found the errors yourselves,
reproduced each one in a public repository, and reversed your own headline claim
on the evidence. That is a better story than a paper with no history, provided
you do not appear to be discovering the corrections for the first time in the
review.
