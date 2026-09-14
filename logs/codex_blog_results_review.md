Reading additional input from stdin...
OpenAI Codex v0.153.4
--------
workdir: /data/users/asvin/self_inoculation
model: gpt-6-astra
provider: openai
approval: never
sandbox: read-only
reasoning effort: none
reasoning summaries: none
session id: 01a09191-05ca-7f40-b6fd-8ce68453bb78
--------
user
Read logs/blog_results_section_for_review.md (read-only). It is the results section and appendix of a blog post for a technically sophisticated general audience (LessWrong / Alignment Forum readers who know transformers, HMMs, Bayesian filtering and gradient descent). They have read the earlier sections, which set up: a toy in which each token is a pair (x, y); x comes from HMM1 whose hidden-state tracker is the 'shared circuit'; y comes from one of two HMMs fixed per sequence defining 'situation A' and 'situation B', inferred from context; pretraining on both; then finetuning where in situation A two x symbols are swapped in the targets ('the swap'), in two variants: situation-A-only data, or mixed A+B data. They have NOT seen any numbers, figures, or the terms 'flip fraction', 'block', 'probe', 'erase', 'graft', 'patch'. Task: as that reader, list as a numbered list every point that is unclear, undefined, context-presupposing, uses a symbol or term before its role is stated, or makes a claim whose evidence is not visible in the text or referenced figure captions. Also list any sentence that would read as over-claiming to a skeptical reader given only what the section shows. Quote the phrase for each. Do not rewrite anything; do not comment on style or voice.
codex
I’ll read the section and flag gaps in definitions, context, and visible evidence, using only the background you specified.
exec
/usr/bin/bash -lc 'cat logs/blog_results_section_for_review.md' in /data/users/asvin/self_inoculation
 succeeded in 0ms:
## What we find

Everything below is for the version of HMM1 whose belief geometry is in the figure above, three training seeds each; a second version with a much simpler HMM1 gives the same picture and is in the appendix. The one number I will keep using is the *flip fraction*: how far the network's x predictions have moved from the pretrained behaviour (0) towards the fully swapped one (1), measured separately on situation-A and situation-B sequences.

**1) We do find a shared circuit.** After pretraining, the network tracks HMM1 the way Simplex's earlier work would lead you to expect: the Bayesian posterior over HMM1's hidden state can be read out linearly from the residual stream, with increasing fidelity through the layers (R² 0.86 after the first block, 0.99 after the last), and it is the same tracker in both situations. The directions that carry it are the same whether the y half of the token is coming from situation A or B, and if we erase them (project out the subspace that the true posterior writes into) x prediction collapses to chance by the same amount in both situations, +0.148 nats, i.e. exactly to uniform guessing, while y prediction is untouched. The situation itself is also there to be read: the posterior over A versus B becomes linearly decodable over the first three blocks, in a subspace nearly orthogonal to the HMM1 tracker (overlap below 0.1). So the pretrained network has the ingredients laid out separately, and as far as we can tell nothing connects them yet, since nothing in pretraining rewarded a connection.

**2) Training only in situation A gives emergent misalignment.** When we finetune with the swapped labels in situation A and no situation-B data at all, the network does the obvious thing. Within about a hundred steps its x predictions are fully swapped, and they are swapped in situation B just as much as in A. On held-out situation-B sequences, which it never saw during finetuning, the flip fraction is 1.00 and its loss on the original task rises from the Bayes optimum of 0.953 nats to 1.243, matching the 1.245 we computed in advance for a network whose HMM1 readout is swapped everywhere. (This is worse than uniform guessing, which would score 1.099: a confidently wrong network.) Transplanting situation-B information into these networks' activations does nothing to their x predictions; there is simply nothing conditional to find. The red trajectories in the figure below are this network, three seeds, walking straight up the diagonal where A and B are equally flipped. A small aside: with no B data, the network's model of situation B's own y-process quietly degrades as well (by 0.34 nats), a reminder that not training on something is not neutral.

[figure: figs_h/f_square.png, the finetuning trajectory of every run in the plane (flip in B, flip in A). Red: situation-A-only finetuning. Blue: mixed finetuning. Star: perfect self-inoculation.]

**3) Mixed finetuning gives self-inoculation, in the shape we guessed, after a detour through emergent misalignment.** With situation B kept in the finetuning data (with its original labels), the endpoint is what the hypothesis predicts: flip fraction 0.98 to 0.99 in situation A, 0.01 to 0.02 in situation B, and both losses back at the Bayes optimum (within 0.002 nats). These are the blue trajectories above. Three things about how it gets there.

*It is a switch on the shared tracker, not a second tracker.* Erasing the HMM1 subspace still kills x prediction equally in both situations, so the tracker is still shared and still used by both. If we take the pretrained network and graft in only the finetuned weights of blocks 3 and 4, we recover essentially the whole behaviour (flip 0.94 to 0.97 in A, at most 0.08 in B); grafting the finetuned embedding, unembedding, or blocks 1 and 2 does nothing at all; and putting the pretrained blocks 3 and 4 back into the finetuned network removes the switch (flip in A drops to 0.12 to 0.20). The switch reads the situation, and reads it early: if we transplant the situation-B component of the activations into a situation-A run at block 1 or 2, keeping the x half of the input identical, the network's x predictions turn unflipped (0.99 to 0.02); doing the same after block 3 does nothing. So the switch sits in block 3, which is also where the situation posterior first becomes linearly readable, and it flips the tracker's output rather than the tracker.

[figure: figs_h/restoration_calibration.png, left: flip fractions after swapping weight groups between the pretrained and finetuned networks; right: per-sequence flip fraction against the Bayesian posterior.]

*It flips by exactly how sure it is.* At each position in the sequence, the flip fraction in situation A lies on top of the Bayesian posterior that the sequence is from A (mean difference below 0.01), and sequence by sequence, the sequences an ideal observer would be 60% sure about get flipped by 60% (slope 0.99, mean error 0.04, right panel above). A model that is only partly sure it is being tested misbehaves only partly, and in proportion.

*Emergent misalignment happens first, then gets rolled back.* For the first fifty to seventy-five steps the two finetuning variants are nearly indistinguishable: both swap the shared readout for both situations at once, and situation B's loss rises by about 0.04 nats, roughly 12% of what a full swap would cost it. Only then does the variant with B in the data bend back: between step 75 and 125 the situation-B flip returns to zero and the situation-A flip completes. Gradient descent takes the shortcut first and builds the switch only once the shortcut has started to hurt. This is the part of the result I find most interesting for the real-world story: the misbehaviour did leak into the protected situation before the switch existed, by an amount set by how much the shortcut hurts there, and the switch was built in about fifty steps once it did.

[figure: figs_h/zoom_first300.png, the first 300 finetuning steps for both variants: flip fractions, situation-B damage, situation-A loss.]

One more thing I did not predict. After mixed finetuning, the situation posterior is *more* linearly readable in the residual stream than it was after pretraining (R² 0.83 to 0.94–0.97), and after A-only finetuning it is less (down to 0.41–0.53). The situation variable acquired a consumer in one case and lost one in the other, and its representation followed. If anything like this holds in real models, training that teaches a model to behave differently in tests also sharpens its ability to tell that it is in one.

## Appendix: details and the less exciting data

*Setup.* HMM1 is a Mess3 process with α = 0.7 and a leave-probability of 0.03 per alternative state (a last-token predictor recovers only half the learnable signal, so the tracker genuinely integrates history); the two situation processes are Mess3 with α = 0.85 and leave-probabilities 0.15 (A, sticky) and 0.5 (B, never stays). Tokens are the 9 pairs plus a start token; sequences are 64 tokens; the model is a 4-block GPT with d = 120 and 4 heads, trained with AdamW at 5e-4, batch 256, fresh samples every step. Pretraining is 10k steps on both situations; each finetuning variant is 10k steps from the same checkpoint with a fresh optimizer, checkpointed every 25 steps early on. The A-only variant uses batch 128 so that it sees as many situation-A sequences per step as the mixed variant. All losses are compared with the exact Bayesian predictor on the same held-out sequences (4096 per situation).

*A simpler HMM1 gives the same result.* We first ran everything with HMM1 = Mess3(α = 0.9, leave 0.1), for which the current token already carries 94% of the learnable signal. Every qualitative finding is the same: mixed finetuning ends at flip (A, B) = (0.98, 0.02) with both losses at optimum; A-only finetuning ends at (1.00, 1.00) with the situation-B loss at 1.360, the pre-computed value for a swapped readout being 1.3599; the two variants coincide for the first ~50 steps and separate between steps 75 and 125; the switch is carried by blocks 3 and 4 (grafting them alone onto the pretrained network recovers 82 to 91% of the A-versus-B contrast); per-sequence calibration has slope 0.96 to 1.00 and mean error 0.03. In this simpler version a further probe result is clean: the belief probe fitted on the pretrained network, applied after mixed finetuning, still reads situation-B activations in the original coordinates from block 3 onward (R² 0.86 to 0.94) but reads situation-A activations only after relabelling the coordinates by the swap (R² 0.57 to 0.84), i.e. the switch is a situation-conditional relabelling of the tracker's output coordinates. With the history-heavy HMM1 the finetune moves the tracker's coordinates in both situations and this transfer test becomes uninformative, so the localisation there rests on the patching and weight-swapping results, which agree with the simpler version.

*The leak before the switch.* Situation B's loss excess peaks at 0.047 to 0.057 nats (simple HMM1) and 0.034 to 0.040 nats (history-heavy HMM1) around step 75 to 100, i.e. 9 to 14% of the cost of a full swap in each case, and is back within 0.01 nats of optimum by step 125 to 150 in every seed.

*A prediction that failed.* I expected the weight change to concentrate in the last block and the unembedding. It does not: at the moment the switch exists, the relative change in weights is spread across all four blocks (0.09 to 0.17), and only the interventions above localise the switch. Reverting the embedding or the unembedding to their pretrained values changes nothing, so those changes are real but functionally irrelevant.

*What the toy does not show.* Nothing here involves an actual RL objective or a KL penalty; the 'incentive to stay itself' is an explicit loss on situation B. Nothing here distinguishes an in-place relabelling in block 3 from a small conditional branch in blocks 3 and 4 that writes an equivalent correction. The situations never switch within a sequence. And the ideal-observer posterior is a stand-in for whatever situational awareness a real model has; the toy shows what gradient descent does once such a variable exists and is being used.

*Preregistration and code.* The design, predictions and their outcomes (including the failed one) are in `PREREG.md` and `RESULTS.md` in the repository, together with two adversarial reviews of the write-up by a different model; the figures above are `figs_h/*.png`, and the belief-geometry figure is `sim_h/geometries.png`.


codex
Based on the supplied text and figure captions, taking the earlier setup you described as known:

1. **“whose belief geometry is in the figure above”** — The version is identified by a figure whose caption is not included here. The text does not explain which feature of that geometry matters for interpreting these results.

2. **“three training seeds each”** — Does each seed cover pretraining and finetuning, or only finetuning? Are the two finetuning variants paired by pretrained checkpoint? Later numerical ranges are not identified as seed ranges, uncertainty intervals, or something else.

3. **“the *flip fraction*: how far the network’s x predictions have moved”** — The endpoints are defined, but the measurement is not. What formula maps a predicted distribution to a fraction? Is this a projection onto the difference between original and swapped distributions, a fitted mixture coefficient, or another statistic? Can it fall outside [0, 1], and how are positions with identical original and swapped predictions handled?

4. **“from the pretrained behaviour (0) towards the fully swapped one (1)”** — Are the reference distributions the actual pretrained network’s predictions or exact Bayesian predictions? “Fully swapped” could mean swapping the pretrained predictions or using the Bayes-optimal swapped predictor.

5. **“measured separately on situation-A and situation-B sequences”** — How are token-level measurements aggregated over positions and sequences? This matters when the text later moves between per-position, per-sequence, and whole-dataset flip fractions.

6. **“Simplex’s earlier work”** — The work is not identified or linked, and its particular prediction for this experiment is not stated.

7. **“read out linearly from the residual stream … R² 0.86 … 0.99”** — The readout procedure is unspecified: what is fitted, on which data, and evaluated on which held-out data? How is R² computed for a vector-valued posterior, and are these readouts trained jointly across situations or separately?

8. **“after the first block … after the last”** — “Block” first appears without identifying it as a transformer block or stating how many there are. The four-block architecture appears only in the appendix. The precise residual-stream measurement point within a block is also unspecified.

9. **“it is the same tracker in both situations” / “The directions that carry it are the same”** — What comparison establishes sameness: cross-situation readout transfer, subspace overlap, or something else? Linear decodability in both situations alone would not establish a shared implementation.

10. **“erase them (project out the subspace that the true posterior writes into)”** — How is this subspace constructed, and where is the projection applied? A posterior is a target quantity, so “writes into” does not identify an operational procedure. In particular, linear decoding directions and directions encoding a variable need not be the same.

11. **“x prediction collapses to chance by the same amount in both situations, +0.148 nats, i.e. exactly to uniform guessing”** — Is this an increase in marginal x cross-entropy, averaged per token? Matching uniform guessing’s average loss does not establish that the output distributions are uniform. “Exactly” also leaves the numerical tolerance and seed variation unclear.

12. **“while y prediction is untouched”** — Which measurement is unchanged, and within what tolerance? Unchanged average y loss is weaker than unchanged y predictions.

13. **“the posterior over A versus B becomes linearly decodable over the first three blocks”** — Is the target the ideal Bayesian posterior given the observed prefix, or the sequence’s actual situation label? Does “over the first three blocks” mean progressively improving decodability or first becoming decodable at block 3?

14. **“in a subspace nearly orthogonal … (overlap below 0.1)”** — “Overlap” has no definition or scale. The dimensions of the two subspaces and the method used to estimate them are also missing.

15. **“nothing connects them yet, since nothing in pretraining rewarded a connection”** — This overstates the evidence. Separately decodable, nearly orthogonal representations do not rule out interactions, and an absence of training reward for a connection does not establish its absence.

16. **“Training only in situation A gives emergent misalignment”** — The operational result is transfer of the swap into B. Given only the specified setup, the criterion for calling that “misalignment” has not been stated.

17. **“Within about a hundred steps its x predictions are fully swapped”** — Is “fully swapped” established by the full predicted distributions or only by an aggregate flip fraction near 1? A single scalar can conceal prediction errors outside the measured direction.

18. **“its loss on the original task … 0.953 nats to 1.243”** — “Loss” could mean joint-token loss, marginal x loss, or another quantity. The uniform baseline of 1.099 suggests a three-symbol x loss, but that scope is not stated here.

19. **“matching the 1.245 we computed in advance for a network whose HMM1 readout is swapped everywhere”** — The aggregate loss supports this behavioral prediction but does not by itself identify a swapped HMM1 readout as the mechanism. The computation and its reference predictor are not described.

20. **“a confidently wrong network”** — Worse-than-uniform cross-entropy establishes harmful probability assignments on average. It does not, on its own, establish the stronger characterization that the network is generally confident and wrong.

21. **“Transplanting situation-B information into these networks’ activations”** — This is the first activation-transplant intervention, but its construction is not explained. Which activations, layers, positions, and components are replaced, how are donor sequences selected, and what makes the transplanted information specifically about B?

22. **“does nothing … there is simply nothing conditional to find”** — A null result from an unspecified intervention cannot rule out all conditional computation. The size of the measured effect and the intervention’s coverage are needed to assess this claim.

23. **“situation B’s own y-process quietly degrades as well (by 0.34 nats)”** — Relative to which baseline and at what checkpoint? Is this marginal y loss, and is 0.34 a seed average or a representative run?

24. **“Star: perfect self-inoculation”** — The caption does not explicitly define the star’s coordinates or whether “perfect” requires only flip (B, A) = (0, 1), or also Bayes-optimal losses. Those are not automatically equivalent.

25. **“in the shape we guessed” / “what the hypothesis predicts”** — The specific prior mechanistic prediction is not stated in the supplied setup. The later repository references do not make clear here exactly what was predicted in advance.

26. **“both losses back at the Bayes optimum (within 0.002 nats)”** — Which losses and which Bayesian predictor? In mixed finetuning, the optimal x prediction must account for uncertainty about the situation, whereas a predictor given the true situation has different information. That distinction matters particularly for the later calibration claim.

27. **“It is a switch on the shared tracker, not a second tracker”** — Equal sensitivity to erasing a subspace supports a shared dependence, but does not exclude a second tracker, duplicated information within that subspace, or additional situation-specific computation. The categorical exclusion exceeds the described evidence.

28. **“Erasing the HMM1 subspace still kills x prediction equally”** — Is this the pretrained subspace or a newly fitted post-finetuning subspace? The appendix says the tracker’s coordinates move, making that distinction material. “Kills” also leaves the resulting loss or behavior unspecified.

29. **“graft in only the finetuned weights of blocks 3 and 4”** — Does this replace both entire blocks together, including normalization parameters and biases? Likewise, were “blocks 1 and 2” tested individually or together? These details affect what the intervention localizes.

30. **“recover essentially the whole behaviour” / “does nothing at all”** — Only flip fractions are reported for the grafted networks. Do their task losses also recover? For “does nothing,” the relevant measurement and numerical tolerance are absent.

31. **“putting the pretrained blocks 3 and 4 back … removes the switch”** — Only the resulting A flip fraction is given. Without B behavior and task loss, the reader cannot tell whether this specifically removes conditional switching or more generally disrupts x prediction. The residual A flip of 0.12–0.20 also makes “removes” stronger than the reported result.

32. **“transplant the situation-B component … keeping the x half of the input identical”** — How is the situation component isolated and replaced without also transferring other y-related information? Does matching x mean matching the entire prefix? The intervention needs enough definition to distinguish changing situation information from broadly changing the activation.

33. **“the network’s x predictions turn unflipped (0.99 to 0.02)”** — What positions and sequences enter this statistic? A mean flip change does not show whether the intervention reliably changes each prediction, and no variation or control intervention is reported.

34. **“So the switch sits in block 3”** — The reported interventions establish sensitivity to information changed before block 3 and functional importance of blocks 3 and 4 together. They do not uniquely localize the whole switch to block 3. The appendix explicitly allows a branch spanning both blocks.

35. **“which is also where the situation posterior first becomes linearly readable”** — The earlier description says it becomes decodable “over the first three blocks,” without identifying a first-readable threshold. Is this statement about the pretrained or finetuned model, and what criterion determines “first”?

36. **“it flips the tracker’s output rather than the tracker”** — The main-text interventions do not distinguish those possibilities directly. The appendix provides coordinate-transfer evidence only for the simpler HMM1 and says that test is uninformative for the main version.

37. **“It flips by exactly how sure it is”** — The following errors are nonzero, so “exactly” overstates the numerical result. Agreement with an ideal observer’s posterior also does not independently establish the network’s own internal degree of certainty.

38. **“At each position … lies on top of the Bayesian posterior … (mean difference below 0.01)”** — A small mean difference does not establish positionwise agreement, especially if it is signed and errors cancel. Is this mean absolute error, a difference between positionwise averages, or something else? The referenced caption describes a per-sequence plot, not this positionwise result.

39. **“the Bayesian posterior that the sequence is from A”** — What observations and prior define this posterior? In particular, is it conditioned only on the prefix available for that prediction, or on the full sequence?

40. **“sequence by sequence … 60% sure … flipped by 60% (slope 0.99, mean error 0.04)”** — How is a single posterior assigned to a sequence when the posterior changes at each position? How is sequence-level flip aggregated, is “mean error” absolute, and what is the regression intercept? A slope near 1 alone does not establish calibration.

41. **“A model that is only partly sure it is being tested misbehaves only partly”** — The experiment distinguishes two generative situations; neither is an actual test, and swapping predictions is the specified training target in A. This sentence presents the testing-and-misbehavior analogy as an experimental finding.

42. **“both swap the shared readout for both situations at once”** — The early trajectories show behavioral transfer into B. No intervention at these early checkpoints is described that establishes the shared readout as the changing mechanism.

43. **“Gradient descent takes the shortcut first and builds the switch only once the shortcut has started to hurt”** — The time course supports a sequence of behavioral changes, but not this causal trigger. No gradient analysis or manipulation of the cost in B is shown.

44. **“before the switch existed” / “the switch was built in about fifty steps”** — Behavioral trajectories do not establish when a conditional mechanism first exists. A weak switch could already be developing during the initial leak, and checkpoints every 25 steps give limited temporal resolution.

45. **“by an amount set by how much the shortcut hurts there”** — This asserts a quantitative causal relationship without a reported experiment varying that cost. Expressing peak loss as a percentage of full-swap loss does not establish what sets the leak’s magnitude.

46. **“R² 0.83 to 0.94–0.97 … down to 0.41–0.53”** — Which block, positions, and evaluation distribution are used? Are readouts refitted after finetuning or transferred unchanged? Without this, improved decodability and improved transfer of a fixed readout cannot be distinguished.

47. **“The situation variable acquired a consumer in one case and lost one in the other”** — The claimed loss of a consumer is not demonstrated or identified. Earlier, the text says there was no connection to the x tracker before finetuning; meanwhile, situation information could already matter for y prediction. The causal explanation of the representation changes is therefore unsupported as stated.

48. **“training that teaches a model to behave differently in tests also sharpens its ability to tell that it is in one”** — Even with the conditional preface, increased linear decodability does not necessarily mean improved situation-inference accuracy. The experiment does not directly measure test recognition.

49. **“Mess3 … α = 0.7 and a leave-probability of 0.03 per alternative state”** — Mess3 and α are undefined for a reader assumed to know HMMs generally. The transition/emission rule or a reference is needed to reconstruct the processes. It is also unclear whether later “leave-probabilities” retain the per-alternative-state convention; “0.5 (B, never stays)” depends on that convention.

50. **“a last-token predictor recovers only half the learnable signal” / “94% of the learnable signal”** — “Learnable signal” has no metric or baseline. Is this the fraction of possible cross-entropy improvement over uniform guessing, and is the last-token predictor itself optimal?

51. **“so the tracker genuinely integrates history”** — The last-token comparison establishes that the task rewards using history. It does not alone establish that the learned tracker integrates it; that inference needs to be connected to the model’s measured performance or posterior tracking.

52. **“d = 120” / “5e-4”** — The roles of these quantities are unstated: presumably model width and learning rate. The distinction between hidden width, embedding width, and other dimensions is left implicit.

53. **“All losses are compared with the exact Bayesian predictor”** — This does not resolve whether losses are marginal or joint, whether the Bayesian predictor knows the actual situation, or which original/swapped objective applies at each comparison. Those choices determine the claimed optima.

54. **“Every qualitative finding is the same”** — The appendix lists several matching findings, but not evidence for every main-text result, such as the representation changes, y degradation, or erasure selectivity. The universal claim exceeds the comparisons visible here.

55. **“recovers 82 to 91% of the A-versus-B contrast”** — “Contrast” is undefined. Is it the difference in flip fractions, normalized to the fully finetuned model, or another quantity?

56. **“a further probe result … the belief probe fitted on the pretrained network”** — “Probe” first appears without explicitly identifying it with the earlier linear readout. Its target, training distribution, and whether separate probes are fitted at each block remain unspecified.

57. **“reads situation-A activations only after relabelling the coordinates by the swap”** — Which coordinates are relabelled: hidden-state posterior coordinates, probe outputs, or something else? The swap acts on observed x symbols, so its correspondence to a permutation of hidden-state coordinates requires explanation. “Only” also needs the unrelabeled comparison, which is not reported.

58. **“i.e. the switch is a situation-conditional relabelling of the tracker’s output coordinates”** — The transfer result is consistent with that interpretation, but does not uniquely establish it, particularly with R² as low as 0.57. The allowed alternative implementation in the final appendix paragraph further limits this claim.

59. **“the finetune moves the tracker’s coordinates in both situations”** — Failed transfer of a pretrained probe does not alone establish coordinate movement rather than changed information content or changed representational form. No additional comparison supporting that diagnosis is given.

60. **“the localisation there rests on the patching and weight-swapping results”** — “Patching” has not been defined or explicitly equated with the earlier activation transplants. “Localisation” also inherits the unresolved distinction between block 3 specifically and blocks 3 and 4 together.

61. **“at the moment the switch exists, the relative change in weights … (0.09 to 0.17)”** — What checkpoint or behavioral threshold defines that moment? What norm and normalization define relative weight change, and what does the range run over?

62. **“those changes are real but functionally irrelevant”** — Reverting embedding or unembedding weights with the rest of the finetuned network intact shows dispensability under those interventions and measured outputs. It does not establish functional irrelevance in general, including during training or through interactions with other changes.

63. **“the ‘incentive to stay itself’”** — This phrase presupposes an earlier framing beyond the setup supplied. Its operational meaning is partly supplied by “an explicit loss on situation B,” but what property of the model counts as “itself” remains unstated.

64. **“The design, predictions and their outcomes … are in `PREREG.md` and `RESULTS.md` in the repository”** — No repository link is supplied. A reader cannot inspect the advance predictions, computations, or additional evidence from these filenames alone.
tokens used
9,505
Based on the supplied text and figure captions, taking the earlier setup you described as known:

1. **“whose belief geometry is in the figure above”** — The version is identified by a figure whose caption is not included here. The text does not explain which feature of that geometry matters for interpreting these results.

2. **“three training seeds each”** — Does each seed cover pretraining and finetuning, or only finetuning? Are the two finetuning variants paired by pretrained checkpoint? Later numerical ranges are not identified as seed ranges, uncertainty intervals, or something else.

3. **“the *flip fraction*: how far the network’s x predictions have moved”** — The endpoints are defined, but the measurement is not. What formula maps a predicted distribution to a fraction? Is this a projection onto the difference between original and swapped distributions, a fitted mixture coefficient, or another statistic? Can it fall outside [0, 1], and how are positions with identical original and swapped predictions handled?

4. **“from the pretrained behaviour (0) towards the fully swapped one (1)”** — Are the reference distributions the actual pretrained network’s predictions or exact Bayesian predictions? “Fully swapped” could mean swapping the pretrained predictions or using the Bayes-optimal swapped predictor.

5. **“measured separately on situation-A and situation-B sequences”** — How are token-level measurements aggregated over positions and sequences? This matters when the text later moves between per-position, per-sequence, and whole-dataset flip fractions.

6. **“Simplex’s earlier work”** — The work is not identified or linked, and its particular prediction for this experiment is not stated.

7. **“read out linearly from the residual stream … R² 0.86 … 0.99”** — The readout procedure is unspecified: what is fitted, on which data, and evaluated on which held-out data? How is R² computed for a vector-valued posterior, and are these readouts trained jointly across situations or separately?

8. **“after the first block … after the last”** — “Block” first appears without identifying it as a transformer block or stating how many there are. The four-block architecture appears only in the appendix. The precise residual-stream measurement point within a block is also unspecified.

9. **“it is the same tracker in both situations” / “The directions that carry it are the same”** — What comparison establishes sameness: cross-situation readout transfer, subspace overlap, or something else? Linear decodability in both situations alone would not establish a shared implementation.

10. **“erase them (project out the subspace that the true posterior writes into)”** — How is this subspace constructed, and where is the projection applied? A posterior is a target quantity, so “writes into” does not identify an operational procedure. In particular, linear decoding directions and directions encoding a variable need not be the same.

11. **“x prediction collapses to chance by the same amount in both situations, +0.148 nats, i.e. exactly to uniform guessing”** — Is this an increase in marginal x cross-entropy, averaged per token? Matching uniform guessing’s average loss does not establish that the output distributions are uniform. “Exactly” also leaves the numerical tolerance and seed variation unclear.

12. **“while y prediction is untouched”** — Which measurement is unchanged, and within what tolerance? Unchanged average y loss is weaker than unchanged y predictions.

13. **“the posterior over A versus B becomes linearly decodable over the first three blocks”** — Is the target the ideal Bayesian posterior given the observed prefix, or the sequence’s actual situation label? Does “over the first three blocks” mean progressively improving decodability or first becoming decodable at block 3?

14. **“in a subspace nearly orthogonal … (overlap below 0.1)”** — “Overlap” has no definition or scale. The dimensions of the two subspaces and the method used to estimate them are also missing.

15. **“nothing connects them yet, since nothing in pretraining rewarded a connection”** — This overstates the evidence. Separately decodable, nearly orthogonal representations do not rule out interactions, and an absence of training reward for a connection does not establish its absence.

16. **“Training only in situation A gives emergent misalignment”** — The operational result is transfer of the swap into B. Given only the specified setup, the criterion for calling that “misalignment” has not been stated.

17. **“Within about a hundred steps its x predictions are fully swapped”** — Is “fully swapped” established by the full predicted distributions or only by an aggregate flip fraction near 1? A single scalar can conceal prediction errors outside the measured direction.

18. **“its loss on the original task … 0.953 nats to 1.243”** — “Loss” could mean joint-token loss, marginal x loss, or another quantity. The uniform baseline of 1.099 suggests a three-symbol x loss, but that scope is not stated here.

19. **“matching the 1.245 we computed in advance for a network whose HMM1 readout is swapped everywhere”** — The aggregate loss supports this behavioral prediction but does not by itself identify a swapped HMM1 readout as the mechanism. The computation and its reference predictor are not described.

20. **“a confidently wrong network”** — Worse-than-uniform cross-entropy establishes harmful probability assignments on average. It does not, on its own, establish the stronger characterization that the network is generally confident and wrong.

21. **“Transplanting situation-B information into these networks’ activations”** — This is the first activation-transplant intervention, but its construction is not explained. Which activations, layers, positions, and components are replaced, how are donor sequences selected, and what makes the transplanted information specifically about B?

22. **“does nothing … there is simply nothing conditional to find”** — A null result from an unspecified intervention cannot rule out all conditional computation. The size of the measured effect and the intervention’s coverage are needed to assess this claim.

23. **“situation B’s own y-process quietly degrades as well (by 0.34 nats)”** — Relative to which baseline and at what checkpoint? Is this marginal y loss, and is 0.34 a seed average or a representative run?

24. **“Star: perfect self-inoculation”** — The caption does not explicitly define the star’s coordinates or whether “perfect” requires only flip (B, A) = (0, 1), or also Bayes-optimal losses. Those are not automatically equivalent.

25. **“in the shape we guessed” / “what the hypothesis predicts”** — The specific prior mechanistic prediction is not stated in the supplied setup. The later repository references do not make clear here exactly what was predicted in advance.

26. **“both losses back at the Bayes optimum (within 0.002 nats)”** — Which losses and which Bayesian predictor? In mixed finetuning, the optimal x prediction must account for uncertainty about the situation, whereas a predictor given the true situation has different information. That distinction matters particularly for the later calibration claim.

27. **“It is a switch on the shared tracker, not a second tracker”** — Equal sensitivity to erasing a subspace supports a shared dependence, but does not exclude a second tracker, duplicated information within that subspace, or additional situation-specific computation. The categorical exclusion exceeds the described evidence.

28. **“Erasing the HMM1 subspace still kills x prediction equally”** — Is this the pretrained subspace or a newly fitted post-finetuning subspace? The appendix says the tracker’s coordinates move, making that distinction material. “Kills” also leaves the resulting loss or behavior unspecified.

29. **“graft in only the finetuned weights of blocks 3 and 4”** — Does this replace both entire blocks together, including normalization parameters and biases? Likewise, were “blocks 1 and 2” tested individually or together? These details affect what the intervention localizes.

30. **“recover essentially the whole behaviour” / “does nothing at all”** — Only flip fractions are reported for the grafted networks. Do their task losses also recover? For “does nothing,” the relevant measurement and numerical tolerance are absent.

31. **“putting the pretrained blocks 3 and 4 back … removes the switch”** — Only the resulting A flip fraction is given. Without B behavior and task loss, the reader cannot tell whether this specifically removes conditional switching or more generally disrupts x prediction. The residual A flip of 0.12–0.20 also makes “removes” stronger than the reported result.

32. **“transplant the situation-B component … keeping the x half of the input identical”** — How is the situation component isolated and replaced without also transferring other y-related information? Does matching x mean matching the entire prefix? The intervention needs enough definition to distinguish changing situation information from broadly changing the activation.

33. **“the network’s x predictions turn unflipped (0.99 to 0.02)”** — What positions and sequences enter this statistic? A mean flip change does not show whether the intervention reliably changes each prediction, and no variation or control intervention is reported.

34. **“So the switch sits in block 3”** — The reported interventions establish sensitivity to information changed before block 3 and functional importance of blocks 3 and 4 together. They do not uniquely localize the whole switch to block 3. The appendix explicitly allows a branch spanning both blocks.

35. **“which is also where the situation posterior first becomes linearly readable”** — The earlier description says it becomes decodable “over the first three blocks,” without identifying a first-readable threshold. Is this statement about the pretrained or finetuned model, and what criterion determines “first”?

36. **“it flips the tracker’s output rather than the tracker”** — The main-text interventions do not distinguish those possibilities directly. The appendix provides coordinate-transfer evidence only for the simpler HMM1 and says that test is uninformative for the main version.

37. **“It flips by exactly how sure it is”** — The following errors are nonzero, so “exactly” overstates the numerical result. Agreement with an ideal observer’s posterior also does not independently establish the network’s own internal degree of certainty.

38. **“At each position … lies on top of the Bayesian posterior … (mean difference below 0.01)”** — A small mean difference does not establish positionwise agreement, especially if it is signed and errors cancel. Is this mean absolute error, a difference between positionwise averages, or something else? The referenced caption describes a per-sequence plot, not this positionwise result.

39. **“the Bayesian posterior that the sequence is from A”** — What observations and prior define this posterior? In particular, is it conditioned only on the prefix available for that prediction, or on the full sequence?

40. **“sequence by sequence … 60% sure … flipped by 60% (slope 0.99, mean error 0.04)”** — How is a single posterior assigned to a sequence when the posterior changes at each position? How is sequence-level flip aggregated, is “mean error” absolute, and what is the regression intercept? A slope near 1 alone does not establish calibration.

41. **“A model that is only partly sure it is being tested misbehaves only partly”** — The experiment distinguishes two generative situations; neither is an actual test, and swapping predictions is the specified training target in A. This sentence presents the testing-and-misbehavior analogy as an experimental finding.

42. **“both swap the shared readout for both situations at once”** — The early trajectories show behavioral transfer into B. No intervention at these early checkpoints is described that establishes the shared readout as the changing mechanism.

43. **“Gradient descent takes the shortcut first and builds the switch only once the shortcut has started to hurt”** — The time course supports a sequence of behavioral changes, but not this causal trigger. No gradient analysis or manipulation of the cost in B is shown.

44. **“before the switch existed” / “the switch was built in about fifty steps”** — Behavioral trajectories do not establish when a conditional mechanism first exists. A weak switch could already be developing during the initial leak, and checkpoints every 25 steps give limited temporal resolution.

45. **“by an amount set by how much the shortcut hurts there”** — This asserts a quantitative causal relationship without a reported experiment varying that cost. Expressing peak loss as a percentage of full-swap loss does not establish what sets the leak’s magnitude.

46. **“R² 0.83 to 0.94–0.97 … down to 0.41–0.53”** — Which block, positions, and evaluation distribution are used? Are readouts refitted after finetuning or transferred unchanged? Without this, improved decodability and improved transfer of a fixed readout cannot be distinguished.

47. **“The situation variable acquired a consumer in one case and lost one in the other”** — The claimed loss of a consumer is not demonstrated or identified. Earlier, the text says there was no connection to the x tracker before finetuning; meanwhile, situation information could already matter for y prediction. The causal explanation of the representation changes is therefore unsupported as stated.

48. **“training that teaches a model to behave differently in tests also sharpens its ability to tell that it is in one”** — Even with the conditional preface, increased linear decodability does not necessarily mean improved situation-inference accuracy. The experiment does not directly measure test recognition.

49. **“Mess3 … α = 0.7 and a leave-probability of 0.03 per alternative state”** — Mess3 and α are undefined for a reader assumed to know HMMs generally. The transition/emission rule or a reference is needed to reconstruct the processes. It is also unclear whether later “leave-probabilities” retain the per-alternative-state convention; “0.5 (B, never stays)” depends on that convention.

50. **“a last-token predictor recovers only half the learnable signal” / “94% of the learnable signal”** — “Learnable signal” has no metric or baseline. Is this the fraction of possible cross-entropy improvement over uniform guessing, and is the last-token predictor itself optimal?

51. **“so the tracker genuinely integrates history”** — The last-token comparison establishes that the task rewards using history. It does not alone establish that the learned tracker integrates it; that inference needs to be connected to the model’s measured performance or posterior tracking.

52. **“d = 120” / “5e-4”** — The roles of these quantities are unstated: presumably model width and learning rate. The distinction between hidden width, embedding width, and other dimensions is left implicit.

53. **“All losses are compared with the exact Bayesian predictor”** — This does not resolve whether losses are marginal or joint, whether the Bayesian predictor knows the actual situation, or which original/swapped objective applies at each comparison. Those choices determine the claimed optima.

54. **“Every qualitative finding is the same”** — The appendix lists several matching findings, but not evidence for every main-text result, such as the representation changes, y degradation, or erasure selectivity. The universal claim exceeds the comparisons visible here.

55. **“recovers 82 to 91% of the A-versus-B contrast”** — “Contrast” is undefined. Is it the difference in flip fractions, normalized to the fully finetuned model, or another quantity?

56. **“a further probe result … the belief probe fitted on the pretrained network”** — “Probe” first appears without explicitly identifying it with the earlier linear readout. Its target, training distribution, and whether separate probes are fitted at each block remain unspecified.

57. **“reads situation-A activations only after relabelling the coordinates by the swap”** — Which coordinates are relabelled: hidden-state posterior coordinates, probe outputs, or something else? The swap acts on observed x symbols, so its correspondence to a permutation of hidden-state coordinates requires explanation. “Only” also needs the unrelabeled comparison, which is not reported.

58. **“i.e. the switch is a situation-conditional relabelling of the tracker’s output coordinates”** — The transfer result is consistent with that interpretation, but does not uniquely establish it, particularly with R² as low as 0.57. The allowed alternative implementation in the final appendix paragraph further limits this claim.

59. **“the finetune moves the tracker’s coordinates in both situations”** — Failed transfer of a pretrained probe does not alone establish coordinate movement rather than changed information content or changed representational form. No additional comparison supporting that diagnosis is given.

60. **“the localisation there rests on the patching and weight-swapping results”** — “Patching” has not been defined or explicitly equated with the earlier activation transplants. “Localisation” also inherits the unresolved distinction between block 3 specifically and blocks 3 and 4 together.

61. **“at the moment the switch exists, the relative change in weights … (0.09 to 0.17)”** — What checkpoint or behavioral threshold defines that moment? What norm and normalization define relative weight change, and what does the range run over?

62. **“those changes are real but functionally irrelevant”** — Reverting embedding or unembedding weights with the rest of the finetuned network intact shows dispensability under those interventions and measured outputs. It does not establish functional irrelevance in general, including during training or through interactions with other changes.

63. **“the ‘incentive to stay itself’”** — This phrase presupposes an earlier framing beyond the setup supplied. Its operational meaning is partly supplied by “an explicit loss on situation B,” but what property of the model counts as “itself” remains unstated.

64. **“The design, predictions and their outcomes … are in `PREREG.md` and `RESULTS.md` in the repository”** — No repository link is supplied. A reader cannot inspect the advance predictions, computations, or additional evidence from these filenames alone.
