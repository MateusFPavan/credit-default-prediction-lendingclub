# Next steps

What this project would do next, in order, and three options that were considered and set
aside. Each item says what has been measured, why it is not done yet, how it would be
done, how success would be judged, and what would move it up the list.

The scope these items assume is the one in the [model card](MODEL_CARD.md) §3-4: a second
decision layer over an already-approved book of 36-month loans, built for methodology
review.

## In order

### 1. Report the headline as a distribution across seeds

- **Measured.** The headline already holds against refit noise. Eight refits of the frozen
  configuration, changing only the seed, spread test profit with a standard deviation of
  $0.76M. The XGB beats the logistic baseline by $5.58M at the eight-seed mean (7.4 SD)
  and by $4.55M at the worst seed ([technical report](technical_report.md) §7.6).
- **Not done yet because** the published figure is one seed's draw, asserted to the cent
  by `run_all.py` and cited across the documents. Replacing it is a cascade of edits for a
  conclusion that does not change.
- **How.** Publish the eight-seed mean and spread next to the single-seed figure, or
  average the seeds into one model, which also lowers the variance of the served score.
  The single-seed figure stays as the reproducibility check.
- **Done when** every comparison between two fits in the repository is judged against the
  seed spread rather than against one draw.
- **Moves up if** a change is expected to move profit by less than about two seed SDs
  ($1.5M). Below that, a single-seed comparison cannot tell the change from the draw, and
  item 2 is that case.

### 2. Close the gap with the baseline for lower-income borrowers

- **Measured.** The XGB beats the logistic baseline on test profit (+$6.3M) and is ahead
  of it within every grade, clearly so in C and D. By income it is behind in the two lower
  quartiles, in all eight refits: by 0.006 in the lowest (AUC 0.648 against 0.654), which
  survives a correction for multiple comparisons, and by 0.004 in the second, where the
  corrected interval just reaches zero ([technical report](technical_report.md) §8).
- **Not done yet because** the difference is small, and a retrain changes the published
  headline and every document that cites it. The decision was to state the gap rather than
  retrain for it.
- **How.** Two candidates, tuned on the validation year and never on the test set:
  up-weighting lower-income borrowers in training, and blending the XGB score with the
  logistic score. Keep whichever closes the gap.
- **Done when** the two lower quartiles are at or above the baseline in every refit, and
  total profit stays within the seed spread of item 1.
- **Moves up if** the intended use concentrates lower-income borrowers, a fairness review
  asks for parity with the baseline by income, or the model is retrained for any other
  reason, in which case this goes into the same retrain at no extra cost.

### 3. A dedicated scorecard for 60-month loans

- **Measured.** Applied without refitting, the 36-month model degrades on 60-month loans
  (AUC 0.6846 to 0.6433), and the logistic baseline's profit gain over approve-all turns
  negative. The two terms are distinct risk populations ([technical
  report](technical_report.md) §9), and the API accepts `term=36` only.
- **Not done yet because** it is a second model rather than a fix to this one. The 54,969
  matured 60-month loans are held out untouched for it.
- **How.** The same pipeline on the 60-month population: temporal split, walk-forward
  tuning, and selection by profit against approve-all and a logistic baseline.
- **Done when** it beats both baselines on held-out profit, the bar the 36-month model
  passed.
- **Moves up if** the book to be scored includes 60-month loans.

### 4. Run the monitoring loop on real batches

- **Measured.** The model is served behind an API, containerized and tested in CI on every
  push; the PSI drift monitor separates real drift from known artifacts; and the
  retraining trigger is a written policy ([model card](MODEL_CARD.md) §10).
- **Not done yet because** there is no production feed. The PSI figures come from test
  vintages, and a scheduler with nothing real to read would only be tested against itself.
- **How.** Run `src/monitor.py` on each incoming batch, alert on the trigger conditions,
  and gate any refit behind the same profit test against both baselines, judged against
  the seed spread of item 1.
- **Done when** one real batch has gone through the monitor, the verdict and, if
  triggered, a gated refit.
- **Moves up if** the model scores real applications.

## Considered and set aside

### Reject inference

Investigated in full ([reject-inference roadmap](reject_inference_roadmap.md), notebooks
16 to 20). On this dataset it cannot be validated: no rejected application has an outcome,
the signal shared between approved and rejected applicants is weak (AUC 0.5620), and a
Bayesian bias-aware evaluation returns close to the prior it is given (slope about 0.99).

**Reopen if** a data source provides outcomes for some rejected applicants.

### Recalibrating the probabilities

Tested by two routes and rejected ([technical report](technical_report.md) §10).
Retraining with a held-out calibration year costs 42% of the training data and brings
validation profit close to approve-all; transforming the score instead improves Brier by
0.6% and moves profit at the threshold by 0.04%. The cause is a base-rate shift that
follows the credit cycle, so there is no stable target to calibrate to. The measured
offset is documented for anyone who needs an absolute probability of default.

**Reopen if** a use consumes the raw probability, such as pricing or loss provisioning,
rather than a fixed threshold.

### A separate approval threshold for lower-income borrowers

The gap in item 2 is in ranking, and AUC does not depend on the threshold: a different
cutoff for lower incomes changes how many of those borrowers are approved, not how well
they are ranked. It would also add a per-segment rule that every credit decision has to
justify.

**Reopen if** the goal becomes equal approval rates across income groups, which is a
policy decision rather than a modeling one.
