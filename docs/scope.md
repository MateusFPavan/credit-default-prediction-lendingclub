# Project Scope — Credit Default Prediction (Lending Club)

Canonical document. Defines the problem, the analytical population, the success metric,
and what was deliberately ruled out. Written before modeling; changes require a recorded
reason.

## 1. Business question

Given a personal loan application, using only information available at the moment of the
credit decision, will the loan be repaid or result in a realized loss?

The model simulates a second decision layer over applicants who already passed the
platform's approval filter. It is not a replacement for that filter: it operates on the
approved population, which is the only one with a known outcome.

## 2. Context

Lending Club was the largest peer-to-peer lending platform in the United States,
operating from 2007 to 2020. Individual investors funded fractions of personal loans and
bore credit risk directly. In that design, predicting default is a capital allocation
decision: every approved loan is a bet with a known return (contracted interest) and a
known loss (unrecovered principal).

## 3. Target and analytical population

Binary target:
- 1 = Charged Off (realized loss; the institution stopped pursuing collection)
- 0 = Fully Paid (loan repaid in full)

Population rule: only loans with a concluded outcome. All other statuses (Current, Late,
In Grace Period, Default, Does not meet the credit policy) are excluded — labeling an
in-progress loan would record a prediction as a fact.

Maturity cutoff: 36-month loans issued through Dec/2015; 60-month loans through
Dec/2013. The cutoff is arithmetic: last date in the data (Dec/2018) minus the
contractual term.

Rationale — maturity bias. A loan issued in 2018 can only be concluded in two ways:
early payoff or fast default. The regular payer is still Current and is removed by the
population rule. Keeping immature vintages would mean training on a sample composed
exclusively of early outcomes, whose default rate does not represent the vintage. In a
mature vintage the problem does not exist: every loan has reached term, every loan has an
outcome, and the filter excludes no one.

Exclusion funnel (fully reconciled):

| Reason | Rows | % |
|---|---|---|
| Total in file | 2,260,701 | 100.00% |
| Status Current | 878,317 | 38.85% |
| Status Late (31-120 days) | 21,467 | 0.95% |
| Status In Grace Period | 8,436 | 0.37% |
| Status Late (16-30 days) | 4,349 | 0.19% |
| Does not meet the credit policy (both) | 2,749 | 0.12% |
| Status Default | 40 | 0.002% |
| CSV footer rows (no loan_status) | 33 | 0.0015% |
| Immature vintage (36m after Dec/2015) | 402,159 | 17.79% |
| Immature vintage (60m after Dec/2013) | 269,598 | 11.93% |
| Analytical population | 673,553 | 29.79% |

Default rate in the population: 14.81%.

## 4. Success metric

Accuracy is inadequate and will not be used as the primary metric. A model that approves
every application is 85.19% accurate and prevents zero losses. When the class of interest
is the minority, accuracy measures the prevalence of the majority class, not the model's
capability.

The decision metric is expected portfolio profit. High recall is not an objective in
itself: catching many low-value defaults while approving the high-value ones is a bad
outcome disguised as a good metric. What matters is the aggregate financial result of
the decision:

profit = sum(approved good loans x contracted interest) - sum(approved bad loans x lost principal)

Both terms are computable from the data itself (installment x term - loan_amnt for the
return; loan_amnt - total_rec_prncp for the realized loss). The classifier's cutoff will
be chosen at the threshold that maximizes this curve — not by convention (0.5), not by
F1 optimization.

Diagnostic metrics (reported, not optimized): confusion matrix, precision, recall, F1,
ROC-AUC, PR-AUC. They explain the model's behavior; they do not decide the cutoff.

Cost asymmetry: approving someone who does not pay (False Negative, under the convention
positive = default) costs the principal. Rejecting a good borrower (False Positive) costs
the interest that would have been earned. The two costs are unequal and depend on loan
size — hence weighting by value rather than counting correct predictions.

## 5. Methodological commitments

- No post-origination column is used as a feature. Each of the 151 columns was
  classified from the official data dictionary definition, not from name heuristics, and
  verified empirically via univariate AUC.
- Temporal split, never random: train on the past, validate and test on the future. A
  shuffled split would let 2015 loans inform predictions about 2011 loans.
- The test set is touched once, at the end. Model selection, tuning, and threshold choice
  happen on a separate validation set.
- Baseline before complexity. Logistic Regression sets the ruler; tree models must
  justify their additional complexity.

## 6. Discarded approaches

| Discarded | Reason |
|---|---|
| ULB credit card fraud dataset | 28 PCA-anonymized features. Rules out interpretable feature engineering and any defense of variable choice in an interview. |
| Sparkov simulated fraud dataset | Simulated data contains only the patterns that were programmed into it. This limits not just generalization but the ceiling of discovery: no one can find a pattern that nobody planted. |
| IEEE-CIS fraud dataset | Real, but most columns are masked (V1-V339) with no public dictionary. Same limitation as ULB. |
| Home Credit (multi-table credit) | Seven relational tables; scope-overrun risk disproportionate to the gain. Heavily solved public competition. |
| Brazilian federal highway crash data (PRF) | Real and underused, but repeats Project 1's sector (transport/government), and error cost is not monetizable without entering moral territory. |
| Reject inference | Technique for inferring outcomes of rejected applicants. Requires strong, unverifiable assumptions; acknowledged as fragile. Out of scope, cited as an extension. |
| Unsupervised anomaly detection | A reliable outcome label exists. No reason to discard it. |
| Group split (GroupKFold) | Technically correct to prevent the same borrower from appearing in both train and test. Inapplicable: member_id is 100% null across the entire dataset history. |
| Model-based imputation of pre-2012 bureau attributes | No comparable complete cases exist in that period: the information was never collected from anyone before 2012. Imputing would extrapolate the 2013-2018 pattern backwards, generating synthetic data. |
| Categorical "economic era" feature | Encodes period, not risk. It does not generalize to future vintages — the opposite of the model's purpose. Eras enter as validation structure (walk-forward), not as a variable. |

## 7. Known limitations (declared before any result)

Selection bias. The dataset contains only loans Lending Club approved. Every learned
pattern is conditional on the platform's approval filter. The model estimates
P(default | approved), not P(default | applicant). A profile the platform rarely approved
appears in the data only through its exceptional cases, which can invert the risk reading
for that group.

Train/test independence is not verifiable at the individual level. The borrower
identifier (member_id) was removed during anonymization — it never held a single non-null
value between 2007 and 2018. If the same individual took two loans, one may sit in train
and the other in test. The expected effect is small (the temporal split tends to keep a
borrower's nearby loans on the same side of the cut, and the same borrower may have
different outcomes), but it is not measurable. The technique that would solve it requires
the identifier that does not exist.

Raw data is not primary. The public CSV underwent minimal third-party transformation:
the % symbol was stripped from int_rate and revol_util and both were cast to float. The
data dictionary in use is a third-party copy — the original source went offline when the
platform shut down.

No label exists for rejected applicants. See reject inference, section 6.

## 8. Success criteria

The project succeeds if:
1. The final model beats an approve-everyone policy on expected profit.
2. Every methodological choice — target, population, metric, split, features — is
   defensible in one sentence, without exception.
3. The limitations above are written down before anyone asks.

The project is not evaluated by high accuracy, nor by high AUC in isolation.

## 9. Term holdout

term (36 or 60 months) is held out rather than pooled or used as a feature. The two terms
default at different base rates — 13.89% for 36-month loans versus 25.16% for 60-month
loans, nearly double — evidence that they are underwritten as distinct risk pools, not two
points on the same scale. Pooling them would blur that difference into a single average
that describes neither. The 36-month loans (618,345 of 673,314 approved loans, 91.8%)
become the primary population, split by issue_d into train (through 2013), validation
(2014) and test (2015). The 60-month loans (54,969) become a held-out transfer set,
touched only after a model is selected on the primary population, to test whether what was
learned on one term generalizes to the other. term remains in the dataset as EVAL_ONLY —
the profit calculation needs it — but is never a training feature, since within each
population it no longer varies.
