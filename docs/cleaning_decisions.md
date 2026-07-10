# Cleaning Decisions — Credit Default Prediction (Lending Club)

Canonical document. Every transformation applied to the raw data, with the mechanism that
justifies it. Written before the cleaning pipeline runs. This is the contract: if the
implementation diverges from this document, flag the divergence and ask — do not rewrite
this file.

Population: 673,553 loans (see scope.md). All decisions below apply within it.

## Column classification

The 151 columns were classified by a single criterion, applied against the official data
dictionary definition:

"Does this information exist and is it known at the moment the loan is approved, or is it
born/updated afterwards?"

| Family | Count | Fate |
|---|---|---|
| A — Identifier | 3 | Dropped |
| B — Post-origination | 37 | Dropped (leakage) |
| C — Available at origination | 99 | Feature candidates |
| D — Free text | 4 | Not features |
| E — Target and evaluation metadata | 5 | Never used for training |

Empirical verification: univariate AUC was computed for all numeric columns against the
target. Family B dominates the top of the ranking — recoveries (0.90),
collection_recovery_fee (0.86), last_fico_range_low (0.91 after inverting direction).
A column that alone nearly knows the outcome was written after the outcome. This is the
leakage test working, not a problem.

Hard rule: family E columns (loan_status, loan_amnt, installment, term, total_rec_prncp)
are used only to compute the financial result of each loan. They never enter any feature
set, in any phase.

### REVISAR columns — resolved by evidence

- funded_amnt → family C. Equals loan_amnt in 99.71% of rows; divergence is concentrated
  in 2007-2011 vintages, when the platform sometimes reduced the funded amount at
  approval. Redundant with loan_amnt.
- funded_amnt_inv → family B. The ratio funded_amnt_inv / funded_amnt rises from 0.235
  (2007) to 0.9996 (2015): in the early years, investor funding accumulated gradually
  after listing. The value completed itself after origination.
- verification_status_joint → family C. Its 239 non-null rows are exactly the 239 rows
  where application_type = joint app; the column's birth month (Oct/2015) coincides with
  annual_inc_joint and dti_joint. The dictionary entry verified_status_joint is a typo.

All three have univariate AUC near 0.50. Correct classification changed nothing
practical here — which is the point: rigor is not about the case where it matters, it is
about not knowing in advance which case is which.

## Missing data — mechanism before treatment

### Structural absence: columns that did not exist yet

Reconstructed column birth log (docs/column_birth_log.csv) confirms three rollouts:

- Mar/2012 and Aug/2012: bureau attribute blocks (tot_cur_bal, tot_coll_amt,
  total_rev_hi_lim, avg_cur_bal, num_*, mo_sin_*, mort_acc, bc_util, and others).
  100% null through 2011, ~52% null in 2012, 0% from 2013 onward.
- Dec/2015: open_acc_6m block (13 columns). 100% null through 2014. The analytical
  population ends in Dec/2015 — these columns are ~97.8% empty within it.
- Mar/2017: sec_app_* block (11 columns), and revol_bal_joint. Born after the population
  window; 100% null within it.

Decisions:
- Drop the Dec/2015 block (13 columns) and every 100%-null column (member_id,
  next_pymnt_d, sec_app_*, revol_bal_joint). Justification is not "too many nulls" — it
  is "the field was not collected during the analytical period", verifiable in the birth
  log.
- Keep the 2012 bureau blocks. Fill nulls with an out-of-domain sentinel and add a binary
  flag era_pre_2012. Rationale: model-based imputation would require comparable complete
  cases from that period, and none exist. The flag declares ignorance rather than
  concealing it. Trade-off acknowledged: the flag correlates with time, so its feature
  importance must be checked and reported.

### Informative absence (MNAR): the null IS the information

Columns mths_since_last_delinq (51.8% null), mths_since_last_record (84.6%),
mths_since_recent_bc_dlq (77.1%), mths_since_recent_revol_delinq (67.8%),
mths_since_last_major_derog (75.4%).

Evidence for the mechanism: null share is stable across every vintage (it does not drop
to zero in any year, unlike the rollout blocks), and the default rate among nulls is
consistently lower than among filled rows (-0.57 to -2.32 percentage points). The null
means "this borrower never had the event", not "the field was not collected".

Decision: flag plus sentinel. For each column, create a binary flag (had_event = 0/1) and
fill the null with 999 — a value outside the real domain, preserving the natural ordering
(higher = longer since the event = lower risk; "never" is the extreme of that direction).

Imputing the median would assert that half the population defaulted 30 months ago —
fabricating a delinquency history for the cleanest borrowers. Filling with zero would
invert the signal entirely.

The flag exists because linear models cannot read a sentinel: to Logistic Regression, 999
is a large number that corrupts the coefficient. Trees use the sentinel, linear models use
the flag, and no model is silently excluded from the project by a cleaning decision.

## Outliers

### Not errors: high-income borrowers

30 rows are extreme (above the 99.99th percentile) in two or more of: bc_open_to_buy,
revol_util, total_il_high_credit_limit, tot_cur_bal, tot_coll_amt, total_rev_hi_lim, dti,
annual_inc.

They are Managing Directors, CEOs, CFOs, Presidents and Partners. Mortgage holders,
annual income between $400K and $5M, grades A/B/C, and 27 of 30 repaid in full.

The columns explode together because they describe the same reality: high income drives
high credit limits, high balances, and high available credit. This is natural
multicollinearity, not corruption.

Decision: keep them. Removing these rows would remove the high-income segment from the
dataset.

The single row extreme in four columns (id 21200242) was audited against its 29 peers and
against the population median across eight internal-consistency ratios. No ratio diverged
from its peers by more than 2x; none is mathematically impossible; tot_coll_amt is zero,
ruling out the suspected contradiction (a millionaire in collections). Verdict: a genuine
borrower.

### Errors: dti

Five rows in the population have dti > 100 (values of 999.00, 672.52, 137.40, 120.66,
104.00), all with reported annual income between $1,200 and $20,000. A debt-to-income
ratio of 999 means monthly debt payments equal to 999 times monthly income — internally
impossible. Four of the five also have null emp_length. All five repaid the loan, which
further contradicts the reported ratio.

Decision: drop these 5 rows. The population of 673,553 is unaffected; the values cannot
describe reality.

Two rows with dti < 0 exist in the file but not in the population (excluded by status or
vintage). Recorded, no action needed.

## Data quality note: revol_util disagrees with itself

In the general population, revol_util (reported, 55.00% median) matches the recomputed
ratio revol_bal / total_rev_hi_lim (55.31% median). Among the 30 high-income outliers the
two diverge sharply: 44.70% reported versus 86.59% recomputed.

The two figures do not describe the same set of accounts. The divergence is invisible at
the median and systematic in the tail.

Consequence: any engineered feature of the form utilization = revol_bal /
total_rev_hi_lim is not equivalent to revol_util, and the two will disagree precisely in
the high-income segment. This is recorded as a constraint on feature engineering, not as
an error to fix.

## Mechanical transformations

- Date columns cast to real datetime (11 columns).
- term parsed to integer months; emp_length parsed to integer years ("< 1 year" -> 0,
  "10+ years" -> 10).
- Low-cardinality categoricals stripped and lowercased. Cardinality was unchanged for all
  eight — the dataset was already clean here (these fields come from form dropdowns, not
  free typing).
- Zero-variance columns dropped (cardinality = 1 within the population): policy_code,
  disbursement_method, pymnt_plan, hardship_flag, out_prncp, out_prncp_inv.
- Free-text columns (emp_title, title, desc, zip_code) are not normalized and do not
  become features. emp_title is retained solely as a lookup field for outlier auditing —
  it is what identified the high-income borrowers above.
- 33 footer rows from the source CSV (text in the id field, all other columns null) are
  excluded by the population rule; no separate action needed.
- Duplicates: zero by id. A weak proxy (loan_amnt + issue_d + annual_inc + addr_state +
  purpose + emp_title) flags 253 extra rows, but with member_id removed there is no
  borrower key: same-person detection is not possible in this dataset. No rows removed;
  the impossibility is declared in scope.md.

## Deferred to modeling

- int_rate, grade and sub_grade encode Lending Club's own risk assessment. Using them
  means predicting how well the platform predicted, not predicting default. Provisional
  decision: exclude them, and train a comparison model that includes them. If the model
  without them beats the baseline with them, that is the project's finding. If it loses,
  the value of the platform's judgment has been quantified. Either outcome is publishable.
- Rare category consolidation (categories below a volume threshold collapsed into
  "other"): the cutoff will be set from value counts, not assumed.
- Sensitivity check on 2007-2009 vintages (kept per D3, on the hypothesis that early
  adopters of a new platform carry their own risk profile — recorded as a hypothesis to
  test, not a fact).

## Corrections and additions (post categorical evidence)

### verification_status is inverted — and it is not a definition error

Verified income correlates with HIGHER default (17.75%) than not-verified income (11.74%).
This is selection, not noise. Lending Club did not verify everyone's income: it verified
when something in the application warranted a check. Verification is a consequence of a
risk signal an analyst saw, not a cause of safety. The column does not measure "income is
trustworthy" — it measures "the platform found it necessary to check".

This corrects an earlier working assumption in this project: verification_status was
proposed as the natural separator between real and misreported income among the
high-income outliers. It is not. The decision to keep those rows stands, but on the
internal-consistency audit (see Outliers section), not on their verification status.

Consequence: the column is a legitimate and strong feature whose meaning is the opposite
of the intuitive reading. This must be stated explicitly wherever it is interpreted.

### emp_length null is MNAR, not missing data

Null emp_length (5.58% of the population) has a 20.84% default rate [20.43-21.25],
against 13.71%-15.20% for every filled category. The absence carries risk signal —
plausibly unemployed, self-employed without documentation, or refusal to disclose. It is
not random.

Corroborating evidence: four of the five rows with impossible dti (>100) also have null
emp_length.

Decision: same treatment as the mths_since_* family — binary flag emp_length_missing plus
an out-of-domain sentinel (-1). Median imputation would erase the single strongest signal
in the column.

### Rare category consolidation

Criterion: a category survives if it can support its own risk estimate. The test is the
width of the 95% binomial confidence interval around its default rate — when the error bar
is wider than the difference worth detecting, the category asserts nothing on its own.

- home_ownership: `any` (2 rows), `none` (~0.01%), `other` (~0.02%) collapsed into
  `other`. CI widths of 12.75pp, 21.18pp, and degenerate respectively. The three mean the
  same thing in practice: not mortgage, not rent, not owned outright.
- purpose: `wedding`, `educational`, `renewable_energy` collapsed into `other`. These
  categories were discontinued by the platform, so they exist only in older vintages —
  keeping them would turn the category into a proxy for issue year, which is harmful under
  temporal validation. `house` is kept: at ~0.5% its CI (~1.5pp) still supports an
  estimate, and the category remains active.
- application_type: `joint app` (239 rows, 0.04%) kept as-is. It is binary, and the rare
  level carries structural meaning (the *_joint columns exist only there). Its CI (9.74pp)
  means the model will likely ignore it; removing rows for being rare requires a better
  reason.
- addr_state: dropped entirely. With 51 levels — a third of them without enough sample to
  sustain an estimate — it would add 50 dummies to the linear model in exchange for signal
  that income, bureau attributes, and credit utilization already capture. Grouping into 4
  census regions was considered and rejected: the grouping is arbitrary and would have to
  be defended against alternatives (Fed districts, cost-of-living bands) without evidence
  that geography adds anything. Recorded as a next step: test whether region-level
  geography adds signal.
- initial_list_status, emp_length, verification_status: no rare categories. Untouched.

### hardship_flag: cardinality 1 for a specific reason

hardship_flag = 'N' for 100% of the population, while 1,386 rows have hardship detail
fields populated. The flag reflects current status; the detail fields preserve history.
Since the population contains only terminated loans, nobody is in hardship now — though
some were. The flag is dropped for zero variance; the hardship detail columns are dropped
as post-origination leakage. Both decisions stand, for different reasons than they
appeared to.
