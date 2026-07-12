# Datasheet for Dataset: Lending Club Accepted Loans (2007–2018)

Following Gebru et al., *Datasheets for Datasets*. Describes the dataset as used in the
`credit-default-prediction-lendingclub` project. Quantitative claims are sourced from
`docs/FACTS.md`, the project's verified fact sheet; license and maintenance details not
covered there are stated below as supplementary facts.

## 1. Motivation

Lending Club, a peer-to-peer lending platform operating 2007–2020, originally published
this data as a record of loans it originated and their outcomes. It was republished on
Kaggle (`wordsforthewise/lending-club`) after Lending Club's data portal went offline
following the company's exit from P2P lending. This project uses it to build and evaluate
a credit-default prediction model as a portfolio/educational exercise, not on behalf of
Lending Club or any funder — repackaging and documentation were done independently by the
project author.

## 2. Composition

**Raw file**: `accepted_2007_to_2018Q4.csv`, ~1.67 GB, 2,260,701 loan records, 151 raw
columns. Each row is one loan application accepted and funded by Lending Club — no larger
sampled-from population exists beyond Lending Club's own historical accepted-loan book.

**Analytical population**: restricted to loans with a concluded outcome (Fully Paid or
Charged Off) and a matured contractual term, since in-progress loans have no realized
label. Funnel: 2,260,701 → 673,553 (removing Current/Late/Grace/Default/"does not meet
credit policy" statuses, 33 footer rows, and immature vintages) → 673,314 (removing 5 rows
with impossible `dti` > 100 and 234 joint-application rows whose `dti` uses a different
basis). Population default rate: 14.81%.

**Label**: binary, derived from `loan_status` — 1 = Charged Off (realized loss), 0 = Fully
Paid. No label exists for in-progress loans, nor for applicants Lending Club rejected
(§7).

**Splits**: temporal, by `issue_d`, never random.

| Split | Rows | Default rate | Period | Term |
|---|---|---|---|---|
| Train | 172,988 | 12.43% | 2007-06 to 2013-12 | 36m |
| Validation | 162,570 | 13.73% | 2014 | 36m |
| Test | 282,787 | 14.88% | 2015 | 36m |
| Transfer (holdout) | 54,969 | 25.16% | 2010-05 to 2013-12 | 60m |

**Sensitive content**: each record describes a real borrower, including self-reported
income, employment title, a 3-digit ZIP prefix, and detailed credit-bureau attributes
(delinquencies, utilization, collections, inquiries). Pseudonymised at source: `member_id`
is 100% null throughout the file's history, and ZIP is truncated to 3 digits — no
borrower is re-identifiable from within this dataset alone. Free-text fields (`emp_title`,
`desc`, `title`) may carry self-disclosed personal detail and are excluded from modeling;
`emp_title` is retained only as a manual lookup aid during outlier auditing.

**Missing data**: extensive, mechanism-dependent (full table in `docs/FACTS.md` §3–4;
summarized in §4 below).

## 3. Data dictionary (representative, by family and missingness mechanism)

`train.parquet` has 89 columns: 1 target, 5 `EVAL_ONLY` (financial calculation only, never
a feature), 3 `EXCLUDED` (Lending Club's own risk grade — withheld to avoid predicting the
platform's own judgment), 65 "family C" predictive columns, 10 engineered missingness
flags, 5 engineered ratio features. Full 89-row table in `docs/FACTS.md` §3; representative
rows by mechanism follow.

| Column | Type | Description | Example | % missing | Mechanism / treatment |
|---|---|---|---|---|---|
| `target` | int64 | 1=Charged Off, 0=Fully Paid | 0 | N/A | derived label |
| `loan_amnt` | float64 | Amount applied for | 5000.0 | 0.0% | none |
| `int_rate` | float64 | LC-assigned interest rate | 10.65 | 0.0% | none; excluded as feature |
| `annual_inc` | float64 | Self-reported annual income | 24000.0 | 0.0% | none |
| `fico_range_low` | float64 | Lower FICO band bound | 735.0 | 0.0% | none |
| `mths_since_last_delinq` | float64 | Months since last delinquency | 999.0 | 51.77% | MNAR: flag + sentinel 999 |
| `mths_since_last_record` | float64 | Months since last public record | 999.0 | 84.58% | MNAR: flag + sentinel 999 |
| `emp_length_anos` | float64 | Years employed | -1 | 5.58% | MNAR: flag + sentinel -1 |
| `tot_cur_bal` | float64 | Total current balance | -1.0 | 10.03% | 2012 rollout: sentinel -1 + `era_pre_2012` flag |
| `bc_util` | float64 | Bankcard balance/limit ratio | 999.0 | 7.99% | rollout+structural: sentinel 999 |
| `mths_since_recent_inq` | float64 | Months since last inquiry | 999.0 | 16.97% | rollout+structural, mostly non-rollout: sentinel 999 + own flag |
| `dti` | float64 | Debt-to-income ratio | 27.65 | 0.0003% | sparse: median + aggregate flag |
| `revol_util` | float64 | Revolving utilization | 83.7 | 0.0561% | sparse: median + aggregate flag |
| `num_tl_120dpd_2m` | float64 | Accounts 120d past due (recent) | -1.0 | 13.13% | structural, inverted signal: sentinel -1 + own flag |

Dropped entirely (never collected within the analytical window, per
`docs/column_birth_log.csv`): the Dec/2015 `open_acc_6m` block (13 cols), the Mar/2017
`sec_app_*` block (11 cols) plus `revol_bal_joint`, `member_id`, `next_pymnt_d`, `il_util`,
and the joint-application-only `*_joint` fields — all 97.8–100% null within the
population.

## 4. Missing-data mechanisms (summary)

| Mechanism | % null range | Evidence | Treatment |
|---|---|---|---|
| MNAR (informative absence) | 5.6–84.6% | Null share stable across vintages; default rate among nulls diverges (both directions) | Flag + sentinel (999 preserves ordering; -1 where none exists) |
| Staged rollout (2012 bureau block, ~40 cols) | 7.0–10.05% | 100% null through 2011, ~52% in 2012, 0% from 2013 | Sentinel -1, disambiguated by `era_pre_2012` |
| Rollout + structural, stacked (6 cols) | 7.9–17.0% | Falls post-2012, never reaches 0% — structural residual | Sentinel 999; one column also gets its own flag |
| Sparse/informative (6 cols, 1,079 rows) | 0.0003–0.10% | Affected rows default at 18.35% vs. 14.81% | Aggregate flag + per-column median |
| Structural, own mechanism (1 col) | 13.13% | Inverted default signal vs. mths_since_* family | Sentinel -1 + dedicated flag |
| Dropped (never collected in window) | 97.8–100% | Birth log confirms non-existence | Dropped entirely |

## 5. Collection Process

Lending Club collected the data as part of its own loan origination and servicing process
(borrower self-reported fields at application; credit-bureau pulls; internal
payment/status tracking through maturity or charge-off), for its own business purposes —
not for research. This project collected nothing directly; it consumes the Kaggle
republication as-is. No new consent for research use was obtained; use relies on the loans
having originally been part of a public release by Lending Club and on the CC0 terms of
the Kaggle republication (§8). No subcontractors were involved on this project's side.

## 6. Preprocessing, Cleaning, and Labeling

The 151 raw columns were classified by a single criterion against the official
dictionary — known at origination vs. born/updated afterward — into: 3 identifiers
(dropped), 37 post-origination columns (dropped as leakage, confirmed via univariate AUC —
`recoveries` and `collection_recovery_fee` nearly perfect-predict the outcome by
construction), 99 origination-time candidates, 4 free-text columns (not used as
features), and 5 target/evaluation columns (used only for financial calculations).
Zero-variance columns (`policy_code`, `hardship_flag`, `out_prncp`, others) were dropped.
Rare categories in `home_ownership`/`purpose` were consolidated via a binomial-CI-width
rule, not an arbitrary cutoff. Five rows with impossible `dti` (>100) and 234
joint-application rows (`dti` computed on a different basis) were dropped from the
population. The raw CSV is retained unmodified in `data/raw/`; every transformation is
re-derivable from it via notebooks 01–05 and `src/` — nothing is hand-edited.
`verification_status` is semantically inverted (verified income correlates with *higher*
default, 17.75% vs. 11.74%) — a selection effect, not an error — flagged wherever
interpreted.

## 7. Uses

**Intended use**: educational/portfolio demonstration of credit-risk modeling methodology —
temporal validation, profit-based evaluation, documented missing-data handling, and
honest reporting of model limitations.

**Discouraged / out-of-scope use**: this dataset and any model trained on it must **not**
be used for real lending decisions. The fundamental limitation is selection bias: the
dataset holds only loans Lending Club already chose to approve and fund. Every learned
pattern is conditional on that filter — the model estimates P(default | approved), never
P(default | applicant), since no outcome exists for rejected applicants. A profile the
platform rarely approved appears only through its exceptional cases, which can invert the
apparent risk reading for that group. Two further limitations compound this:
individual-level train/test independence is unverifiable (`member_id` is never populated,
so a repeat borrower can't be tracked across splits), and the 60-month loan population
degrades severely under a model trained on 36-month loans (AUC drop 0.04–0.07; M1's
profit gain turns negative) — the two terms are not interchangeable risk pools.

## 8. Distribution & License

The raw dataset is distributed via Kaggle (`wordsforthewise/lending-club`) under
**CC0 1.0 Universal (Public Domain Dedication)**:
<https://creativecommons.org/publicdomain/zero/1.0/>. It is a third-party republication of
data Lending Club originally made public; the original Lending Club data portal is
offline. The data dictionary used here is likewise a third-party copy (Kaggle user
`jonchan2003`), for the same reason. This project's own code, notebooks, and derived
documentation are distributed under its GitHub repository's terms (§9); the CC0
dedication covers only the underlying loan data.

## 9. Maintenance

Maintained by the project author, Mateus Fardin Pavan, as a personal portfolio project —
not by Lending Club or Kaggle. Repository:
<https://github.com/MateusFPavan/credit-default-prediction-lendingclub>. Contact via
GitHub or LinkedIn (<https://www.linkedin.com/in/mateus-fardin-pavan/>). Reproducible from
the raw CSV via notebooks 01–14 and `src/`; `models/model_meta.json` records a SHA256 hash
of the exact `train.parquet` used to fit the final model. Versioned via git; corrections
are applied as commits with rationale in `docs/cleaning_decisions.md`. No formal erratum
process beyond commit history, and no planned update schedule — a static, one-time
analysis, not a maintained data feed.

---

**Self-check**: provenance ✓ (§1–§2), data dictionary ✓ (§3), collection process ✓ (§5),
preprocessing/exclusions ✓ (§6), uses + discouraged uses ✓ (§7), license ✓ (§8).
