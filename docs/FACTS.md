# FACTS.md: Canonical Facts Sheet

Single source of truth for all downstream documentation (README, technical report,
one-pager). Every number here was extracted by direct reading of the project's actual
artifacts (parquet files, `models/model_meta.json`, `docs/column_birth_log.csv`,
`docs/column_inventory.csv`, the official data dictionary, and the committed notebooks) on
2026-07-12, not copied from memory or from prose in other docs. Where a fact could not be
found in an artifact, it is marked **N/A**, with an explanation.

A verification pass cross-checked every number below against `docs/scope.md` and
`docs/cleaning_decisions.md`. Divergences found are called out explicitly in
**Section 8: Verification notes**. That section should be read before this sheet is used
as a source.

---

## 1. Dataset provenance

- **Source**: Kaggle dataset `wordsforthewise/lending-club` (public mirror of Lending
  Club's own historical data release).
- **Raw file**: `data/raw/accepted_2007_to_2018Q4.csv`.
  - **Size on disk**: 1,675,133,810 bytes (≈1.56 GiB / 1.67 GB), confirmed via direct file
    stat.
  - **Rows**: 2,260,701 data rows (2,260,702 lines including header), confirmed via `wc
    -l` on the raw file and via a full read of the `loan_status` column.
  - **Columns**: 151 raw columns, confirmed via header row (`head -1 | tr ',' '\n' | wc
    -l`) and matching `docs/column_inventory.csv`'s row count (151 entries).
- **Known third-party transformations** (per `docs/scope.md` §7): the `%` symbol was
  stripped from `int_rate` and `revol_util` and both were cast to float, before this
  project received the file. This is the only documented pre-processing done upstream of
  this project.
- **Data dictionary**: `docs/Lending Club Data Dictionary Approved.csv` (155 lines,
  `LoanStatNew` naming, matching raw column names) and
  `docs/Lending Club Data Dictionary Notes.csv` (123 lines, `BrowseNotesFile` camelCase
  naming, a secondary source not used for the lookup below). Both are third-party
  copies. The original Lending Club dictionary source went offline when the platform shut
  down (`docs/scope.md` §7).
  - Of the 151 raw columns, **150 have an exact-name definition** in the Approved
    dictionary. The 151st, `verification_status_joint`, has a definition present but
    filed under a typo'd key, `verified_status_joint`. This is documented in
    `docs/cleaning_decisions.md` ("the dictionary entry verified_status_joint is a
    typo") and confirmed by direct grep on the dictionary file. Effectively 151/151 have a
    findable definition, 150 by exact match and 1 via the known typo.
  - `total_rev_hi_lim` initially appeared undefined in an automated key-match. The cause
    was a trailing non-breaking space (`\xa0`) in the dictionary's own key, not a missing
    entry. Noted here since it is exactly the kind of silent extraction error this sheet
    is meant to catch.
- **SHA256 of `train.parquet`** (the exact file the final model was trained on, per
  `models/model_meta.json`): `0f7e711edc7c20839c0bd569a0c6f6e3125cadd3dcd1264a57a323b2b89a1191`.

---

## 2. Population funnel

All rows below were reproduced by directly reading `loan_status`, `issue_d`, and `term`
from the raw CSV (not copied from `docs/scope.md`) and independently recomputing every
cutoff. **All figures reconcile exactly with `docs/scope.md`. No divergence found.**

| Reason | Rows | % of total in file |
|---|---|---|
| Total in file | 2,260,701 | 100.00% |
| Status Current | 878,317 | 38.85% |
| Status Late (31-120 days) | 21,467 | 0.95% |
| Status In Grace Period | 8,436 | 0.37% |
| Status Late (16-30 days) | 4,349 | 0.19% |
| Does not meet the credit policy (Fully Paid + Charged Off) | 2,749 (1,988 + 761) | 0.12% |
| Status Default | 40 | 0.002% |
| CSV footer rows (no loan_status) | 33 | 0.0015% |
| Immature vintage (36-month, issued after Dec/2015) | 402,159 | 17.79% |
| Immature vintage (60-month, issued after Dec/2013) | 269,598 | 11.93% |
| **Analytical population** (Fully Paid + Charged Off, mature vintages) | **673,553** | **29.79%** |

Default rate in the analytical population: **14.8147%**, recomputed directly from the raw
CSV (Charged Off rows / total concluded, mature-vintage rows), matching `docs/scope.md`'s
"14.81%".

Second-stage exclusion (recorded in `docs/cleaning_decisions.md`, confirmed against
`data/processed/loans_clean.parquet`):

| Step | Rows | Delta |
|---|---|---|
| Analytical population | 673,553 | — |
| minus: 5 rows with impossible dti (>100) | -5 | |
| minus: 234 joint-application rows (dti systematically distorted, see §4) | -234 | |
| **Final cleaned population** (`loans_clean.parquet`) | **673,314** | **-239 total** |

`loans_clean.parquet` shape confirmed by direct read: **(673,314 rows, 84 columns)**. This
is the artifact the task's phrase "84 colunas" refers to. See §8 for why
`train.parquet` has 89, not 84, columns.

Within the final cleaned population: `loan_status` value counts are Fully Paid 573,572 /
Charged Off 99,742 (default rate 14.8136%, matching the population-level rate to within
rounding). Term split: 36-month 618,345 rows (91.79%), 60-month 54,969 rows (8.21%),
matching `docs/scope.md` §9's "618,345 (91.8%)" exactly.

### Split sizes and default rates (directly read from each parquet)

| Split | File | N | Default rate | issue_d range | term |
|---|---|---|---|---|---|
| Train | `train.parquet` | 172,988 | 12.4332% | 2007-06 to 2013-12 | 36m |
| Validation | `validation.parquet` | 162,570 | 13.7264% | 2014-01 to 2014-12 | 36m |
| Test | `test.parquet` | 282,787 | 14.8836% | 2015-01 to 2015-12 | 36m |
| Transfer (60m) | `transfer_60m.parquet` | 54,969 | 25.1596% | 2010-05 to 2013-12 | 60m |

172,988 + 162,570 + 282,787 = 618,345, matching the 36-month population total above.
Default rate rises monotonically from train to validation to test (12.43% to 13.73% to
14.88%), matching `docs/scope.md` §10 exactly.

---

## 3. Data dictionary (train.parquet's 89 columns)

`train.parquet` has **89 columns** (confirmed by direct read), not 84. See §8 for the
reconciliation. Grouped by family. "% null (pre-treatment)" is the null rate **within the
final analytical population**, from `docs/column_inventory.csv`'s
`%nulos_populacao_aprovada` column (computed before any imputation/sentinel was applied).
"N/A (engineered)" means the column does not exist in the raw file. "Example value" is the
first non-null value found in `train.parquet`.

### Target (1 column)

| Column | dtype | % null (pre-treatment) | Mechanism | Treatment | Example | Dictionary definition |
|---|---|---|---|---|---|---|
| target | int64 | N/A (engineered) | none | derived from loan_status: 1=Charged Off, 0=Fully Paid | 0 | N/A — engineered, not in official dictionary |

### EVAL_ONLY: never a model feature, used only for the profit calculation (5 columns)

| Column | dtype | % null (pre-treatment) | Mechanism | Treatment | Example | Dictionary definition |
|---|---|---|---|---|---|---|
| loan_status | str | 0.0% | none | none | Fully Paid | Current status of the loan |
| loan_amnt | float64 | 0.0% | none | none | 5000.0 | The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value. |
| installment | float64 | 0.0% | none | none | 162.87 | The monthly payment owed by the borrower if the loan originates. |
| term | float64 | 0.0% | none | parsed to integer months | 36.0 | The number of payments on the loan. Values are in months and can be either 36 or 60. |
| total_rec_prncp | float64 | 0.0% | none | none | 5000.0 | Principal received to date |

### EXCLUDED: Lending Club's own risk assessment, deliberately withheld (3 columns)

| Column | dtype | % null (pre-treatment) | Mechanism | Treatment | Example | Dictionary definition |
|---|---|---|---|---|---|---|
| int_rate | float64 | 0.0% | none | none (excluded from FEATURE_SET) | 10.65 | Interest Rate on the loan |
| grade | str | 0.0% | none | none (excluded from FEATURE_SET) | b | LC assigned loan grade |
| sub_grade | str | 0.0% | none | none (excluded from FEATURE_SET) | b2 | LC assigned loan subgrade |

### Family C: available at origination (65 columns; 64 are in FEATURE_SET, 1 excluded for redundancy)

| Column | dtype | % null (pre-treatment) | Mechanism | Treatment | Example | Dictionary definition |
|---|---|---|---|---|---|---|
| funded_amnt | float64 | 0.0% | none | none | 5000.0 | The total amount committed to that loan at that point in time. |
| home_ownership | str | 0.0% | none | rare categories (any/none/other) collapsed into other | rent | The home ownership status provided by the borrower... RENT, OWN, MORTGAGE, OTHER |
| annual_inc | float64 | 0.0% | none | none | 24000.0 | The self-reported annual income provided by the borrower during registration. |
| verification_status | str | 0.0% | none | none — semantically inverted, see §5 | verified | Indicates if income was verified by LC, not verified, or if the income source was verified |
| issue_d | datetime64 | 0.0% | none | cast to datetime | 2011-12-01 | The month which the loan was funded |
| purpose | str | 0.0% | none | rare categories (wedding/educational/renewable_energy) collapsed into other | credit_card | A category provided by the borrower for the loan request. |
| dti | float64 | 0.0003% | sparse (informative) | median imputation + sparse_bureau_missing flag; separately, 5 rows >100 and 234 joint-app rows dropped at the population level (§2) | 27.65 | Ratio of total monthly debt payments (excl. mortgage and the requested loan) to self-reported monthly income |
| delinq_2yrs | float64 | 0.0% | none | none | 0.0 | Number of 30+ days past-due incidences in the past 2 years |
| earliest_cr_line | datetime64 | 0.0% | none | cast to datetime | 1985-01-01 | The month the borrower's earliest reported credit line was opened |
| fico_range_low | float64 | 0.0% | none | none | 735.0 | Lower boundary range of the borrower's FICO at origination |
| fico_range_high | float64 | 0.0% | none | dropped from FEATURE_SET (r=1.0 with fico_range_low; kept as a raw column) | 739.0 | Upper boundary range of the borrower's FICO at origination |
| inq_last_6mths | float64 | 0.0% | none | none | 1.0 | Number of inquiries in past 6 months (excl. auto/mortgage) |
| mths_since_last_delinq | float64 | 51.7655% | MNAR | flag (mths_since_last_delinq_missing) + sentinel 999 | 999.0 | Number of months since the borrower's last delinquency |
| mths_since_last_record | float64 | 84.5785% | MNAR | flag (mths_since_last_record_missing) + sentinel 999 | 999.0 | Number of months since the last public record |
| open_acc | float64 | 0.0% | none | none | 3.0 | Number of open credit lines |
| pub_rec | float64 | 0.0% | none | none | 0.0 | Number of derogatory public records |
| revol_bal | float64 | 0.0% | none | none | 13648.0 | Total credit revolving balance |
| revol_util | float64 | 0.0561% | sparse (informative) | median imputation + sparse_bureau_missing flag | 83.7 | Revolving line utilization rate |
| total_acc | float64 | 0.0% | none | none | 9.0 | Total number of credit lines |
| initial_list_status | str | 0.0% | none | none — genuine policy shift over time, not a null issue (see §8) | f | Initial listing status: W or F |
| collections_12_mths_ex_med | float64 | 0.0083% | sparse (informative) | median imputation + sparse_bureau_missing flag | 0.0 | Number of collections in 12 months excluding medical |
| mths_since_last_major_derog | float64 | 75.4238% | MNAR | flag (mths_since_last_major_derog_missing) + sentinel 999 | 999.0 | Months since most recent 90-day or worse rating |
| application_type | str | 0.0% | none | rare category (joint app, 0.04%) kept as-is | individual | Individual or joint application |
| acc_now_delinq | float64 | 0.0% | none | none | 0.0 | Number of accounts on which the borrower is now delinquent |
| tot_coll_amt | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Total collection amounts ever owed |
| tot_cur_bal | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Total current balance of all accounts |
| total_rev_hi_lim | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Total revolving high credit/credit limit |
| acc_open_past_24mths | float64 | 7.0196% | rollout-2012 (exact 4-column mask) | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of trades opened in past 24 months |
| avg_cur_bal | float64 | 10.0271% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Average current balance of all accounts |
| bc_open_to_buy | float64 | 7.9286% | rollout-2012 + structural (stacked) | sentinel 999 | 999.0 | Total open to buy on revolving bankcards |
| bc_util | float64 | 7.9896% | rollout-2012 + structural (stacked) | sentinel 999 | 999.0 | Ratio of current balance to limit, bankcard accounts |
| chargeoff_within_12_mths | float64 | 0.0083% | sparse (informative) | median imputation + sparse_bureau_missing flag | 0.0 | Number of charge-offs within 12 months |
| delinq_amnt | float64 | 0.0% | none | none | 0.0 | Past-due amount on delinquent accounts |
| mo_sin_old_il_acct | float64 | 13.2947% | rollout-2012 + structural (stacked) | sentinel 999 | 999.0 | Months since oldest bank installment account opened |
| mo_sin_old_rev_tl_op | float64 | 10.0256% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Months since oldest revolving account opened |
| mo_sin_rcnt_rev_tl_op | float64 | 10.0256% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Months since most recent revolving account opened |
| mo_sin_rcnt_tl | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Months since most recent account opened |
| mort_acc | float64 | 7.0196% | rollout-2012 (exact 4-column mask) | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of mortgage accounts |
| mths_since_recent_bc | float64 | 7.8611% | rollout-2012 + structural (stacked) | sentinel 999 | 999.0 | Months since most recent bankcard account opened |
| mths_since_recent_bc_dlq | float64 | 77.07% | MNAR | flag (mths_since_recent_bc_dlq_missing) + sentinel 999 | 999.0 | Months since most recent bankcard delinquency |
| mths_since_recent_inq | float64 | 16.9709% | rollout-2012 + structural, majority non-rollout | sentinel 999 + own flag (mths_since_recent_inq_missing) | 999.0 | Months since most recent inquiry |
| mths_since_recent_revol_delinq | float64 | 67.7555% | MNAR | flag (mths_since_recent_revol_delinq_missing) + sentinel 999 | 999.0 | Months since most recent revolving delinquency |
| num_accts_ever_120_pd | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of accounts ever 120+ days past due |
| num_actv_bc_tl | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of currently active bankcard accounts |
| num_actv_rev_tl | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of currently active revolving trades |
| num_bc_sats | float64 | 8.2905% | rollout-2012 (Jun/2012 variant) | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of satisfactory bankcard accounts |
| num_bc_tl | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of bankcard accounts |
| num_il_tl | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of installment accounts |
| num_op_rev_tl | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of open revolving accounts |
| num_rev_accts | float64 | 10.0256% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of revolving accounts |
| num_rev_tl_bal_gt_0 | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of revolving trades with balance > 0 |
| num_sats | float64 | 8.2905% | rollout-2012 (Jun/2012 variant) | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of satisfactory accounts |
| num_tl_120dpd_2m | float64 | 13.1344% | structural (own mechanism, inverted signal) | sentinel -1 + own flag (num_tl_120dpd_2m_missing) | -1.0 | Number of accounts currently 120 days past due (updated in last 2 months) |
| num_tl_30dpd | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of accounts currently 30 days past due |
| num_tl_90g_dpd_24m | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of accounts 90+ days past due in last 24 months |
| num_tl_op_past_12m | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Number of accounts opened in past 12 months |
| pct_tl_nvr_dlq | float64 | 10.0482% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Percent of trades never delinquent |
| percent_bc_gt_75 | float64 | 7.9813% | rollout-2012 + structural (stacked) | sentinel 999 | 999.0 | Percentage of bankcard accounts > 75% of limit |
| pub_rec_bankruptcies | float64 | 0.1035% | sparse (informative) | median imputation + sparse_bureau_missing flag | 0.0 | Number of public record bankruptcies |
| tax_liens | float64 | 0.0058% | sparse (informative) | median imputation + sparse_bureau_missing flag | 0.0 | Number of tax liens |
| tot_hi_cred_lim | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Total high credit/credit limit |
| total_bal_ex_mort | float64 | 7.0196% | rollout-2012 (exact 4-column mask) | sentinel -1, covered by era_pre_2012 flag | -1.0 | Total credit balance excluding mortgage |
| total_bc_limit | float64 | 7.0196% | rollout-2012 (exact 4-column mask) | sentinel -1, covered by era_pre_2012 flag | -1.0 | Total bankcard high credit/credit limit |
| total_il_high_credit_limit | float64 | 10.0255% | rollout-2012 | sentinel -1, covered by era_pre_2012 flag | -1.0 | Total installment high credit/credit limit |
| emp_length_anos | float64 | 5.578% (raw `emp_length`, pre-transformation) | MNAR | flag (emp_length_missing) + sentinel -1; parsed from string ("< 1 year"→0, "10+ years"→10) | 10.0 | N/A — engineered from `emp_length` (str), not itself in the dictionary |

### Engineering flags (10 columns, all int64, 0/1, never null by construction)

| Column | Purpose | Example |
|---|---|---|
| era_pre_2012 | Disambiguates sentinel from signal in the 2012 bureau-rollout block | 1 |
| mths_since_last_delinq_missing | MNAR flag for mths_since_last_delinq | 1 |
| mths_since_last_record_missing | MNAR flag for mths_since_last_record | 1 |
| mths_since_recent_bc_dlq_missing | MNAR flag for mths_since_recent_bc_dlq | 1 |
| mths_since_recent_revol_delinq_missing | MNAR flag for mths_since_recent_revol_delinq | 1 |
| mths_since_last_major_derog_missing | MNAR flag for mths_since_last_major_derog | 1 |
| emp_length_missing | MNAR flag for emp_length_anos | 0 |
| mths_since_recent_inq_missing | Own flag for mths_since_recent_inq's non-rollout nulls | 1 |
| num_tl_120dpd_2m_missing | Own flag for num_tl_120dpd_2m's structural nulls | 1 |
| sparse_bureau_missing | Aggregate flag for the 6-column sparse mechanism (§4) | 0 |

### New engineered features (5 columns, all row-wise ratios from `src.features.build_features`)

| Column | dtype | Formula | Example |
|---|---|---|---|
| installment_to_income | float64 | installment / (annual_inc / 12) | 0.081435 |
| loan_to_income | float64 | loan_amnt / annual_inc | 0.208333 |
| credit_history_months | int32 | months between earliest_cr_line and issue_d | 323 |
| revol_bal_to_income | float64 | revol_bal / annual_inc | 0.568667 |
| open_acc_ratio | float64 | open_acc / total_acc | 0.333333 |

---

## 4. Missing-data mechanisms (summary)

| Mechanism | Columns | Typical % null | Evidence | Treatment |
|---|---|---|---|---|
| **MNAR (informative absence)** | mths_since_last_delinq (51.77%), mths_since_last_record (84.58%), mths_since_recent_bc_dlq (77.07%), mths_since_recent_revol_delinq (67.76%), mths_since_last_major_derog (75.42%), emp_length (5.58%, pre-transformation) | 5.6%-84.6% | Null share stable across every vintage (never drops to zero); default rate among nulls consistently LOWER (-0.57 to -2.32pp) for the mths_since_* family, but HIGHER (20.84% vs 13.71-15.20%) for emp_length | Binary flag + sentinel 999 (mths_since_* family, preserves "higher = safer" ordering) or sentinel -1 + flag (emp_length_anos, a count/duration in years, no such ordering) |
| **Staged rollout — 2012 bureau block** | ~40 columns (tot_coll_amt, tot_cur_bal, total_rev_hi_lim, avg_cur_bal, mort_acc, acc_open_past_24mths, total_bal_ex_mort, total_bc_limit, mo_sin_*, num_*, pct_tl_nvr_dlq, tot_hi_cred_lim, total_il_high_credit_limit, and others) | 7.02%-10.05% (num_bc_sats/num_sats at 8.29%, a slightly later Jun/2012 sub-rollout) | 100% null through 2011, ~52% null in 2012, 0% from 2013 onward (`docs/column_birth_log.csv`); 4 columns (mort_acc, total_bc_limit, total_bal_ex_mort, acc_open_past_24mths) share an identical null mask at 7.0197% | Sentinel -1, disambiguated by the era_pre_2012 flag (no separate flag per column) |
| **Staged rollout + structural (stacked)** | bc_util, bc_open_to_buy, percent_bc_gt_75, mths_since_recent_bc, mo_sin_old_il_acct, mths_since_recent_inq | 7.86%-16.97% | Falls sharply after 2012 but never reaches 0% in mature vintages — a residual 1%-11%+ persists structurally (e.g., bankcard utilization cannot be computed for a borrower with no bankcard) | Sentinel 999 (preserves ordering); mths_since_recent_inq additionally gets its own flag since >50% of its nulls (16.97% total, only 43.69% overlap with era_pre_2012) are non-rollout |
| **Sparse (informative, low-volume)** | pub_rec_bankruptcies, revol_util, chargeoff_within_12_mths, collections_12_mths_ex_med, tax_liens, dti | 0.0003%-0.1035% (1,079 unique rows total) | Affected rows default at 18.35% vs. population 14.81% (a 3.54pp gap); 65% concentrated in 2007-2008 vintages, consistent with immature collection processes | Aggregate flag sparse_bureau_missing (columns rarely co-occur) + per-column median imputation |
| **Structural, own mechanism (inverted signal)** | num_tl_120dpd_2m | 13.13% | Residual 0.19%-5.03% in mature vintages; 23.67% of nulls unrelated to rollout; null rows default 1.15pp MORE than filled — the opposite direction of mths_since_recent_inq | Sentinel -1 (a count with no "higher = safer" ordering) + own flag num_tl_120dpd_2m_missing |
| **Dropped — structural absence (never collected in window)** | Dec/2015 open_acc_6m block (13 cols, ~97.8% null within population), Mar/2017 sec_app_* block (11 cols) + revol_bal_joint (100% null), member_id, next_pymnt_d, il_util, dti_joint/annual_inc_joint/verification_status_joint (joint-app only) | 97.8%-100% | Column birth log shows the field literally did not exist yet for the analytical window | Dropped entirely — not present in train.parquet |

---

## 5. Final results (test set, frozen, from notebooks 12/13/14)

### Point estimates and bootstrap CIs (test 2015, N=282,787, 1,000 resamples, seed=42)

| Model | Threshold | AUC-ROC [95% CI] | Brier [95% CI] | Profit [95% CI] | % approved | Default among approved |
|---|---|---|---|---|---|---|
| M0b (approve all) | — | N/A (no score) | N/A | $233,202,813.06 [$228,626,556.25, $237,526,093.71] | 100.00% | 14.8836% |
| M1 (LogReg) | 0.38 | 0.6847 [0.6820, 0.6874] | 0.1210 [0.1200, 0.1220] | $235,936,408.63 [$231,368,999.00, $240,458,423.72] | 98.88% | 14.6184% |
| XGB_walkforward | 0.31 | 0.6846 [0.6818, 0.6871] | 0.1205 [0.1195, 0.1214] | $242,230,710.89 [$237,888,392.29, $246,720,384.43] | 96.24% | 14.0492% |

Bootstrap differences (paired, same 1,000 resamples):

| Comparison | Mean difference | 95% CI | Crosses zero? | Win rate |
|---|---|---|---|---|
| M1 vs M0b | +$2,750,813.30 | [$2,037,165.24, $3,542,444.80] | No | 100.00% M1 |
| XGB_walkforward vs M0b | +$9,068,587.76 | [$7,627,319.06, $10,664,983.20] | No | 100.00% XGB |
| M1 vs XGB_walkforward | -$6,317,774.45 | [-$7,706,571.79, -$4,931,521.12] | **No** | 100.00% XGB, 0.00% M1 |

XGB_walkforward is the statistically distinguishable winner on the test set (the paired
CI does not cross zero).

### Validation to test: gain over M0b did not shrink (it grew)

| Model | Gain on validation | Gain on test | Change | Change % |
|---|---|---|---|---|
| M1 | $874,222.01 | $2,733,595.57 | +$1,859,373.56 | +212.69% |
| XGB_walkforward | $1,865,541.47 | $9,027,897.83 | +$7,162,356.36 | +383.93% |

### Error decomposition (XGB_walkforward, threshold 0.31, test)

| Metric | Value |
|---|---|
| N rejected | 10,644 |
| Total value rejected | $136,302,375.00 |
| Rejected, charged off (correct rejection) | 3,855 |
| Rejected, fully paid (lost interest) | 6,789 |
| Avoided loss | $32,151,026.71 |
| Lost interest | $23,123,128.88 |
| Net gain (= profit above M0b) | $9,027,897.83 |

Confusion matrix at threshold 0.31 (test, N=282,787): TP (rejected, defaulted) 3,855, TN
(approved, paid) 233,909, FN (approved, defaulted) 38,234, FP (rejected, paid) 6,789. Only
9.16% of actual defaulters (3,855 of 42,089) are rejected by the model.

Cost of each error type: FN (approved bad loans) cost $274,484,090.07 (N=38,234), FP
(rejected good loans) cost $23,123,128.88 (N=6,789). FN cost is ~11.9x FP cost.

### 60-month transfer (both models trained on 36-month data only)

| Model | AUC 36m | AUC 60m | Δ AUC | Gain vs M0b (36m) | Gain vs M0b (60m) | Δ gain % |
|---|---|---|---|---|---|---|
| M1 | 0.6847 | 0.6120 | -0.0727 | $2,733,595.57 | -$20,472.60 | -100.75% |
| XGB_walkforward | 0.6846 | 0.6433 | -0.0412 | $9,027,897.83 | $1,066,477.36 | -88.19% |

Default rate: 25.16% (60m) vs. 14.88% (36m, test). Degradation is severe for both models
(M1's gain turns negative). This is evidence of structurally distinct populations between
36- and 60-month loans, per the reading criterion set in `docs/scope.md` §9.

### Subgroup performance (test, XGB_walkforward)

By grade:

| Grade | N | Default rate | AUC-ROC |
|---|---|---|---|
| A | 70,100 | 5.42% | 0.6477 |
| B | 91,645 | 11.89% | 0.6048 |
| C | 77,348 | 19.44% | 0.5965 |
| D | 32,668 | 26.21% | 0.5889 |
| E | 9,424 | 32.84% | 0.5818 |
| F | 1,358 | 42.42% | 0.5878 |
| G | 244 | 46.31% | 0.5854 |

By income quartile:

| Quartile | N | Default rate | Mean income | AUC-ROC |
|---|---|---|---|---|
| Q1 (lowest) | 71,696 | 19.09% | $32,845 | 0.6480 |
| Q2 | 71,580 | 15.56% | $52,952 | 0.6722 |
| Q3 | 72,906 | 13.62% | $75,406 | 0.6881 |
| Q4 (highest) | 66,605 | 11.01% | $141,672 | 0.6974 |

By purpose (main categories, sorted by N descending, as in notebook 13):

| Purpose | N | Default rate | AUC-ROC |
|---|---|---|---|
| debt_consolidation | 160,841 | 15.73% | 0.6804 |
| credit_card | 70,592 | 12.23% | 0.6896 |
| home_improvement | 17,145 | 13.51% | 0.6867 |
| other | 15,399 | 16.92% | 0.6619 |
| major_purchase | 5,340 | 14.46% | 0.6762 |
| medical | 3,144 | 17.84% | 0.6581 |
| car | 2,768 | 13.29% | 0.6888 |
| small_business | 2,423 | 22.95% | 0.6367 |
| vacation | 2,070 | 16.33% | 0.6504 |
| moving | 2,061 | 20.33% | 0.6762 |
| house | 1,004 | 21.61% | 0.6680 |

Calibration by decile (predicted probability vs. observed default, test): the model
consistently **underestimates** default rate: observed exceeds predicted in every decile
(e.g., decile 1: 3.30% observed vs. 2.46% predicted, decile 10: 31.80% observed vs. 30.86%
predicted). Profit per decile (if that decile alone were fully approved) turns negative in
the riskiest decile (-$11,289,402.27), consistent with the 0.31 threshold sitting near
where per-decile profit crosses zero.

### SHAP (test, 50,000-row stratified sample: full test set estimated at ~22 min, exceeding the 10-min budget)

Top 10 features by mean |SHAP| (margin/log-odds space), with their gain-based rank for
comparison:

| Rank (SHAP) | Feature | Mean abs SHAP | Rank (gain) |
|---|---|---|---|
| 1 | fico_range_low | 0.2013 | 1 |
| 2 | installment_to_income | 0.1677 | 10 |
| 3 | annual_inc | 0.1266 | 7 |
| 4 | acc_open_past_24mths | 0.1240 | 3 |
| 5 | dti | 0.0942 | 30 |
| 6 | revol_util | 0.0722 | 18 |
| 7 | inq_last_6mths | 0.0649 | 5 |
| 8 | total_bc_limit | 0.0640 | 20 |
| 9 | mo_sin_old_rev_tl_op | 0.0635 | 42 |
| 10 | purpose_credit_card | 0.0632 | 4 |

`verification_status` note: the two dummy levels behave differently under SHAP.
`verification_status_source verified` (mean SHAP +0.0281 when active vs. -0.0071 when
not) **preserves** the univariate inversion documented in
`docs/cleaning_decisions.md` (verified associates with higher default). But
`verification_status_verified` (mean SHAP -0.0039 when active vs. +0.0044 when not, rank
#61 of 90) **reverses** it, with a small effect. This suggests part of the univariate
"Verified" signal was confounded by other correlated features, once controlled for
multivariately. `era_pre_2012` has mean |SHAP| = 0.0000 (rank #90). It is never used by
the tree on the test set, since the flag is always 0 outside the training population.

---

## 6. Class balance

| Population | Default rate |
|---|---|
| Full analytical population (673,553 rows) | 14.81% |
| Train | 12.4332% |
| Validation | 13.7264% |
| Test | 14.8836% |
| Transfer (60-month) | 25.1596% |

Cost asymmetry (computed directly on `train.parquet` via `src.economics.compute_interest_loss`):

| Metric | Value |
|---|---|
| N good (target=0) | 151,480 |
| N bad (target=1) | 21,508 |
| Median interest, good loans | $2,023.62 |
| Median loss, bad loans | $5,398.84 |
| Ratio (median loss / median interest) | 2.67x |

A single bad loan costs, at the median, 2.67x what a single good loan returns. This is the
quantitative basis for `docs/scope.md`'s decision to weight by value rather than count
correct predictions.

---

## 7. Model facts (`models/model_meta.json`)

| Field | Value |
|---|---|
| Model | XGB_walkforward |
| max_depth | 8 |
| learning_rate | 0.03 |
| n_estimators | 600 |
| min_child_weight | 10 |
| subsample | 0.8 |
| colsample_bytree | 0.6 |
| random_state | 42 |
| n_jobs | 1 |
| eval_metric | logloss |
| FEATURE_SET size | 79 (raw names; expands to 90 columns after one-hot encoding of 5 categorical fields) |
| Operational threshold | 0.31 |
| sklearn version | 1.9.0 |
| xgboost version | 3.3.0 |
| python version | 3.12.10 |
| Training timestamp (UTC) | 2026-07-11T23:55:16.960768+00:00 |
| train.parquet SHA256 | 0f7e711edc7c20839c0bd569a0c6f6e3125cadd3dcd1264a57a323b2b89a1191 |
| train N rows | 172,988 |

Three conditions required for bit-exact reproducibility (established in notebook 11, and
verified again during the Phase 7 refactor's integrity check, documented in
`notebooks/14_train_final_model.ipynb`):

1. **random_state=42** on the estimator itself.
2. **n_jobs=1**: this environment's XGBoost is only bit-for-bit deterministic across
   separate runs at a fixed thread count (multi-threaded histogram building is not
   provably reproducible run-to-run).
3. **Training row order preserved**: the training rows must be used in `train.parquet`'s
   on-disk order, never re-sorted (e.g., by `issue_d`) before fitting. Row order measurably
   changes XGBoost's histogram splits even with the same seed.

---

## 8. Verification notes

Everything in this sheet was checked against `docs/scope.md` and
`docs/cleaning_decisions.md`, and against the notebooks' own recorded outputs. One
discrepancy surfaced, plus one clarification worth flagging explicitly:

1. **"84 columns" vs. 89 columns.** The task's brief for this sheet describes "as 84
   colunas finais do dataset processado", but instructs building the table "para CADA
   coluna de train.parquet". These are two different artifacts: `loans_clean.parquet`
   (notebook 03's output, the cleaned analytical population before the temporal split and
   before feature engineering) has exactly **84 columns**, confirmed by direct read. But
   `train.parquet` (notebook 04's temporal split + notebook 05's `build_features`, which
   adds `installment_to_income`, `loan_to_income`, `credit_history_months`,
   `revol_bal_to_income`, `open_acc_ratio`) has **89 columns**: 84 + 5. Both numbers are
   correct for their respective artifact. This sheet follows the explicit instruction
   ("cada coluna de train.parquet") and therefore reports 89, with this note so the
   discrepancy isn't mistaken for an error.
2. Every number in the population funnel (§2) was independently recomputed from the raw
   CSV rather than copied from `docs/scope.md`, and matched exactly (2,260,701 →
   673,553 → 673,314, default rate 14.81%, every intermediate exclusion count). No
   divergence found.
3. Every split size and default rate (§2) was read directly from the four processed
   parquet files and matches `docs/scope.md` §10 exactly (12.4332%, 13.7264%, 14.8836%,
   25.1596%).
4. All final-results figures in §5 were re-extracted directly from the committed outputs
   of notebooks 12 and 13 (not retyped from an earlier summary) and matched on a full
   re-read.
5. No numeric divergence was found anywhere else in this sheet.
