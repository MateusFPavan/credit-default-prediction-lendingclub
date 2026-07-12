# Turning Loan Approvals Into a Profit Decision

**Summary:** Built a credit-default model for real peer-to-peer consumer loans and
optimized it to maximize portfolio profit, not prediction accuracy — because for a
lender, those are not the same target.

## Headline Impact

**On the held-out 2015 test set, the model rejects just 3.8% of loan applications —
avoiding $32.2M in losses at a cost of $23.1M in foregone interest, for a net gain of
$9.0M over an approve-everyone policy.** That gain is statistically robust (95% CI
$7.6M–$10.7M) and holds up against a logistic-regression baseline too (+$6.3M).

## The Problem

Every loan a lender approves is a bet: a known potential return (interest) against a
known potential loss (unpaid principal). Approving too freely erodes the loan book with
defaults; rejecting too aggressively turns away paying customers. Getting the cutoff
right — and getting it right on genuinely new applicants, not on data the model already
knows — is the entire business problem. Real historical outcomes are available (~673K
matured Lending Club loans, 2007–2015, 14.8% default rate), making it possible to test a
decision policy against what actually happened, not a hypothetical.

## What I Did

- Optimized for portfolio profit, weighting each loan by its real dollar outcome, instead
  of treating every prediction error the same way.
- Validated the way a bank actually operates: trained on the past, tested on the future
  (walk-forward validation), never on a random shuffle of the data.
- Calibrated the model's probability estimates and stress-tested them, then audited where
  the model is weakest by borrower segment rather than reporting one aggregate score.

## Results

- **Net gain of +$9.0M** over approving every applicant — chosen by optimizing profit,
  not accuracy, and that choice mattered: the winning model ties a simpler baseline on
  AUC (0.68) but wins decisively on profit, value a standard accuracy contest would have
  missed entirely.
- **Rejecting the riskiest 10% of applicants avoids ~21% of all defaults** — twice as
  effective as a random cut of the same size.
- **Honest limitation, stated up front:** the model is least reliable in the highest-risk,
  lowest-income segment, and it was trained only on approved loans — it cannot score
  rejected applicants (a selection-bias limit), and it is not built for live lending
  decisions as-is.

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP

## Links

- Repository: https://github.com/MateusFPavan/credit-default-prediction-lendingclub
- Full technical report: `docs/technical_report.md`
- Contact: https://www.linkedin.com/in/mateus-fardin-pavan/
