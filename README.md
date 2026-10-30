# CreditLens

Credit risk and audit-ready decision pipeline for consumer/MSME lending, built on the Home Credit Default Risk dataset.

## Plan
1. EDA + SQL feature pipeline over raw loan/bureau/repayment tables
2. Logistic regression baseline, then XGBoost default-prediction model
3. FastAPI prediction endpoint with inference-latency benchmarking
4. Rule-based drift check comparing live batches against training distribution
5. Lineage log so every prediction traces back to its source rows for audit review

Work in progress — see commit history for current stage.
