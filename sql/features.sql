-- Stage 2: one row per applicant (SK_ID_CURR) with credit-bureau, past-application,
-- and repayment-behaviour features. Read top to bottom: each CTE is one evidence source.

DROP TABLE IF EXISTS features;

CREATE TABLE features AS
WITH
-- Credit bureau: other lenders' view of the applicant.
bureau_agg AS (
  SELECT
    SK_ID_CURR,
    COUNT(*)                                                         AS bureau_loans,
    SUM(CREDIT_ACTIVE = 'Active')                                    AS bureau_active_loans,
    SUM(CREDIT_ACTIVE = 'Bad debt')                                  AS bureau_bad_debt_loans,
    MAX(CREDIT_DAY_OVERDUE)                                          AS bureau_max_days_overdue,
    SUM(COALESCE(AMT_CREDIT_SUM_DEBT, 0))                            AS bureau_total_debt,
    SUM(COALESCE(AMT_CREDIT_SUM_OVERDUE, 0))                         AS bureau_total_overdue,
    SUM(COALESCE(AMT_CREDIT_SUM_DEBT, 0)) * 1.0
      / NULLIF(SUM(COALESCE(AMT_CREDIT_SUM, 0)), 0)                  AS bureau_debt_to_credit,
    -MAX(DAYS_CREDIT) / 365.25                                       AS bureau_years_since_last_loan
  FROM bureau
  GROUP BY SK_ID_CURR
),

-- Past applications with this lender, plus the outcome of the most recent one.
prev_ranked AS (
  SELECT
    SK_ID_CURR, NAME_CONTRACT_STATUS, AMT_APPLICATION, AMT_CREDIT,
    ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION DESC) AS recency_rank
  FROM previous_application
),
prev_agg AS (
  SELECT
    SK_ID_CURR,
    COUNT(*)                                                         AS prev_applications,
    AVG(NAME_CONTRACT_STATUS = 'Refused')                            AS prev_refusal_rate,
    AVG(NAME_CONTRACT_STATUS = 'Approved')                           AS prev_approval_rate,
    AVG(AMT_CREDIT * 1.0 / NULLIF(AMT_APPLICATION, 0))               AS prev_granted_to_asked_ratio,
    MAX(CASE WHEN recency_rank = 1 AND NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)
                                                                     AS prev_last_was_refused
  FROM prev_ranked
  GROUP BY SK_ID_CURR
),

-- Repayment behaviour: how late and how short past instalments were paid.
inst_ranked AS (
  SELECT
    SK_ID_CURR,
    DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT                             AS days_late,
    AMT_PAYMENT * 1.0 / NULLIF(AMT_INSTALMENT, 0)                    AS paid_ratio,
    ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_INSTALMENT DESC) AS recency_rank
  FROM installments
  WHERE DAYS_ENTRY_PAYMENT IS NOT NULL
),
inst_agg AS (
  SELECT
    SK_ID_CURR,
    COUNT(*)                                                         AS inst_payments,
    AVG(days_late > 0)                                               AS inst_late_rate,
    AVG(CASE WHEN recency_rank <= 12 THEN (days_late > 0) END)       AS inst_late_rate_last12,
    MAX(days_late)                                                   AS inst_max_days_late,
    AVG(CASE WHEN days_late > 0 THEN days_late END)                  AS inst_avg_days_late_when_late,
    AVG(paid_ratio < 0.99)                                           AS inst_underpaid_rate
  FROM inst_ranked
  GROUP BY SK_ID_CURR
)

SELECT
  a.SK_ID_CURR,
  a.TARGET,

  -- Applicant profile
  a.NAME_CONTRACT_TYPE, a.CODE_GENDER, a.FLAG_OWN_CAR, a.FLAG_OWN_REALTY,
  a.NAME_INCOME_TYPE, a.NAME_EDUCATION_TYPE, a.NAME_FAMILY_STATUS, a.NAME_HOUSING_TYPE,
  a.CNT_CHILDREN, a.REGION_RATING_CLIENT,
  -a.DAYS_BIRTH / 365.25                                             AS age_years,
  -a.DAYS_EMPLOYED / 365.25                                          AS years_employed,
  a.DAYS_EMPLOYED_ANOM                                               AS not_employed_flag,
  -a.DAYS_ID_PUBLISH / 365.25                                        AS years_since_id_change,

  -- Affordability
  a.AMT_INCOME_TOTAL, a.AMT_CREDIT, a.AMT_ANNUITY,
  a.AMT_CREDIT / NULLIF(a.AMT_INCOME_TOTAL, 0)                       AS credit_to_income,
  a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0)                      AS annuity_to_income,
  a.AMT_CREDIT / NULLIF(a.AMT_ANNUITY, 0)                            AS loan_term_months,
  a.AMT_CREDIT / NULLIF(a.AMT_GOODS_PRICE, 0)                        AS credit_to_goods,

  -- External scores
  a.EXT_SOURCE_1, a.EXT_SOURCE_2, a.EXT_SOURCE_3,
  a.DEF_30_CNT_SOCIAL_CIRCLE, a.AMT_REQ_CREDIT_BUREAU_YEAR,

  b.bureau_loans, b.bureau_active_loans, b.bureau_bad_debt_loans, b.bureau_max_days_overdue,
  b.bureau_total_debt, b.bureau_total_overdue, b.bureau_debt_to_credit, b.bureau_years_since_last_loan,

  p.prev_applications, p.prev_refusal_rate, p.prev_approval_rate,
  p.prev_granted_to_asked_ratio, p.prev_last_was_refused,

  i.inst_payments, i.inst_late_rate, i.inst_late_rate_last12, i.inst_max_days_late,
  i.inst_avg_days_late_when_late, i.inst_underpaid_rate
FROM application a
LEFT JOIN bureau_agg b ON b.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN prev_agg   p ON p.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN inst_agg   i ON i.SK_ID_CURR = a.SK_ID_CURR;

CREATE UNIQUE INDEX ix_features_curr ON features(SK_ID_CURR);
