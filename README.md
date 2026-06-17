# LendingClub XAI Professional Model Audit

This repository contains notebooks and data for the XAI model audit of LendingClub loan default predictions and a warmup tutorial on subscription churn.

## Dataset Explanations

Here is a detailed, structured explanation of the two CSV files located in the project's data directory (`data/`).

---

### **Overview**
1. **`lendingclub_loan_data.csv`** (approx. 875 KB, 9,578 rows, 15 columns)
   - **Purpose**: Used as the primary dataset in the main project workbook ([project_workbook.ipynb](project_workbook.ipynb)).
   - **Objective**: Build a default-risk predictor to identify risky loan applications for manual review and conduct a professional XAI (Explainable AI) model audit.
2. **`subscription_churn.csv`** (approx. 7 KB, 180 rows, 10 columns)
   - **Purpose**: Used as a simple, reproducible tutorial dataset in the warmup notebook ([tutorial_warmup.ipynb](tutorial_warmup.ipynb)).
   - **Objective**: Practice pipeline steps (EDA, splitting, modeling, SHAP/LIME explanation, subgroup analysis) on a small, fast-running dataset.

---

### **1. `lendingclub_loan_data.csv` (LendingClub Loan Dataset)**
This dataset contains historical loan records from LendingClub. The goal is to predict borrower default risk. 

* **Shape**: 9,578 rows, 15 columns
* **Target Variable**: `not.fully.paid` (1 = default/charged-off, 0 = paid back fully)
* **Positive Class Rate (Default Rate)**: ~16.0% (1,533 defaults vs. 8,045 fully paid loans)
* **Missing Values**: 0 null values across all columns.

#### **Detailed Column Breakdown & Statistical Insights**

| Column Name | Data Type | Description | Key Statistics & Ranges | Correlation with `not.fully.paid` |
| :--- | :--- | :--- | :--- | :--- |
| **`credit.policy`** | `int64` | `1` if the borrower meets LendingClub's credit underwriting criteria; `0` otherwise. | Mean: 0.80 (80% of applicants met criteria). | **`-0.158`** (Strong negative; meeting criteria decreases default risk). |
| **`purpose`** | `object` | The purpose category of the loan. | 7 unique values. Top: `debt_consolidation` (3,957), `all_other` (2,331), `credit_card` (1,262). | Categorical (Target-grouped variance). |
| **`int.rate`** | `float64` | The interest rate of the loan (as a proportion). | Range: 6.0% – 21.64% (Mean: 12.26%). | **`+0.160`** (Strong positive; higher interest rate = higher default risk). |
| **`installment`** | `float64` | The monthly installment (payment) amount owed by the borrower if the loan is funded. | Range: \$15.67 – \$940.14 (Mean: \$319.09). | `+0.050` (Slight positive correlation). |
| **`log.annual.inc`** | `float64` | The natural log of the borrower's self-reported annual income. | Range: 7.55 – 14.53 (Mean: 10.93 $\approx$ \$55,800/yr). | `-0.033` (Slight negative correlation). |
| **`dti`** | `float64` | Debt-To-Income ratio: total debt payments (excluding mortgage) divided by monthly income. | Range: 0.00 – 29.96 (Mean: 12.61). | `+0.037` (Slight positive correlation). |
| **`fico`** | `int64` | FICO credit score of the borrower. | Range: 612 – 827 (Mean: 710.8). | **`-0.150`** (Strong negative; higher credit score = lower default risk). |
| **`days.with.cr.line`** | `float64` | The number of days the borrower has had an active credit line. | Range: 178 – 17,640 days (Mean: 4,561 days $\approx$ 12.5 years). | `-0.029` (Slight negative correlation). |
| **`revol.bal`** | `int64` | Revolving balance: amount unpaid at the end of the credit card billing cycle. | Range: \$0 – \$1.2M (Mean: \$16,914). | `+0.054` (Slight positive correlation). |
| **`revol.util`** | `float64` | Revolving line utilization rate: credit limit utilization percentage. | Range: 0.0% – 119.0% (Mean: 46.8%). | `+0.082` (Moderate positive correlation). |
| **`inq.last.6mths`** | `int64` | Number of inquiries by creditors in the last 6 months (signals search for credit). | Range: 0 – 33 inquiries (Mean: 1.58). | **`+0.149`** (Strong positive; more inquiries = higher default risk). |
| **`delinq.2yrs`** | `int64` | The number of times the borrower was 30+ days past due on a payment in the past 2 years. | Range: 0 – 13 (Mean: 0.16; mostly 0). | `+0.009` (Very low correlation). |
| **`pub.rec`** | `int64` | Number of derogatory public records (e.g. bankruptcies, tax liens, judgments). | Range: 0 – 5 (Mean: 0.06). | `+0.049` (Slight positive correlation). |
| **`not.fully.paid`** | `int64` | **Target label**: `1` if the loan defaulted/charged off; `0` if paid in full. | Binary: `0` (repaid) or `1` (defaulted). | **1.000** (Self). |
| **`loan_status`** | `object` | The text outcome of the loan. | `"Fully Paid"` (8,045) or `"Charged Off"` (1,533). | **Perfect target leakage.** |

> [!WARNING]
> **Data Leakage Alert:** There is a perfect 1-to-1 match between the target variable `not.fully.paid` and the column `loan_status` (e.g. `"Fully Paid"` maps to `0`, and `"Charged Off"` maps to `1`). During modeling, **`loan_status` must be dropped** as a feature to prevent target leakage.

---

### **2. `subscription_churn.csv` (Subscription Churn Dataset)**
This small dataset represents subscription metrics for customer accounts. It is used to quickly demonstrate binary classification models and local/global explainability.

* **Shape**: 180 rows, 10 columns
* **Target Variable**: `churned` (1 = customer churned, 0 = customer stayed)
* **Positive Class Rate (Churn Rate)**: ~35.6% (64 churns vs. 116 retained)
* **Missing Values**: 0 null values.

#### **Detailed Column Breakdown & Statistical Insights**

| Column Name | Data Type | Description | Key Statistics & Ranges | Correlation with `churned` |
| :--- | :--- | :--- | :--- | :--- |
| **`customer_id`** | `object` | Unique customer identifier (e.g. `C0001`, `C0002`). | 180 unique strings. Unique key. | Drop (leakage/identifier). |
| **`age`** | `int64` | The age of the customer. | Range: 18 – 64 (Mean: 40.5). | `+0.010` (Negligible correlation). |
| **`tenure_months`** | `int64` | Number of months the customer has been subscribed. | Range: 1 – 60 (Mean: 33.2). | **`-0.197`** (Strong negative; longer tenure = lower likelihood of churn). |
| **`monthly_spend`** | `int64` | Monthly subscription fee paid by the customer. | Range: \$25 – \$139 (Mean: \$78.65). | `+0.130` (Moderate positive; higher spend = slightly higher churn). |
| **`num_logins`** | `int64` | Total times the customer logged in (during the review window). | Range: 5 – 94 (Mean: 49.37). | **`-0.171`** (Strong negative; higher engagement/logins = lower churn). |
| **`support_tickets`** | `int64` | Number of customer support requests filed. | Range: 0 – 7 (Mean: 3.36). | **`+0.338`** (Highest positive correlation; more support tickets = high frustration = high churn). |
| **`plan_type`** | `object` | Subscription tier of the user. | `basic` (94), `pro` (56), `enterprise` (30). | Categorical. |
| **`region`** | `object` | Geographical territory of the customer. | `prairies` (49), `pacific` (44), `atlantic` (44), `central` (43). | Categorical. |
| **`auto_pay`** | `int64` | Billing setup: `1` if auto-renewal is enabled; `0` otherwise. | Mean: 0.54 (54% of users use auto-pay). | `-0.105` (Moderate negative; auto-pay helps retain customers). |
| **`churned`** | `int64` | **Target label**: `1` if user canceled; `0` if active. | Binary: `0` (retained) or `1` (canceled). | **1.000** (Self). |

---

### **Summary Insights**
1. In the **LendingClub** dataset, borrower default is heavily driven by credit history attributes: **`fico`**, **`credit.policy`**, **`int.rate`**, and credit inquiries (**`inq.last.6mths`**). 
2. In the **Subscription Churn** dataset, churn behavior is mostly behavior-driven: users with high customer service friction (**`support_tickets`**), short tenure (**`tenure_months`**), and low interaction (**`num_logins`**) show a significantly higher risk of leaving the service.