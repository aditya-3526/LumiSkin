"""
LumiSkin CLV & RFM Segmentation — Component 6: Master Notebook Generator
==========================================================================
Generates the CMO-ready Jupyter notebook programmatically using nbformat.
This ensures reproducibility and avoids manual cell creation.

Output: notebooks/lumiskin_clv_rfm_analysis.ipynb
"""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
OUTPUT_FILE = os.path.join(NOTEBOOK_DIR, 'lumiskin_clv_rfm_analysis.ipynb')
os.makedirs(NOTEBOOK_DIR, exist_ok=True)


def build_notebook():
    nb = new_notebook()
    nb.metadata.kernelspec = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }

    cells = []

    # ── Title ──
    cells.append(new_markdown_cell(
"""# LumiSkin: Customer Lifetime Value & RFM Segmentation
### A Consulting-Grade Analytical Framework for D2C Retention Optimization

---

**Client:** LumiSkin (fictional D2C skincare brand, Brazil)  
**Problem:** R\\$180,000 annual retention budget deployed uniformly — every lapsed customer receives the same 15% discount, regardless of historical value, purchase frequency, or churn probability.  
**Objective:** Build a segmentation framework, CLV model, and retention ROI analysis that enables the CMO to reallocate budget from wasteful uniform spending to targeted, segment-differentiated interventions.

**Data Source:** Olist Brazilian E-Commerce Dataset (93,350 unique customers after filtering)  
**Reference Date:** 2018-08-29 (max order timestamp in dataset — used for reproducibility, never `datetime.now()`)

> ⚠️ **Critical Data Note:** This analysis uses `customer_unique_id` (stable person-level identifier), NOT `customer_id` (which is unique per order in the Olist schema). Failing to use the correct identifier would inflate the customer count by ~8%.
"""
    ))

    # ── Setup ──
    cells.append(new_markdown_cell("## 1. Environment Setup"))
    cells.append(new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
import warnings
warnings.filterwarnings('ignore')

# Paths
import os
PROJECT_ROOT = os.path.dirname(os.getcwd())
PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
CHARTS = os.path.join(PROJECT_ROOT, 'outputs', 'charts')

print("Environment ready.")
print(f"Project root: {PROJECT_ROOT}")
"""
    ))

    # ── Data Overview ──
    cells.append(new_markdown_cell(
"""## 2. Data Preparation & Base Table

The analytical base table was constructed via a **6-CTE SQL pipeline** (`sql/build_base_table.sql`) that:
1. Filters to delivered orders only (`order_status = 'delivered'`)
2. Resolves payment installment duplication (`SUM(payment_value) GROUP BY order_id`)
3. Aggregates to `customer_unique_id` level with modal category, review score, and state
4. Computes recency, tenure, and order-level metrics

**Result:** 93,350 unique customers with 17 analytical features each.
"""
    ))
    cells.append(new_code_cell(
"""# Load the base table
base = pd.read_csv(os.path.join(PROCESSED, 'customer_base.csv'))
print(f"Customer base: {len(base):,} rows × {len(base.columns)} columns")
print(f"\\nColumn summary:")
print(base.dtypes.to_string())
print(f"\\nKey statistics:")
print(f"  Single-purchase customers: {(base['total_orders'] == 1).sum():,} ({(base['total_orders'] == 1).mean():.1%})")
print(f"  Multi-purchase customers:  {(base['total_orders'] > 1).sum():,} ({(base['total_orders'] > 1).mean():.1%})")
print(f"  Median total spend:        R${base['total_spend'].median():,.2f}")
print(f"  Median recency:            {base['recency_days'].median():.0f} days")
"""
    ))

    # ── RFM ──
    cells.append(new_markdown_cell(
"""## 3. RFM Segmentation

### Scoring Methodology

| Dimension | Method | Rationale |
|-----------|--------|-----------|
| **Recency** | Business breakpoints (30/60/90/150 days) | Aligns with D2C retention windows |
| **Frequency** | Absolute counts (1/2/3/4/5+) | 97% of customers have 1 order — quintiles would be useless |
| **Monetary** | Quintiles on total_spend | Continuously distributed, no natural breakpoints |

### Weight Derivation

A logistic regression was fitted to derive data-driven weights for the RFM dimensions. **Dimensional collapse was detected** — M_score captured 86% of weight due to the extreme frequency skew. The model defaulted to **industry-standard 40/35/25 weights** (R/F/M), which produce a genuinely multi-dimensional composite score.

### Segment Assignment Rules
Segments are assigned via a rule-based engine (first-match wins):
1. **Champions:** Composite ≥ 4.0 AND R_score ≥ 4
2. **Loyal Customers:** Composite 3.0–3.9 AND F_score ≥ 3
3. **Potential Loyalists:** Composite 2.5–3.4 AND R_score ≥ 3 AND F_score ≤ 2
4. **At-Risk Customers:** Composite 2.0–2.9 AND R_score ≤ 2 AND (F_score ≥ 2 OR M_score ≥ 3)
5. **Lost Customers:** Composite < 2.0 OR (R_score = 1 AND F_score = 1)
"""
    ))
    cells.append(new_code_cell(
"""rfm = pd.read_csv(os.path.join(PROCESSED, 'rfm_segments.csv'))

# Segment distribution
seg_summary = rfm.groupby('rfm_segment').agg(
    customers=('customer_unique_id', 'count'),
    median_spend=('total_spend', 'median'),
    median_recency=('recency_days', 'median'),
    median_orders=('total_orders', 'median'),
).reindex(['Champions', 'Loyal Customers', 'Potential Loyalists',
           'At-Risk Customers', 'Lost Customers'])

seg_summary['pct'] = (seg_summary['customers'] / seg_summary['customers'].sum() * 100).round(1)
seg_summary['median_spend'] = seg_summary['median_spend'].apply(lambda x: f'R${x:,.2f}')

print("RFM Segment Distribution:")
print("=" * 70)
display(seg_summary[['customers', 'pct', 'median_spend', 'median_recency', 'median_orders']])
"""
    ))

    # Charts
    cells.append(new_code_cell(
"""# Segment Overview
display(Image(os.path.join(CHARTS, 'segment_overview.png'), width=700))
"""
    ))
    cells.append(new_code_cell(
"""# RFM Heatmap
display(Image(os.path.join(CHARTS, 'rfm_heatmap.png'), width=600))
"""
    ))

    # ── CLV ──
    cells.append(new_markdown_cell(
"""## 4. Customer Lifetime Value Modeling

### Methodology

**CLV Formula:**
$$CLV = AOV \\times Annualized\\ Frequency \\times Gross\\ Margin \\times Estimated\\ Lifespan$$

**Key decisions:**
- **BG/NBD model** was attempted for lifespan estimation but **failed to converge** due to the extreme frequency skew (97% of customers have frequency=0 in the BG/NBD framework). This is a known limitation documented in the academic literature for datasets with very low repeat-purchase rates.
- **Empirical fallback** was used: multi-purchase customers use actual tenure; single-purchase customers use segment-level imputed tenure (0.5-year floor).
- **Single-purchase frequency imputation** uses recency-adjusted conservative rates (1-2 purchases/year) rather than inflated segment medians.
- **Gross margin:** 65% (default for D2C skincare). Sensitivity analysis tests 50%, 60%, 65%, 70%.

> ⚠️ **Uncertainty Note:** 97% of CLV estimates rest on imputed frequency and tenure assumptions rather than observed repeat-purchase behavior. All CLV figures should be interpreted as directional estimates for budget allocation, not precise predictions.
"""
    ))
    cells.append(new_code_cell(
"""clv = pd.read_csv(os.path.join(PROCESSED, 'clv_scores.csv'))

# CLV summary by segment
clv_summary = clv.groupby('rfm_segment').agg(
    customers=('customer_unique_id', 'count'),
    median_clv=('clv_estimate', 'median'),
    mean_clv=('clv_estimate', 'mean'),
    total_clv=('clv_estimate', 'sum'),
).reindex(['Champions', 'Loyal Customers', 'Potential Loyalists',
           'At-Risk Customers', 'Lost Customers'])

clv_summary['clv_share'] = (clv_summary['total_clv'] / clv_summary['total_clv'].sum() * 100).round(1)

for col in ['median_clv', 'mean_clv', 'total_clv']:
    clv_summary[col] = clv_summary[col].apply(lambda x: f'R${x:,.2f}')

print(f"Total Portfolio CLV: R${clv['clv_estimate'].sum():,.2f}")
print(f"Mean CLV per Customer: R${clv['clv_estimate'].mean():,.2f}")
print()
display(clv_summary)
"""
    ))
    cells.append(new_code_cell(
"""# CLV Distribution
display(Image(os.path.join(CHARTS, 'clv_distribution.png'), width=700))
"""
    ))
    cells.append(new_code_cell(
"""# CLV Sensitivity
display(Image(os.path.join(CHARTS, 'clv_sensitivity.png'), width=650))
"""
    ))

    # ── Retention ROI ──
    cells.append(new_markdown_cell(
"""## 5. Retention ROI Analysis

### The Core Problem
LumiSkin currently deploys its R\\$180,000 retention budget **uniformly** — every customer who hasn't purchased in 60+ days gets the same 15% discount coupon. This means:
- **72% of budget (R\\$129K) goes to Lost Customers** — the segment with **negative ROI** (-4.5%)
- Only **R\\$45K reaches At-Risk Customers** — the segment with the highest recovery potential
- Champions and Loyal Customers receive almost nothing (R\\$174 combined)

### Proposed Approach: Budget-Share Allocation

| Segment | Budget Share | Intervention | Expected Impact |
|---------|-------------|-------------|-----------------|
| Champions (2%) | R\\$3,600 | Loyalty recognition (no discount) | Churn reduction: 8%→4% |
| Loyal Customers (15%) | R\\$27,000 | Referral incentive program | 15% frequency uplift |
| Potential Loyalists (35%) | R\\$63,000 | Day-45 nudge + 10% offer | 22% conversion to repeat |
| At-Risk Customers (48%) | R\\$86,400 | 3-email win-back sequence | 28% recovery rate |
| Lost Customers (0%) | R\\$0 | No spend — below ROI threshold | — |
"""
    ))
    cells.append(new_code_cell(
"""roi = pd.read_csv(os.path.join(PROCESSED, 'retention_roi.csv'))

# Summary comparison
current_rev = roi['current_revenue_impact'].sum()
proposed_rev = roi['proposed_revenue_impact'].sum()
delta = proposed_rev - current_rev
pct = delta / current_rev * 100

print("=" * 60)
print("HEADLINE FINDING")
print("=" * 60)
print(f"\\n  Current approach revenue:  R${current_rev:>12,.2f}")
print(f"  Proposed approach revenue: R${proposed_rev:>12,.2f}")
print(f"  Additional revenue:        R${delta:>12,.2f}")
print(f"  Improvement:               {pct:>11.0f}%")
print(f"\\n  Same R$180,000 budget → {pct:.0f}% more revenue protected.")

# Show per-segment comparison
display(roi[['rfm_segment', 'current_budget_allocation', 'proposed_budget_allocation',
             'current_revenue_impact', 'proposed_revenue_impact']].set_index('rfm_segment'))
"""
    ))
    cells.append(new_code_cell(
"""# Retention ROI Comparison Chart
display(Image(os.path.join(CHARTS, 'retention_roi_comparison.png'), width=800))
"""
    ))

    # ── Geographic ──
    cells.append(new_markdown_cell("## 6. Geographic Insights"))
    cells.append(new_code_cell(
"""# Geographic CLV
display(Image(os.path.join(CHARTS, 'geographic_clv.png'), width=700))
"""
    ))

    # ── Recommendations ──
    cells.append(new_markdown_cell(
"""## 7. Recommendations

### Priority 1: Redirect R$129K from Lost Customers to High-ROI Segments
**Impact:** Immediate — 72% of current budget is spent on negative-ROI customers.  
Reallocating to At-Risk (48%) and Potential Loyalists (35%) produces **+131% revenue uplift**.

### Priority 2: Implement CLV-Ranked Targeting Within Each Segment
**Impact:** 2-4 weeks — instead of reaching all customers uniformly, target the highest-CLV customers first within each segment's budget allocation. The At-Risk segment alone has 22,464 customers but budget covers only 3,600 — the top 16% by CLV.

### Priority 3: Build a Second-Purchase Conversion Engine
**Impact:** 3-6 months — 97% of customers are single-purchase. Converting even 5% of Potential Loyalists to repeat buyers would add ~574 customers to the Loyal/At-Risk segments, significantly expanding the addressable high-value base.

---

## 8. Limitations & Caveats

1. **97% single-purchase base:** The overwhelming majority of CLV estimates rely on imputed frequency and tenure assumptions. CLV figures are directional, not precise.
2. **BG/NBD non-convergence:** The probabilistic model could not fit this dataset. Empirical lifespan estimates are used as a conservative fallback.
3. **Intervention assumptions:** Recovery rates, conversion rates, and frequency uplift percentages are assumed based on industry benchmarks, not A/B test results. LumiSkin should validate these through controlled experiments.
4. **Cross-channel effects not modeled:** The analysis considers direct purchase behavior only. Referral value, social influence, and brand advocacy are not captured in CLV.
5. **Geographic analysis limited to state level:** Zip-code level geolocation data is available but not used in CLV modeling due to privacy and granularity concerns.
"""
    ))

    nb.cells = cells
    return nb


def main():
    print("="*70)
    print("COMPONENT 6: Master Notebook")
    print("="*70)

    nb = build_notebook()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"\n  ✓ Notebook saved to {OUTPUT_FILE}")
    print(f"  Total cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown, "
          f"{sum(1 for c in nb.cells if c.cell_type == 'code')} code)")
    
    print(f"\n{'='*70}")
    print("COMPONENT 6 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
