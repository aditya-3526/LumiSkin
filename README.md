# LumiSkin: Customer Lifetime Value & RFM Segmentation

> A consulting-grade analytical framework for D2C customer retention optimization, demonstrating end-to-end data analysis from raw transactional data to actionable budget reallocation.

---

## 📋 Executive Summary

**Client:** LumiSkin (fictional D2C skincare brand, Brazil)  
**Problem:** R$180,000 annual retention budget deployed uniformly — 72% wasted on Lost Customers with negative ROI  
**Solution:** RFM segmentation + CLV modeling → segment-differentiated retention strategy

### Key Finding

> **By reallocating the same R$180,000 budget from uniform 15% discounts to targeted, segment-specific interventions, LumiSkin can protect an additional R$360,000 in customer lifetime value — a 131% improvement in retention ROI.**

| Metric | Current | Proposed | Δ |
|--------|---------|----------|---|
| Total Budget | R$180,000 | R$180,000 | — |
| Revenue Protected | R$274,855 | R$634,663 | **+131%** |
| Portfolio ROI | 52.7% | 252.6% | **+200pp** |
| Lost Customer Spend | R$128,792 (72%) | R$0 (0%) | **-R$129K redirected** |

---

## 🏗️ Methodology

### Data Pipeline

```
Olist Raw Data (9 CSVs)
    ↓  6-CTE SQL Pipeline (SQLite)
Customer Base Table (93,350 customers × 17 features)
    ↓  Business-Breakpoint RFM Scoring
RFM Segments (5 segments, 40/35/25 weights)
    ↓  CLV Modeling (empirical lifespan + recency-adjusted frequency)
Individual CLV Estimates (R$6.4M total portfolio)
    ↓  Budget-Share Retention ROI Model
Segment-Differentiated Intervention Plan
    ↓  6 Publication-Quality Charts + Tableau Export
CMO-Ready Deliverables
```

### Analytical Components

| # | Component | Script | Key Output |
|---|-----------|--------|------------|
| 1 | Data Ingestion | `analysis/build_base_table.py` | `customer_base.csv` (93,350 rows) |
| 2 | RFM Scoring | `analysis/rfm_scoring.py` | `rfm_segments.csv` (5 segments) |
| 3 | CLV Modeling | `analysis/clv_model.py` | `clv_scores.csv` (R$6.4M portfolio) |
| 4 | Retention ROI | `analysis/retention_roi.py` | `retention_roi.csv` (+131% uplift) |
| 5 | Visualizations | `analysis/visualizations.py` | 6 charts at 300 DPI |
| 6 | Master Notebook | `analysis/generate_notebook.py` | Jupyter narrative |
| 7 | Tableau Export | `analysis/tableau_export.py` | 5 CSVs + build guide |

---

## 📊 Segment Breakdown

| Segment | Customers | % | Median CLV | CLV Share | Intervention |
|---------|-----------|---|------------|-----------|-------------|
| Champions | 27 | 0.03% | R$330 | 0.2% | Loyalty recognition (no discount) |
| Loyal Customers | 1,578 | 1.7% | R$213 | 7.8% | Referral incentive program |
| Potential Loyalists | 11,480 | 12.3% | R$82 | 20.0% | Day-45 second-purchase nudge |
| At-Risk Customers | 22,464 | 24.1% | R$82 | 45.0% | 3-email win-back sequence |
| Lost Customers | 57,801 | 61.9% | R$27 | 27.0% | No spend (below ROI threshold) |

---

## 🔑 Technical Decisions & Trade-offs

### RFM Weights
- Logistic regression attempted → **dimensional collapse detected** (M_score captured 86% of weight)
- Root cause: 97% single-purchase rate collapses the feature space
- Decision: **Industry-standard 40/35/25 (R/F/M)** — produces genuinely multi-dimensional composite scores

### CLV Model
- **BG/NBD model attempted → failed to converge** at all penalizer values (0.001 to 10.0)
- Root cause: 97% of customers have frequency=0 in BG/NBD framework (zero repeat purchases)
- Decision: **Empirical fallback** with conservative, recency-adjusted frequency imputation (1-2 purchases/year for single-purchase customers)

### Frequency Imputation
- Segment-median imputation rejected (produced 15-25 purchases/year — unrealistic for single-purchasers)
- Decision: Recency-adjusted base rates (R_score 4-5: 2/yr, R_score 3: 1.5/yr, R_score 1-2: 1/yr)

### Budget Model
- Per-customer cost × all customers approach rejected (exceeded R$180K budget 3.7×, required unrealistic rescaling)
- Decision: **Budget-share allocation** — fixed percentage of R$180K per segment, CLV-ranked targeting within each allocation

---

## 📁 Project Structure

```
lumiskin-clv-rfm/
├── PLAN.md                          # Master build plan & checklist
├── README.md                        # This file
├── sql/
│   └── build_base_table.sql         # 6-CTE analytical base table pipeline
├── analysis/
│   ├── build_base_table.py          # Component 1: Data ingestion
│   ├── rfm_scoring.py               # Component 2: RFM segmentation
│   ├── clv_model.py                 # Component 3: CLV modeling
│   ├── retention_roi.py             # Component 4: Retention ROI
│   ├── visualizations.py            # Component 5: Charts
│   ├── generate_notebook.py         # Component 6: Notebook generator
│   └── tableau_export.py            # Component 7: Tableau exports
├── data/
│   ├── raw/                         # Original Olist CSVs (9 files)
│   └── processed/                   # Analytical outputs
│       ├── customer_base.csv
│       ├── rfm_segments.csv
│       ├── clv_scores.csv
│       ├── clv_sensitivity.csv
│       └── retention_roi.csv
├── notebooks/
│   └── lumiskin_clv_rfm_analysis.ipynb
└── outputs/
    ├── charts/                      # 6 publication-quality PNGs
        ├── segment_overview.png
        ├── clv_distribution.png
        ├── rfm_heatmap.png
        ├── retention_roi_comparison.png
        ├── clv_sensitivity.png
        └── geographic_clv.png
    
```

---

## 🔧 Reproduction

### Prerequisites
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn lifetimes nbformat
```

### Run Sequence
```bash
# Component 1: Build analytical base table
python analysis/build_base_table.py

# Component 2: RFM scoring & segmentation
python analysis/rfm_scoring.py

# Component 3: CLV modeling
python analysis/clv_model.py

# Component 4: Retention ROI model
python analysis/retention_roi.py

# Component 5: Generate all 6 charts
python analysis/visualizations.py

# Component 6: Generate Jupyter notebook
python analysis/generate_notebook.py

# Component 7: Export Tableau package
python analysis/tableau_export.py
```

### Critical Data Notes
1. **Always use `customer_unique_id`** (not `customer_id`) — `customer_id` is unique per order, not per person
2. **Reference date:** `2018-08-29` (dataset max timestamp) — never `datetime.now()`
3. **Order filter:** Only `order_status = 'delivered'` with non-null delivery date
4. **Payment deduplication:** `SUM(payment_value) GROUP BY order_id` before joining

---

## 🛠 Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python 3** | Core analysis language |
| **pandas / NumPy** | Data manipulation and computation |
| **SQLite** | SQL pipeline execution |
| **scikit-learn** | Logistic regression for RFM weight derivation |
| **lifetimes** | BG/NBD model attempt (non-convergent on this dataset) |
| **matplotlib / seaborn** | Publication-quality visualizations |
| **Tableau** | Interactive dashboard (build guide provided) |

---

## ⚠️ Limitations

1. **97% single-purchase base** — CLV estimates for the majority of customers rely on imputed assumptions, not observed behavior
2. **BG/NBD non-convergence** — the probabilistic model could not fit this dataset; empirical lifespan estimates are conservative but less precise
3. **Intervention assumptions** — recovery rates and conversion rates are based on industry benchmarks, not A/B test results
4. **Cross-channel effects** — referral value, social influence, and brand advocacy are not captured in CLV
5. **Static snapshot** — analysis uses a fixed reference date; production deployment would require automated refresh

---

## 📄 Data Source

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — real anonymized commercial data from the Brazilian marketplace Olist, covering ~100K orders from 2016-2018.
