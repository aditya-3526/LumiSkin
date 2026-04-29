"""
LumiSkin CLV & RFM Segmentation — Component 7: Tableau Export Package
======================================================================
Exports five wide-format, self-contained flat CSVs and a build guide
for constructing the "LumiSkin Customer Intelligence Dashboard" in Tableau.

Outputs:
  outputs/tableau/01_customer_detail.csv      — Full customer-level detail
  outputs/tableau/02_segment_summary.csv      — Segment-level aggregations
  outputs/tableau/03_retention_comparison.csv  — Current vs proposed ROI
  outputs/tableau/04_clv_sensitivity.csv      — Margin sensitivity matrix
  outputs/tableau/05_geographic_summary.csv   — State-level CLV with seller geography
  outputs/tableau/TABLEAU_BUILD_GUIDE.md      — Sheet-by-sheet build instructions
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
TABLEAU_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'tableau')
os.makedirs(TABLEAU_DIR, exist_ok=True)

SEGMENT_ORDER = ['Champions', 'Loyal Customers', 'Potential Loyalists',
                 'At-Risk Customers', 'Lost Customers']


def export_customer_detail():
    """Export 01: Full customer-level detail for Tableau scatter/detail views."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'clv_scores.csv'))
    
    # Select and rename columns for Tableau-friendly headers
    export = df[[
        'customer_unique_id', 'customer_state', 'rfm_segment',
        'R_score', 'F_score', 'M_score', 'rfm_composite',
        'total_orders', 'total_spend', 'avg_order_value',
        'recency_days', 'customer_tenure_days',
        'clv_estimate', 'annualized_frequency', 'estimated_lifespan_years',
        'first_purchase_date', 'last_purchase_date',
        'avg_review_score', 'primary_category', 'primary_seller_state',
    ]].copy()
    
    # Add segment sort order for Tableau
    seg_order_map = {s: i+1 for i, s in enumerate(SEGMENT_ORDER)}
    export['segment_sort_order'] = export['rfm_segment'].map(seg_order_map)
    
    path = os.path.join(TABLEAU_DIR, '01_customer_detail.csv')
    export.to_csv(path, index=False)
    print(f"  ✓ 01_customer_detail.csv ({len(export):,} rows × {len(export.columns)} cols)")
    return path


def export_segment_summary():
    """Export 02: Segment-level aggregations for Tableau summary cards/bars."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'clv_scores.csv'))
    
    summary = df.groupby('rfm_segment').agg(
        customer_count=('customer_unique_id', 'count'),
        total_clv=('clv_estimate', 'sum'),
        median_clv=('clv_estimate', 'median'),
        mean_clv=('clv_estimate', 'mean'),
        p25_clv=('clv_estimate', lambda x: x.quantile(0.25)),
        p75_clv=('clv_estimate', lambda x: x.quantile(0.75)),
        total_spend=('total_spend', 'sum'),
        median_spend=('total_spend', 'median'),
        median_recency=('recency_days', 'median'),
        median_orders=('total_orders', 'median'),
        median_aov=('avg_order_value', 'median'),
        median_r_score=('R_score', 'median'),
        median_f_score=('F_score', 'median'),
        median_m_score=('M_score', 'median'),
        median_composite=('rfm_composite', 'median'),
    ).round(2)
    
    total = summary['customer_count'].sum()
    summary['customer_pct'] = (summary['customer_count'] / total * 100).round(1)
    summary['clv_share_pct'] = (summary['total_clv'] / summary['total_clv'].sum() * 100).round(1)
    
    seg_order_map = {s: i+1 for i, s in enumerate(SEGMENT_ORDER)}
    summary['segment_sort_order'] = summary.index.map(seg_order_map)
    
    summary = summary.reindex(SEGMENT_ORDER)
    
    path = os.path.join(TABLEAU_DIR, '02_segment_summary.csv')
    summary.to_csv(path)
    print(f"  ✓ 02_segment_summary.csv ({len(summary)} rows × {len(summary.columns)} cols)")
    return path


def export_retention_comparison():
    """Export 03: Current vs proposed retention ROI for Tableau dual-bar/comparison."""
    roi = pd.read_csv(os.path.join(PROCESSED_DIR, 'retention_roi.csv'))
    
    # Select key comparison columns
    export = roi[[col for col in roi.columns if col != 'Unnamed: 0']].copy()
    
    seg_order_map = {s: i+1 for i, s in enumerate(SEGMENT_ORDER)}
    export['segment_sort_order'] = export['rfm_segment'].map(seg_order_map)
    export = export.sort_values('segment_sort_order')
    
    # Add computed fields for Tableau
    export['budget_shift'] = export['proposed_budget_allocation'] - export['current_budget_allocation']
    export['revenue_shift'] = export['proposed_revenue_impact'] - export['current_revenue_impact']
    export['roi_improvement'] = export['roi_proposed'] - export['roi_current']
    
    path = os.path.join(TABLEAU_DIR, '03_retention_comparison.csv')
    export.to_csv(path, index=False)
    print(f"  ✓ 03_retention_comparison.csv ({len(export)} rows × {len(export.columns)} cols)")
    return path


def export_sensitivity():
    """Export 04: CLV sensitivity matrix for Tableau multi-line chart."""
    sens = pd.read_csv(os.path.join(PROCESSED_DIR, 'clv_sensitivity.csv'))
    
    seg_order_map = {s: i+1 for i, s in enumerate(SEGMENT_ORDER)}
    sens['segment_sort_order'] = sens['rfm_segment'].map(seg_order_map)
    sens['margin_pct'] = (sens['gross_margin_assumption'] * 100).astype(int)
    
    path = os.path.join(TABLEAU_DIR, '04_clv_sensitivity.csv')
    sens.to_csv(path, index=False)
    print(f"  ✓ 04_clv_sensitivity.csv ({len(sens)} rows × {len(sens.columns)} cols)")
    return path


def export_geographic_summary():
    """Export 05: State-level CLV with seller geography for Tableau geo map."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'clv_scores.csv'))
    
    # Customer state aggregation
    geo = df.groupby('customer_state').agg(
        customer_count=('customer_unique_id', 'count'),
        total_clv=('clv_estimate', 'sum'),
        median_clv=('clv_estimate', 'median'),
        mean_clv=('clv_estimate', 'mean'),
        total_spend=('total_spend', 'sum'),
        median_orders=('total_orders', 'median'),
        median_recency=('recency_days', 'median'),
    ).round(2)
    
    # Add seller state enrichment — cross-state purchasing patterns
    if 'primary_seller_state' in df.columns:
        # For each customer state, find the most common seller state
        seller_mode = df.groupby('customer_state')['primary_seller_state'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        )
        geo['top_seller_state'] = seller_mode
        
        # Compute % of customers buying from same state (local vs distant)
        local_pct = df.groupby('customer_state').apply(
            lambda g: (g['customer_state'] == g['primary_seller_state']).mean() * 100
        ).round(1)
        geo['local_seller_pct'] = local_pct
    
    # Add CLV share
    geo['clv_share_pct'] = (geo['total_clv'] / geo['total_clv'].sum() * 100).round(1)
    geo = geo.sort_values('total_clv', ascending=False)
    
    path = os.path.join(TABLEAU_DIR, '05_geographic_summary.csv')
    geo.to_csv(path)
    print(f"  ✓ 05_geographic_summary.csv ({len(geo)} rows × {len(geo.columns)} cols)")
    return path


def write_build_guide():
    """Write the Tableau dashboard build guide."""
    guide = """# LumiSkin Customer Intelligence Dashboard — Tableau Build Guide

## Dashboard Overview

**Title:** LumiSkin Customer Intelligence Dashboard  
**Size:** 1200 × 900 px (fixed)  
**Sheets:** 5  
**Filter Interaction:** Cross-sheet filter on `rfm_segment`

---

## Color Palette

| Segment | Hex Code | Usage |
|---------|----------|-------|
| Champions | `#2E86AB` | Steel blue |
| Loyal Customers | `#A23B72` | Muted magenta |
| Potential Loyalists | `#F18F01` | Amber |
| At-Risk Customers | `#C73E1D` | Terra cotta |
| Lost Customers | `#6B717E` | Slate grey |

**Assign colors:** Right-click any segment legend → Edit Colors → Assign manually using hex codes above.

---

## Sheet 1: Segment Scorecard (Summary Cards)

**Data Source:** `02_segment_summary.csv`  
**Chart Type:** Text table / KPI cards  
**Layout:** Horizontal row of 5 cards, one per segment

| Field | Shelf |
|-------|-------|
| `rfm_segment` | Columns (sorted by `segment_sort_order`) |
| `customer_count` | Text (formatted as "#,##0") |
| `median_clv` | Text (formatted as "R$#,##0") |
| `clv_share_pct` | Text (formatted as "0.0%") |

**Calculated Field — CLV per Customer:**
```
[total_clv] / [customer_count]
```

---

## Sheet 2: CLV Distribution (Box Plot)

**Data Source:** `01_customer_detail.csv`  
**Chart Type:** Box-and-whisker plot  

| Field | Shelf |
|-------|-------|
| `rfm_segment` | Columns (sorted by `segment_sort_order`) |
| `clv_estimate` | Rows (log axis) |
| `rfm_segment` | Color (use palette above) |

**Settings:** Reference line → Median per pane

---

## Sheet 3: Retention ROI Comparison (Dual Bar)

**Data Source:** `03_retention_comparison.csv`  
**Chart Type:** Side-by-side bar chart  

| Field | Shelf |
|-------|-------|
| `rfm_segment` | Rows (sorted by `segment_sort_order`) |
| `current_budget_allocation` | Columns (bar 1, grey #B0B0B0) |
| `proposed_budget_allocation` | Columns (bar 2, segment color) |

**Second axis:** Drag `revenue_shift` to right axis → Dual axis → Line mark

**Calculated Field — Revenue Uplift %:**
```
([proposed_revenue_impact] - [current_revenue_impact]) / [current_revenue_impact]
```

---

## Sheet 4: CLV Sensitivity (Multi-Line)

**Data Source:** `04_clv_sensitivity.csv`  
**Chart Type:** Line chart  

| Field | Shelf |
|-------|-------|
| `margin_pct` | Columns (continuous) |
| `median_clv` | Rows |
| `rfm_segment` | Color (use palette above) |

**Axis:** X-axis label = "Gross Margin (%)", Y-axis = "Median CLV (R$)"

---

## Sheet 5: Geographic CLV Map

**Data Source:** `05_geographic_summary.csv`  
**Chart Type:** Filled map (Brazil states) OR horizontal bar chart  

**Option A — Map:**
| Field | Shelf |
|-------|-------|
| `customer_state` | Geography (State) → Detail |
| `total_clv` | Color (sequential orange) |
| `customer_count` | Label |

**Option B — Bar Chart (recommended if map geocoding issues):**
| Field | Shelf |
|-------|-------|
| `customer_state` | Rows (sorted by `total_clv` descending, top 10) |
| `total_clv` | Columns |
| `customer_count` | Label |

**Tooltip:** Include `local_seller_pct` to show % buying from local sellers.

---

## Dashboard Assembly

1. Create new Dashboard → Fixed size 1200 × 900
2. **Top row:** Sheet 1 (Scorecard) — full width, ~150px height
3. **Middle left:** Sheet 2 (CLV Distribution) — ~600 × 350px
4. **Middle right:** Sheet 3 (Retention ROI) — ~600 × 350px
5. **Bottom left:** Sheet 4 (Sensitivity) — ~600 × 350px
6. **Bottom right:** Sheet 5 (Geographic) — ~600 × 350px
7. **Filter:** Add `rfm_segment` as a dashboard filter → Apply to All Sheets
8. **Title:** "LumiSkin Customer Intelligence Dashboard" — 16pt, bold

---

## Cross-Filter Setup

1. Select Sheet 1 (Scorecard) on the dashboard
2. Click the filter icon (funnel) on the sheet dropdown
3. Select "Use as Filter"
4. Clicking a segment card now filters all other sheets

---

## Final Polish

- [ ] Remove all sheet titles (use dashboard title only)
- [ ] Set tooltip format to include R$ currency and comma separators
- [ ] Add source annotation: "Data: Olist Brazilian E-Commerce | Analysis: LumiSkin CLV & RFM Framework"
- [ ] Export as `.twbx` packaged workbook for portability
"""
    
    path = os.path.join(TABLEAU_DIR, 'TABLEAU_BUILD_GUIDE.md')
    with open(path, 'w') as f:
        f.write(guide)
    print(f"  ✓ TABLEAU_BUILD_GUIDE.md")
    return path


def main():
    print("="*70)
    print("COMPONENT 7: Tableau Export Package")
    print("="*70)
    
    print("\n  Exporting Tableau-ready CSVs...\n")
    
    export_customer_detail()
    export_segment_summary()
    export_retention_comparison()
    export_sensitivity()
    export_geographic_summary()
    
    print("\n  Writing build guide...\n")
    write_build_guide()
    
    print(f"\n{'='*70}")
    print("COMPONENT 7 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
