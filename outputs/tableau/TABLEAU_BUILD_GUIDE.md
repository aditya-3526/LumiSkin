# LumiSkin Customer Intelligence Dashboard — Tableau Build Guide

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
