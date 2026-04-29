"""
LumiSkin CLV & RFM Segmentation — Component 5: Visualizations
===============================================================
Six publication-quality charts with insight annotations.

Style: Clean white background, muted professional palette, clear titles/labels,
one-line insight annotation per chart. No default matplotlib colors.

Inputs:  data/processed/clv_scores.csv
         data/processed/clv_sensitivity.csv
         data/processed/retention_roi.csv
Outputs: outputs/charts/ (6 PNGs at 300 DPI)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
CHARTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Professional color palette — muted, corporate-friendly
SEGMENT_COLORS = {
    'Champions':          '#2E86AB',  # Steel blue
    'Loyal Customers':    '#A23B72',  # Muted magenta
    'Potential Loyalists':'#F18F01',  # Amber
    'At-Risk Customers':  '#C73E1D',  # Terra cotta
    'Lost Customers':     '#6B717E',  # Slate grey
}

SEGMENT_ORDER = ['Champions', 'Loyal Customers', 'Potential Loyalists',
                 'At-Risk Customers', 'Lost Customers']

# Global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})


def load_all_data():
    """Load all processed datasets."""
    clv = pd.read_csv(os.path.join(PROCESSED_DIR, 'clv_scores.csv'))
    sensitivity = pd.read_csv(os.path.join(PROCESSED_DIR, 'clv_sensitivity.csv'))
    roi = pd.read_csv(os.path.join(PROCESSED_DIR, 'retention_roi.csv'))
    return clv, sensitivity, roi


# =========================================================================
# Chart 1: Segment Overview — Bubble Chart
# =========================================================================
def chart_segment_overview(df):
    """
    Bubble chart: X=median recency, Y=median CLV, size=customer count.
    Shows CLV concentration by segment with customer volume as bubble size.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for segment in SEGMENT_ORDER:
        seg = df[df['rfm_segment'] == segment]
        x = seg['recency_days'].median()
        y = seg['clv_estimate'].median()
        size = len(seg) / 100  # Scale for visibility
        
        ax.scatter(x, y, s=size, alpha=0.7, color=SEGMENT_COLORS[segment],
                   edgecolors='white', linewidth=1.5, zorder=3)
        
        # Label positioning — offset to avoid overlap
        offset_y = y * 0.12 if segment != 'Lost Customers' else y * 0.15
        ax.annotate(f"{segment}\n({len(seg):,})", (x, y),
                    textcoords='offset points', xytext=(0, 15),
                    ha='center', fontsize=9, fontweight='bold',
                    color=SEGMENT_COLORS[segment])
    
    ax.set_xlabel('Median Recency (days since last purchase)')
    ax.set_ylabel('Median CLV (R$)')
    ax.set_title('Customer Segment Landscape\nCLV vs. Recency by Segment')
    
    # Insight annotation
    ax.annotate('← Recent, high-value        Lapsed, low-value →',
                xy=(0.5, -0.12), xycoords='axes fraction',
                ha='center', fontsize=9, fontstyle='italic', color='#888888')
    
    ax.grid(True, alpha=0.2, linestyle='--')
    
    path = os.path.join(CHARTS_DIR, 'segment_overview.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ segment_overview.png")
    return path


# =========================================================================
# Chart 2: CLV Distribution — Box Plot (Log Scale)
# =========================================================================
def chart_clv_distribution(df):
    """
    Box plot showing within-segment CLV variance on log scale.
    Highlights the range and outliers within each segment.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to CLV > 0 for log scale
    plot_df = df[df['clv_estimate'] > 0].copy()
    
    # Create ordered categorical for proper segment ordering
    plot_df['rfm_segment'] = pd.Categorical(
        plot_df['rfm_segment'], categories=SEGMENT_ORDER, ordered=True
    )
    
    colors = [SEGMENT_COLORS[s] for s in SEGMENT_ORDER]
    
    bp = ax.boxplot(
        [plot_df[plot_df['rfm_segment'] == s]['clv_estimate'].values for s in SEGMENT_ORDER],
        labels=SEGMENT_ORDER, patch_artist=True, widths=0.6,
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='.', markersize=2, alpha=0.3),
        whiskerprops=dict(color='#888888'),
        capprops=dict(color='#888888'),
    )
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_yscale('log')
    ax.set_ylabel('CLV (R$, log scale)')
    ax.set_title('CLV Distribution by Segment')
    ax.tick_params(axis='x', rotation=20)
    
    # Add median labels
    for i, segment in enumerate(SEGMENT_ORDER):
        med = plot_df[plot_df['rfm_segment'] == segment]['clv_estimate'].median()
        ax.annotate(f'R${med:,.0f}', (i + 1, med),
                    textcoords='offset points', xytext=(25, 0),
                    fontsize=9, fontweight='bold', color=colors[i])
    
    ax.grid(True, axis='y', alpha=0.2, linestyle='--')
    
    # Insight
    ax.annotate('Champions show 10× higher median CLV than Lost Customers',
                xy=(0.5, -0.18), xycoords='axes fraction',
                ha='center', fontsize=9, fontstyle='italic', color='#888888')
    
    path = os.path.join(CHARTS_DIR, 'clv_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ clv_distribution.png")
    return path


# =========================================================================
# Chart 3: RFM Heatmap — R×F Matrix
# =========================================================================
def chart_rfm_heatmap(df):
    """
    Matrix heatmap showing customer count at each R×F score intersection.
    Reveals the diagonal R-F correlation pattern.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create R×F cross-tabulation
    heatmap_data = pd.crosstab(df['R_score'], df['F_score'])
    
    # Ensure all scores 1-5 are present
    for s in range(1, 6):
        if s not in heatmap_data.index:
            heatmap_data.loc[s] = 0
        if s not in heatmap_data.columns:
            heatmap_data[s] = 0
    heatmap_data = heatmap_data.sort_index(ascending=False).reindex(columns=sorted(heatmap_data.columns))
    
    # Use log scale for better color differentiation (counts span 0 to 60K)
    log_data = np.log10(heatmap_data.replace(0, np.nan))
    
    sns.heatmap(log_data, annot=heatmap_data.values, fmt=',d',
                cmap='YlOrRd', ax=ax, linewidths=1, linecolor='white',
                cbar_kws={'label': 'Customer Count (log₁₀ scale)'},
                annot_kws={'size': 9})
    
    ax.set_xlabel('Frequency Score')
    ax.set_ylabel('Recency Score')
    ax.set_title('RFM Heatmap: Customer Count by R × F Score')
    
    # Insight
    ax.annotate('97% of customers cluster at F=1 — the core segmentation challenge',
                xy=(0.5, -0.12), xycoords='axes fraction',
                ha='center', fontsize=9, fontstyle='italic', color='#888888')
    
    path = os.path.join(CHARTS_DIR, 'rfm_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ rfm_heatmap.png")
    return path


# =========================================================================
# Chart 4: Retention ROI Comparison — Dual Bar
# =========================================================================
def chart_retention_roi(roi_df):
    """
    Dual bar chart comparing current vs proposed budget allocation
    and revenue impact per segment.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ensure segment order
    roi_ordered = roi_df.set_index('rfm_segment').reindex(SEGMENT_ORDER)
    
    x = np.arange(len(SEGMENT_ORDER))
    width = 0.35
    colors_list = [SEGMENT_COLORS[s] for s in SEGMENT_ORDER]
    
    # Panel 1: Budget Allocation
    bars1 = ax1.bar(x - width/2, roi_ordered['current_budget_allocation'] / 1000,
                    width, label='Current (uniform)', color='#B0B0B0', edgecolor='white')
    bars2 = ax1.bar(x + width/2, roi_ordered['proposed_budget_allocation'] / 1000,
                    width, label='Proposed (segmented)', color=colors_list,
                    edgecolor='white')
    
    ax1.set_ylabel('Budget Allocation (R$ thousands)')
    ax1.set_title('Budget Allocation: Current vs Proposed')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace(' ', '\n') for s in SEGMENT_ORDER], fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', alpha=0.2, linestyle='--')
    
    # Panel 2: Revenue Impact
    bars3 = ax2.bar(x - width/2, roi_ordered['current_revenue_impact'] / 1000,
                    width, label='Current', color='#B0B0B0', edgecolor='white')
    bars4 = ax2.bar(x + width/2, roi_ordered['proposed_revenue_impact'] / 1000,
                    width, label='Proposed', color=colors_list, edgecolor='white')
    
    ax2.set_ylabel('Revenue Protected (R$ thousands)')
    ax2.set_title('Revenue Impact: Current vs Proposed')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace(' ', '\n') for s in SEGMENT_ORDER], fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.2, linestyle='--')
    
    fig.suptitle('Retention Budget Reallocation: +131% Revenue Uplift at Same R$180K Spend',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    path = os.path.join(CHARTS_DIR, 'retention_roi_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ retention_roi_comparison.png")
    return path


# =========================================================================
# Chart 5: CLV Sensitivity — Multi-Line
# =========================================================================
def chart_clv_sensitivity(sensitivity_df):
    """
    Multi-line chart showing how median CLV changes across gross margin
    assumptions for each segment.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for segment in SEGMENT_ORDER:
        seg_data = sensitivity_df[sensitivity_df['rfm_segment'] == segment]
        ax.plot(seg_data['gross_margin_assumption'] * 100,
                seg_data['median_clv'],
                marker='o', linewidth=2, markersize=6,
                color=SEGMENT_COLORS[segment], label=segment)
    
    ax.set_xlabel('Gross Margin Assumption (%)')
    ax.set_ylabel('Median CLV (R$)')
    ax.set_title('CLV Sensitivity to Gross Margin Assumptions')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Insight
    ax.annotate('Champions are most sensitive: R$254→R$355 across margin range (40% swing)',
                xy=(0.5, -0.12), xycoords='axes fraction',
                ha='center', fontsize=9, fontstyle='italic', color='#888888')
    
    path = os.path.join(CHARTS_DIR, 'clv_sensitivity.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ clv_sensitivity.png")
    return path


# =========================================================================
# Chart 6: Geographic CLV — Horizontal Bar
# =========================================================================
def chart_geographic_clv(df):
    """
    Horizontal bar chart showing top 10 states by total CLV.
    State-level CLV concentration insight.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate CLV by state
    state_clv = df.groupby('customer_state').agg(
        total_clv=('clv_estimate', 'sum'),
        customer_count=('customer_unique_id', 'count'),
        median_clv=('clv_estimate', 'median'),
    ).sort_values('total_clv', ascending=False).head(10)
    
    # Reverse for horizontal bar (top at top)
    state_clv = state_clv.iloc[::-1]
    
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.8, len(state_clv)))[::-1]
    
    bars = ax.barh(state_clv.index, state_clv['total_clv'] / 1000,
                   color=colors, edgecolor='white', height=0.6)
    
    # Add customer count labels
    for i, (state, row) in enumerate(state_clv.iterrows()):
        ax.annotate(f"  {row['customer_count']:,.0f} customers",
                    (row['total_clv'] / 1000, i),
                    va='center', fontsize=8, color='#666666')
    
    ax.set_xlabel('Total CLV (R$ thousands)')
    ax.set_title('Top 10 States by Total Customer Lifetime Value')
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    
    # Compute concentration stat for insight
    top3_share = state_clv['total_clv'].tail(3).sum() / df['clv_estimate'].sum() * 100
    ax.annotate(f'Top 3 states (SP, RJ, MG) account for ~{top3_share:.0f}% of total portfolio CLV',
                xy=(0.5, -0.10), xycoords='axes fraction',
                ha='center', fontsize=9, fontstyle='italic', color='#888888')
    
    path = os.path.join(CHARTS_DIR, 'geographic_clv.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ geographic_clv.png")
    return path


# =========================================================================
# Main
# =========================================================================
def main():
    print("="*70)
    print("COMPONENT 5: Visualizations")
    print("="*70)
    
    print("\n  Loading data...")
    clv_df, sensitivity_df, roi_df = load_all_data()
    print(f"  ✓ CLV data: {len(clv_df):,} rows")
    print(f"  ✓ Sensitivity data: {len(sensitivity_df)} rows")
    print(f"  ✓ ROI data: {len(roi_df)} rows")
    
    print(f"\n  Generating charts to {CHARTS_DIR}/\n")
    
    chart_segment_overview(clv_df)
    chart_clv_distribution(clv_df)
    chart_rfm_heatmap(clv_df)
    chart_retention_roi(roi_df)
    chart_clv_sensitivity(sensitivity_df)
    chart_geographic_clv(clv_df)
    
    print(f"\n  All 6 charts saved at 300 DPI.")
    
    print(f"\n{'='*70}")
    print("COMPONENT 5 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
