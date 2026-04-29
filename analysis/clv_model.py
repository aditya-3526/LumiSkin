"""
LumiSkin CLV & RFM Segmentation — Component 3: CLV Modeling
=============================================================
This script computes individual-level Customer Lifetime Value (CLV) for each
customer using the formula:

    CLV = avg_order_value × annualized_purchase_frequency × gross_margin × estimated_lifespan_years

Key methodological decisions:
  1. BG/NBD (Beta-Geometric/Negative Binomial Distribution) model for lifespan
     estimation — the academically appropriate probabilistic model for non-contractual
     customer relationships.
  2. Single-purchase frequency imputation using segment-level medians.
  3. Sensitivity analysis across four gross margin assumptions.

CRITICAL DATA NOTE:
  All analysis uses customer_unique_id (stable person-level identifier),
  NOT customer_id (which is unique per order in the Olist schema).

Inputs:  data/processed/rfm_segments.csv
Outputs: data/processed/clv_scores.csv
         data/processed/clv_sensitivity.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Suppress deprecation warnings from lifetimes (uses older numpy/scipy APIs)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
INPUT_FILE = os.path.join(PROCESSED_DIR, 'rfm_segments.csv')
OUTPUT_CLV = os.path.join(PROCESSED_DIR, 'clv_scores.csv')
OUTPUT_SENSITIVITY = os.path.join(PROCESSED_DIR, 'clv_sensitivity.csv')

# GROSS MARGIN ASSUMPTION
# Default: 0.65 (65%) — a reasonable assumption for D2C skincare brands.
# Skincare products typically have high gross margins (60-75%) due to low
# ingredient costs relative to branding, packaging, and perceived value.
# The sensitivity analysis below tests 0.50, 0.60, 0.65, and 0.70 to show
# how CLV estimates change under different margin assumptions.
DEFAULT_GROSS_MARGIN = 0.65
SENSITIVITY_MARGINS = [0.50, 0.60, 0.65, 0.70]

# CLV prediction horizon (years) for BG/NBD future purchase predictions
PREDICTION_HORIZON_YEARS = 2
PREDICTION_HORIZON_DAYS = PREDICTION_HORIZON_YEARS * 365

# Flag to control whether BG/NBD or empirical fallback is used
USE_BGNBD = True  # Set to False to force empirical fallback


def load_data(filepath: str) -> pd.DataFrame:
    """Load RFM-segmented customer data and validate."""
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found. Run Component 2 first.")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    required = ['customer_unique_id', 'total_orders', 'total_spend',
                'avg_order_value', 'recency_days', 'customer_tenure_days',
                'rfm_segment']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} customers from rfm_segments.csv")
    return df


def fit_bgnbd_model(df: pd.DataFrame) -> tuple:
    """
    Fit a BG/NBD (Beta-Geometric/Negative Binomial Distribution) model.
    
    WHAT IS BG/NBD AND WHY USE IT:
    The BG/NBD model (Fader, Hardie, & Lee 2005) is the standard probabilistic
    model for customer purchase behavior in non-contractual settings — i.e.,
    businesses where customers can leave at any time without explicit notice
    (like e-commerce, unlike subscriptions).
    
    The model makes two key assumptions:
      1. While "alive" (active), a customer's purchases follow a Poisson process
         with rate λ (purchase frequency varies across customers via a Gamma prior).
      2. After each purchase, there is a probability p that the customer "dies"
         (becomes permanently inactive), with p varying across customers via a
         Beta prior.
    
    WHY SIMPLER APPROACHES ARE INSUFFICIENT:
    Simpler approaches (e.g., "average purchases per year × years") violate the
    key insight that customer "death" is unobserved in non-contractual settings.
    A customer who hasn't purchased in 6 months might be dead (churned permanently)
    or just in a long inter-purchase interval. The BG/NBD model explicitly
    separates these two states and produces calibrated probability estimates
    for each customer being alive, which directly feeds into lifespan estimation.
    
    The lifetimes library requires these inputs per customer:
      - frequency: number of REPEAT purchases (total_orders - 1, NOT total_orders)
      - recency: time between first and last purchase (customer_tenure_days)
      - T: time between first purchase and the analysis end date
         (recency_days + customer_tenure_days, i.e., total observation period)
    
    Returns: (bgf model, predicted_purchases DataFrame) or (None, None) on failure.
    """
    try:
        from lifetimes import BetaGeoFitter
    except ImportError:
        print("  ⚠️  lifetimes library not available — using empirical fallback")
        return None, None
    
    print("  Fitting BG/NBD model...")
    
    # Prepare the data in the format lifetimes expects
    # CRITICAL: frequency = REPEAT purchases = total_orders - 1
    # A customer with 1 order has frequency 0 in the BG/NBD framework
    summary = pd.DataFrame({
        'customer_unique_id': df['customer_unique_id'],
        'frequency': df['total_orders'] - 1,  # Repeat purchases only
        'recency': df['customer_tenure_days'],  # Time between first and last purchase
        'T': df['recency_days'] + df['customer_tenure_days'],  # Total observation period
    })
    
    # Filter out customers where T = 0 (edge case: purchased on the last day)
    # These can't be modeled because the observation window has zero length
    valid_mask = summary['T'] > 0
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"  ⚠️  {invalid_count:,} customers with T=0 (purchased on last day) — "
              f"excluded from BG/NBD fitting, will use segment median")
    
    summary_valid = summary[valid_mask].copy()
    
    # Fit the BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.001)  # Light regularization for stability
    try:
        bgf.fit(
            summary_valid['frequency'],
            summary_valid['recency'],
            summary_valid['T']
        )
        print(f"  ✓ BG/NBD model fitted successfully")
        print(f"    Model parameters: r={bgf.params_['r']:.4f}, "
              f"alpha={bgf.params_['alpha']:.4f}, "
              f"a={bgf.params_['a']:.4f}, "
              f"b={bgf.params_['b']:.4f}")
    except Exception as e:
        print(f"  ⚠️  BG/NBD fitting failed: {e}")
        return None, None
    
    # Predict expected purchases over the prediction horizon for each customer
    predicted = bgf.conditional_expected_number_of_purchases_up_to_time(
        PREDICTION_HORIZON_DAYS,
        summary_valid['frequency'],
        summary_valid['recency'],
        summary_valid['T']
    )
    
    # Create a full predictions series (including T=0 customers as NaN)
    pred_series = pd.Series(np.nan, index=df.index)
    pred_series.loc[valid_mask] = predicted.values
    
    # Compute probability alive for each customer
    prob_alive = pd.Series(np.nan, index=df.index)
    alive_vals = bgf.conditional_probability_alive(
        summary_valid['frequency'],
        summary_valid['recency'],
        summary_valid['T']
    )
    prob_alive.loc[valid_mask] = alive_vals
    
    print(f"\n  BG/NBD Predictions ({PREDICTION_HORIZON_YEARS}-year horizon):")
    print(f"    Median predicted future purchases: {pred_series.median():.2f}")
    print(f"    Mean predicted future purchases:   {pred_series.mean():.2f}")
    print(f"    Max predicted future purchases:    {pred_series.max():.2f}")
    print(f"    Median P(alive):                   {prob_alive.median():.2%}")
    
    return bgf, pred_series, prob_alive


def compute_empirical_lifespan(df: pd.DataFrame) -> pd.Series:
    """
    EMPIRICAL FALLBACK: Estimate lifespan using median tenure by frequency cohort.
    
    This is used when the lifetimes library is unavailable or BG/NBD fitting fails.
    
    Approach: Group customers by their order count, compute median tenure for each
    group, and use that as the lifespan estimate (converted to years). For single-
    purchase customers, use the overall median of multi-purchase customers.
    
    LIMITATION: This approach does not account for the probability of a customer
    being "dead" vs merely in a long inter-purchase interval. It systematically
    overestimates lifespan for inactive customers and underestimates it for
    active ones. The BG/NBD model is strongly preferred.
    """
    print("  Computing empirical lifespan estimates (fallback method)...")
    
    # Multi-purchase customers: use actual tenure
    multi = df[df['total_orders'] > 1].copy()
    
    if len(multi) == 0:
        print("  ⚠️  No multi-purchase customers found — using 1-year default lifespan")
        return pd.Series(1.0, index=df.index)
    
    # Compute median tenure by frequency cohort
    cohort_tenure = multi.groupby('total_orders')['customer_tenure_days'].median()
    print(f"\n  Median tenure by frequency cohort:")
    for orders, tenure in cohort_tenure.items():
        print(f"    {orders} orders: {tenure:.0f} days ({tenure/365:.1f} years)")
    
    # For multi-purchase: tenure / 365, with a floor of 0.5 years
    lifespan = pd.Series(np.nan, index=df.index)
    multi_mask = df['total_orders'] > 1
    lifespan[multi_mask] = np.maximum(df.loc[multi_mask, 'customer_tenure_days'] / 365, 0.5)
    
    # For single-purchase: use segment-level median of multi-purchase customers
    # This is an imputation — flagged as higher uncertainty
    for segment in df['rfm_segment'].unique():
        seg_multi = df[(df['rfm_segment'] == segment) & (df['total_orders'] > 1)]
        if len(seg_multi) > 0:
            seg_median_lifespan = max(seg_multi['customer_tenure_days'].median() / 365, 0.5)
        else:
            # No multi-purchase in this segment — use overall median
            seg_median_lifespan = max(multi['customer_tenure_days'].median() / 365, 0.5)
        
        single_seg_mask = (df['rfm_segment'] == segment) & (df['total_orders'] == 1)
        lifespan[single_seg_mask] = seg_median_lifespan
    
    return lifespan


def compute_annualized_frequency(df: pd.DataFrame) -> pd.Series:
    """
    Compute annualized purchase frequency for each customer.
    
    For multi-purchase customers (total_orders > 1):
      annualized_frequency = (total_orders / customer_tenure_days) × 365
      Capped at 12 purchases/year (monthly) to avoid inflation from customers
      who placed multiple orders within days of each other.
    
    For single-purchase customers:
      A CONSERVATIVE approach is used instead of segment-median imputation.
      
      WHY NOT SEGMENT MEDIANS: Multi-purchase customers in this dataset have
      very short median tenure (29 days for 2-order customers), producing
      annualized frequencies of 15-25x/year. Imputing these to single-purchase
      customers would massively inflate their CLV — a customer who bought once
      6 months ago is NOT expected to buy 15 times next year.
      
      INSTEAD: We use a base rate of 1 purchase/year for single-purchase
      customers, adjusted by recency. This is the most defensible assumption
      for someone who has demonstrated exactly one transaction:
        - Recent (R_score 4-5): 2 purchases/year (optimistic — they may return)
        - Moderate (R_score 3): 1.5 purchases/year
        - Lapsed (R_score 1-2): 1 purchase/year (conservative — low return probability)
    
    ⚠️ UNCERTAINTY FLAG: Single-purchase CLV estimates carry higher uncertainty
    because their frequency is entirely imputed. In a dataset where 97% of
    customers are single-purchase, this means the majority of CLV estimates
    rest on assumptions rather than observed behavior. This is a known limitation
    documented in the notebook narrative.
    """
    freq = pd.Series(np.nan, index=df.index)
    
    # Multi-purchase customers: actual annualized frequency
    multi_mask = (df['total_orders'] > 1) & (df['customer_tenure_days'] > 0)
    freq[multi_mask] = (df.loc[multi_mask, 'total_orders'] / 
                        df.loc[multi_mask, 'customer_tenure_days']) * 365
    
    # Cap at 12 purchases/year (monthly frequency) — even loyal skincare customers
    # rarely order more than monthly. This prevents inflation from customers who
    # placed burst orders (e.g., 3 orders in 2 days → 547/year without cap).
    freq = freq.clip(upper=12)
    
    # Multi-purchase customers with 0 tenure (all orders on same day):
    # Treat as 2 purchases/year — we know they ordered at least twice but can't
    # compute a rate, so use a moderate assumption
    same_day_mask = (df['total_orders'] > 1) & (df['customer_tenure_days'] == 0)
    freq[same_day_mask] = 2.0
    
    # Print actual multi-purchase stats for transparency
    print(f"  Multi-purchase actual frequency stats (capped at 12/yr):")
    multi_freq = freq[multi_mask | same_day_mask].dropna()
    if len(multi_freq) > 0:
        print(f"    Count:  {len(multi_freq):,}")
        print(f"    Median: {multi_freq.median():.2f}/year")
        print(f"    Mean:   {multi_freq.mean():.2f}/year")
    
    # Single-purchase customers: conservative recency-adjusted imputation
    # This is NOT segment-median because those are inflated by short-tenure effects
    single_mask = df['total_orders'] == 1
    
    # Recency-adjusted base rates
    # R_score 4-5 (recent): 2/year — purchased recently, plausible near-term return
    # R_score 3 (moderate): 1.5/year — cooling but still within re-engagement window
    # R_score 1-2 (lapsed): 1/year — minimal expectation of return
    recency_freq = pd.Series(1.0, index=df.index)  # default
    recency_freq[df['R_score'] >= 4] = 2.0
    recency_freq[df['R_score'] == 3] = 1.5
    
    freq[single_mask] = recency_freq[single_mask]
    
    print(f"\n  Single-purchase imputed frequency (recency-adjusted):")
    for r in sorted(df['R_score'].unique()):
        r_single = df[(df['R_score'] == r) & single_mask]
        if len(r_single) > 0:
            print(f"    R_score={r}: {freq[r_single.index].iloc[0]:.1f}/year ({len(r_single):,} customers)")
    
    # Handle any remaining NaNs (shouldn't happen but defensive)
    freq = freq.fillna(1.0)
    
    return freq


def compute_clv(df: pd.DataFrame, gross_margin: float,
                annualized_freq: pd.Series,
                lifespan_years: pd.Series) -> pd.Series:
    """
    Compute individual CLV:
        CLV = avg_order_value × annualized_frequency × gross_margin × estimated_lifespan_years
    
    All components are positive by construction. CLV is floored at 0.
    """
    clv = (
        df['avg_order_value'] *
        annualized_freq *
        gross_margin *
        lifespan_years
    )
    return clv.clip(lower=0).round(2)


def run_sensitivity_analysis(df: pd.DataFrame, annualized_freq: pd.Series,
                              lifespan_years: pd.Series) -> pd.DataFrame:
    """
    Run CLV calculations at multiple gross margin assumptions.
    Produces a summary table: median CLV per segment under each margin.
    """
    print(f"\n  Running sensitivity analysis across margins: {SENSITIVITY_MARGINS}")
    
    results = []
    for margin in SENSITIVITY_MARGINS:
        clv = compute_clv(df, margin, annualized_freq, lifespan_years)
        for segment in df['rfm_segment'].unique():
            seg_mask = df['rfm_segment'] == segment
            results.append({
                'rfm_segment': segment,
                'gross_margin_assumption': margin,
                'median_clv': clv[seg_mask].median(),
                'total_clv': clv[seg_mask].sum(),
            })
    
    sensitivity_df = pd.DataFrame(results)
    return sensitivity_df


def print_clv_summary(df: pd.DataFrame, clv: pd.Series) -> None:
    """Print comprehensive CLV summary per segment."""
    SEGMENT_ORDER = ['Champions', 'Loyal Customers', 'Potential Loyalists',
                     'At-Risk Customers', 'Lost Customers']
    
    print(f"\n{'='*70}")
    print("CLV SUMMARY BY SEGMENT (Gross Margin = {:.0%})".format(DEFAULT_GROSS_MARGIN))
    print(f"{'='*70}")
    
    total_clv = clv.sum()
    
    rows = []
    for segment in SEGMENT_ORDER:
        seg_mask = df['rfm_segment'] == segment
        seg_clv = clv[seg_mask]
        count = seg_mask.sum()
        
        row = {
            'Segment': segment,
            'Customers': f"{count:,}",
            'Median CLV': f"R${seg_clv.median():,.2f}",
            'Mean CLV': f"R${seg_clv.mean():,.2f}",
            'Total CLV': f"R${seg_clv.sum():,.2f}",
            'CLV Share': f"{seg_clv.sum()/total_clv:.1%}",
        }
        rows.append(row)
        
        print(f"\n  {segment}")
        print(f"    Customers:    {count:>7,}")
        print(f"    Median CLV:   R${seg_clv.median():>12,.2f}")
        print(f"    Mean CLV:     R${seg_clv.mean():>12,.2f}")
        print(f"    Total CLV:    R${seg_clv.sum():>12,.2f}")
        print(f"    CLV Share:    {seg_clv.sum()/total_clv:>11.1%}")
    
    print(f"\n  {'─'*50}")
    print(f"  TOTAL PORTFOLIO CLV: R${total_clv:>12,.2f}")
    print(f"  Customers:           {len(df):>12,}")
    print(f"  Mean CLV per cust:   R${clv.mean():>12,.2f}")
    print(f"  Median CLV per cust: R${clv.median():>12,.2f}")


def main():
    print("="*70)
    print("COMPONENT 3: CLV Modeling")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/6] Loading RFM-segmented customer data...")
    df = load_data(INPUT_FILE)
    
    # Step 2: Compute annualized purchase frequency
    print("\n[2/6] Computing annualized purchase frequency...")
    annualized_freq = compute_annualized_frequency(df)
    
    multi_mask = df['total_orders'] > 1
    print(f"\n  Multi-purchase frequency stats:")
    print(f"    Count: {multi_mask.sum():,}")
    print(f"    Median annualized freq: {annualized_freq[multi_mask].median():.2f}")
    print(f"    Mean annualized freq:   {annualized_freq[multi_mask].mean():.2f}")
    print(f"\n  Single-purchase (imputed) frequency stats:")
    print(f"    Count: {(~multi_mask).sum():,}")
    print(f"    Median annualized freq: {annualized_freq[~multi_mask].median():.2f}")
    
    # Step 3: Estimate customer lifespan
    print("\n[3/6] Estimating customer lifespan...")
    
    bgnbd_success = False
    prob_alive = None
    
    if USE_BGNBD:
        try:
            result = fit_bgnbd_model(df)
            if result[0] is not None:
                bgf, predicted_purchases, prob_alive = result
                bgnbd_success = True
                
                # Convert predicted future purchases to effective lifespan
                # Lifespan = prediction_horizon × (predicted_purchases / expected_total)
                # But more directly: if BG/NBD predicts X purchases in 2 years,
                # the effective lifespan for CLV is derived from when purchases taper off.
                # 
                # APPROACH: Use predicted purchases + P(alive) to estimate effective years.
                # If a customer is predicted to make 0.5 purchases over 2 years with
                # P(alive) = 0.3, their effective lifespan is much shorter than someone
                # predicted to make 4 purchases with P(alive) = 0.9.
                #
                # Effective lifespan = P(alive) × prediction_horizon_years
                # This is a simplification but captures the key insight: customers with
                # low P(alive) have near-zero effective lifespan regardless of past behavior.
                lifespan_years = prob_alive * PREDICTION_HORIZON_YEARS
                
                # Floor at a small positive value to avoid zero CLV for everyone
                lifespan_years = lifespan_years.clip(lower=0.1)
                
                # For customers excluded from BG/NBD (T=0), use segment median
                nan_mask = lifespan_years.isna()
                if nan_mask.any():
                    for segment in df['rfm_segment'].unique():
                        seg_mask = (df['rfm_segment'] == segment) & (~nan_mask)
                        if seg_mask.any():
                            seg_median = lifespan_years[seg_mask].median()
                        else:
                            seg_median = 0.5
                        fill_mask = (df['rfm_segment'] == segment) & nan_mask
                        lifespan_years[fill_mask] = seg_median
                
                print(f"\n  BG/NBD-derived lifespan estimates:")
                print(f"    Median: {lifespan_years.median():.2f} years")
                print(f"    Mean:   {lifespan_years.mean():.2f} years")
                print(f"    Range:  {lifespan_years.min():.2f} – {lifespan_years.max():.2f} years")
        except Exception as e:
            print(f"  ⚠️  BG/NBD failed: {e}")
            bgnbd_success = False
    
    if not bgnbd_success:
        print("  Using empirical fallback for lifespan estimation...")
        lifespan_years = compute_empirical_lifespan(df)
    
    # Step 4: Compute CLV at default gross margin
    print(f"\n[4/6] Computing CLV at gross margin = {DEFAULT_GROSS_MARGIN:.0%}...")
    clv = compute_clv(df, DEFAULT_GROSS_MARGIN, annualized_freq, lifespan_years)
    
    # Print summary
    print_clv_summary(df, clv)
    
    # Step 5: Run sensitivity analysis
    print(f"\n[5/6] Running CLV sensitivity analysis...")
    sensitivity_df = run_sensitivity_analysis(df, annualized_freq, lifespan_years)
    
    print(f"\n  Sensitivity results (median CLV by segment × margin):")
    pivot = sensitivity_df.pivot_table(
        index='rfm_segment', columns='gross_margin_assumption',
        values='median_clv'
    )
    print(pivot.to_string())
    
    # Step 6: Save outputs
    print(f"\n[6/6] Saving outputs...")
    
    # Add CLV columns to the main dataframe
    df['clv_estimate'] = clv
    df['annualized_frequency'] = annualized_freq.round(4)
    df['estimated_lifespan_years'] = lifespan_years.round(4)
    if prob_alive is not None:
        df['prob_alive'] = prob_alive.round(4)
    df['lifespan_method'] = 'BG/NBD' if bgnbd_success else 'empirical'
    
    # Save primary CLV output
    df.to_csv(OUTPUT_CLV, index=False)
    print(f"  ✓ Saved {len(df):,} rows to {OUTPUT_CLV}")
    
    # Save sensitivity analysis
    sensitivity_df.to_csv(OUTPUT_SENSITIVITY, index=False)
    print(f"  ✓ Saved {len(sensitivity_df)} rows to {OUTPUT_SENSITIVITY}")
    
    # Final verification
    print(f"\n  Verification:")
    print(f"    CLV nulls:          {clv.isna().sum()}")
    print(f"    CLV zeros:          {(clv == 0).sum()}")
    print(f"    CLV negatives:      {(clv < 0).sum()}")
    print(f"    Lifespan method:    {'BG/NBD' if bgnbd_success else 'Empirical fallback'}")
    print(f"    Total portfolio CLV: R${clv.sum():,.2f}")
    
    print(f"\n{'='*70}")
    print("COMPONENT 3 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
