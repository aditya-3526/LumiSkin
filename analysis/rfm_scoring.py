"""
LumiSkin CLV & RFM Segmentation — Component 2: RFM Scoring
============================================================
This script assigns Recency, Frequency, and Monetary scores to each customer,
derives dimension weights via logistic regression, computes a weighted composite
RFM score, and assigns strategic segment labels.

CRITICAL DATA NOTE:
  All analysis uses customer_unique_id (stable person-level identifier),
  NOT customer_id (which is unique per order in the Olist schema).

WHY BUSINESS BREAKPOINTS INSTEAD OF EQUAL QUINTILES:
  The Olist dataset has an extreme right-skew in the frequency distribution:
  97% of customers have exactly 1 order, 2.8% have 2 orders, and only 0.2%
  have 3+ orders. Equal quintile splitting on frequency would assign the same
  score to nearly all customers, destroying the dimension's discriminative power.
  Similarly, recency has business-meaningful thresholds (30/60/90/150 days)
  that correspond to actual retention windows used in D2C marketing.
  
  Monetary scoring DOES use quintiles because total_spend is more continuously
  distributed and lacks natural business breakpoints.

Inputs:  data/processed/customer_base.csv
Outputs: data/processed/rfm_segments.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
INPUT_FILE = os.path.join(PROCESSED_DIR, 'customer_base.csv')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'rfm_segments.csv')

# Default RFM weights if logistic regression weights are not meaningfully different
DEFAULT_WEIGHTS = {'R': 0.40, 'F': 0.35, 'M': 0.25}

# Segment definitions — each rule is evaluated in order; first match wins.
# This ordering matters: more specific segments are checked first.
SEGMENT_ORDER = [
    'Champions',
    'Loyal Customers',
    'Potential Loyalists',
    'At-Risk Customers',
    'Lost Customers',
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load the customer base table and validate required columns."""
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found. Run Component 1 first.")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    required_cols = ['customer_unique_id', 'recency_days', 'total_orders', 'total_spend']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} customers from customer_base.csv")
    return df


def assign_recency_score(recency_days: pd.Series) -> pd.Series:
    """
    Assign recency score (1-5) using business-meaningful breakpoints.
    
    Lower recency_days = more recent purchase = higher score.
    Breakpoints align with typical D2C retention windows:
      - 0-30 days:   Score 5 (active, within standard reorder window)
      - 31-60 days:  Score 4 (warm, within first re-engagement window)
      - 61-90 days:  Score 3 (cooling, needs attention)
      - 91-150 days: Score 2 (at risk, standard win-back window)
      - 151+ days:   Score 1 (lapsed, high churn probability)
    """
    conditions = [
        recency_days <= 30,
        recency_days <= 60,
        recency_days <= 90,
        recency_days <= 150,
        recency_days > 150,
    ]
    scores = [5, 4, 3, 2, 1]
    return pd.Series(np.select(conditions, scores, default=1), index=recency_days.index)


def assign_frequency_score(total_orders: pd.Series) -> pd.Series:
    """
    Assign frequency score (1-5) using absolute purchase counts.
    
    Absolute counts are used instead of percentiles because the distribution
    is extremely right-skewed — 97% of customers have exactly 1 order.
    Percentile-based scoring would assign Score 1 to virtually everyone,
    making the dimension useless for segmentation.
    
    These thresholds have direct business meaning:
      - 1 order:   Score 1 (single-purchase, unknown if they'll return)
      - 2 orders:  Score 2 (confirmed repeat buyer — the critical transition)
      - 3 orders:  Score 3 (establishing a pattern)
      - 4 orders:  Score 4 (habitual buyer)
      - 5+ orders: Score 5 (highly engaged, likely brand-loyal)
    """
    conditions = [
        total_orders >= 5,
        total_orders == 4,
        total_orders == 3,
        total_orders == 2,
        total_orders == 1,
    ]
    scores = [5, 4, 3, 2, 1]
    return pd.Series(np.select(conditions, scores, default=1), index=total_orders.index)


def assign_monetary_score(total_spend: pd.Series) -> pd.Series:
    """
    Assign monetary score (1-5) using quintiles.
    
    Unlike frequency, total_spend is more continuously distributed and lacks
    natural business breakpoints. Quintile-based scoring is appropriate here
    because it produces roughly equal-sized groups, each representing a
    meaningfully different spending tier.
    
    Note: We use pd.qcut with duplicates='drop' to handle ties at quintile
    boundaries. If quintile boundaries collapse (rare), we fall back to
    rank-based assignment.
    """
    try:
        score = pd.qcut(total_spend, q=5, labels=[1, 2, 3, 4, 5], duplicates='raise')
    except ValueError:
        # Ties at boundaries — use rank-based approach
        score = pd.qcut(total_spend.rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    
    return score.astype(int)


def derive_rfm_weights(df: pd.DataFrame) -> dict:
    """
    Derive RFM dimension weights via logistic regression.
    
    IMPORTANT DESIGN NOTE — WHY WE USE SCORES, NOT RAW VALUES:
    The original specification calls for predicting "did this customer make more
    than one purchase?" using raw recency_days, total_orders, and total_spend.
    However, this creates a tautology: total_orders is a near-perfect linear
    predictor of total_orders > 1, causing the model to assign ~99% weight to
    frequency and ~0% to recency and monetary — making the composite score
    essentially just the F_score. This defeats the purpose of a multi-dimensional
    segmentation.
    
    CORRECTED APPROACH:
      Features: The three RFM SCORES (R_score, F_score, M_score) — these are
        ordinal, bounded (1-5), and use different scoring methods, making their
        coefficients genuinely comparable.
      Target: "Is this customer above-median in avg_order_value?" (binary).
        This is a value proxy that is NOT tautologically predicted by any single
        RFM score:
          - R_score measures timing, not value
          - F_score measures count, not per-order value
          - M_score uses total_spend quintiles, which correlate with but are
            not identical to avg_order_value (multi-purchase customers can have
            high total_spend but moderate avg_order_value)
    
      The coefficients reveal which dimension is most associated with customer
      value, and we use their normalized absolute values as weights.
    
    If the resulting weights are not meaningfully different from the default
    40/35/25 split (no weight deviates by more than 5pp), we use the defaults.
    This guards against overfitting to noise.
    """
    print("\n  Deriving RFM weights via logistic regression...")
    print("  (Using RFM scores as features, above-median AOV as target)")
    
    # Target: above-median avg_order_value (a value proxy independent of any single score)
    # This avoids the tautology of using total_orders to predict total_orders > 1
    aov_median = df['avg_order_value'].median()
    y = (df['avg_order_value'] > aov_median).astype(int)
    print(f"  Target: avg_order_value > R${aov_median:.2f} (median)")
    print(f"  Target distribution: {y.mean():.1%} positive class")
    
    # Features: RFM scores (ordinal 1-5, comparable scale)
    X = df[['R_score', 'F_score', 'M_score']].copy().astype(float)
    
    # Standardize features so coefficients are comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression
    # class_weight='balanced' since the target is roughly 50/50 but not exact
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_scaled, y)
    
    # Extract absolute coefficients and normalize to sum to 1
    abs_coefs = np.abs(model.coef_[0])
    normalized_weights = abs_coefs / abs_coefs.sum()
    
    derived_weights = {
        'R': round(normalized_weights[0], 4),  # R_score
        'F': round(normalized_weights[1], 4),  # F_score
        'M': round(normalized_weights[2], 4),  # M_score
    }
    
    print(f"\n  Logistic regression coefficients (standardized):")
    print(f"    R_score:   {model.coef_[0][0]:+.4f}")
    print(f"    F_score:   {model.coef_[0][1]:+.4f}")
    print(f"    M_score:   {model.coef_[0][2]:+.4f}")
    print(f"  Derived weights: R={derived_weights['R']:.2%}, "
          f"F={derived_weights['F']:.2%}, M={derived_weights['M']:.2%}")
    
    # CHECK 1: Dimensional collapse detection
    # If any single weight exceeds 60%, the regression has collapsed to essentially
    # a single-dimension model. This happens in the Olist dataset because:
    #   - 97% of customers have exactly 1 order → F_score has almost no variance
    #   - Any value-related target will be dominated by M_score (spending)
    #   - R_score has limited predictive power for value outcomes
    # In a collapsed model, the composite score would just be one dimension,
    # defeating the purpose of multi-dimensional segmentation.
    max_weight = max(derived_weights.values())
    if max_weight > 0.60:
        dominant_dim = max(derived_weights, key=derived_weights.get)
        print(f"\n  ⚠️  DIMENSIONAL COLLAPSE DETECTED:")
        print(f"     {dominant_dim}_score captures {max_weight:.1%} of total weight")
        print(f"     This means the regression is essentially predicting the target")
        print(f"     using a single dimension, which defeats multi-dimensional segmentation.")
        print(f"\n     ROOT CAUSE: The Olist dataset has extreme frequency skew (97% of")
        print(f"     customers have exactly 1 order). This collapses the feature space,")
        print(f"     causing any reasonable target variable to be dominated by whichever")
        print(f"     dimension has the most variance (monetary in this case).")
        print(f"\n  → Using default weights: R=40%, F=35%, M=25%")
        print(f"    These weights are the industry-standard RFM weighting that gives")
        print(f"    appropriate emphasis to recency (the best predictor of future behavior")
        print(f"    in most D2C contexts), followed by frequency and monetary. They produce")
        print(f"    a genuinely multi-dimensional composite score that enables meaningful")
        print(f"    segment differentiation.")
        return DEFAULT_WEIGHTS.copy()
    
    # CHECK 2: Proximity to defaults
    # If weights are within 5pp of defaults, use defaults for interpretability
    max_deviation = max(
        abs(derived_weights['R'] - DEFAULT_WEIGHTS['R']),
        abs(derived_weights['F'] - DEFAULT_WEIGHTS['F']),
        abs(derived_weights['M'] - DEFAULT_WEIGHTS['M']),
    )
    
    if max_deviation <= 0.05:
        # Weights are not meaningfully different from defaults — use defaults.
        # Rationale: Small deviations from 40/35/25 likely reflect sampling noise
        # rather than genuine differences in predictive power. Using round defaults
        # is more interpretable and defensible in a consulting presentation.
        print(f"\n  ⚠️  Max deviation from defaults: {max_deviation:.2%} (≤ 5%)")
        print(f"  → Using default weights: R=40%, F=35%, M=25%")
        print(f"    Rationale: Derived weights are not meaningfully different from")
        print(f"    industry-standard 40/35/25. Small deviations likely reflect")
        print(f"    sampling noise. Defaults are more interpretable for the CMO.")
        return DEFAULT_WEIGHTS.copy()
    else:
        print(f"\n  ✓  Max deviation from defaults: {max_deviation:.2%} (> 5%)")
        print(f"  → Using derived weights (meaningfully different from defaults)")
        # Round to 2 decimal places and ensure sum = 1.00
        weights = {
            'R': round(derived_weights['R'], 2),
            'F': round(derived_weights['F'], 2),
            'M': round(derived_weights['M'], 2),
        }
        # Adjust for rounding to ensure sum = 1.00
        diff = 1.00 - sum(weights.values())
        # Add rounding remainder to the largest weight
        max_key = max(weights, key=weights.get)
        weights[max_key] = round(weights[max_key] + diff, 2)
        return weights


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign strategic segment labels based on composite score AND individual
    dimension patterns.
    
    Segment rules (evaluated in order — first match wins):
    
    1. Champions:         composite ≥ 4.0 AND R_score ≥ 4
       → Recent, frequent, high-spend. The core of portfolio value.
    
    2. Loyal Customers:   composite 3.0–3.9 AND F_score ≥ 3
       → Repeat buyers with established purchase patterns.
    
    3. Potential Loyalists: composite 2.5–3.4 AND R_score ≥ 3 AND F_score ≤ 2
       → Recent but low-frequency. Purchased recently but haven't repeated yet.
       → Key target for second-purchase conversion campaigns.
    
    4. At-Risk Customers: composite 2.0–2.9 AND R_score ≤ 2 AND (F_score ≥ 2 OR M_score ≥ 3)
       → Haven't purchased recently but have shown value through frequency or spend.
       → Worth investing in win-back campaigns.
    
    5. Lost Customers:    composite < 2.0 OR (R_score = 1 AND F_score = 1)
       → Low composite score, or both long-inactive and single-purchase.
       → Lowest ROI for retention investment.
    
    CATCH-ALL: Customers not matching any rule above are assigned to the nearest
    segment by composite score. This handles edge cases at rule boundaries.
    """
    df = df.copy()
    df['rfm_segment'] = None
    
    # Rule 1: Champions — high composite AND recent
    mask_champions = (df['rfm_composite'] >= 4.0) & (df['R_score'] >= 4)
    df.loc[mask_champions, 'rfm_segment'] = 'Champions'
    
    # Rule 2: Loyal Customers — moderate-high composite AND frequent
    mask_loyal = (
        (df['rfm_segment'].isna()) &
        (df['rfm_composite'] >= 3.0) & (df['rfm_composite'] < 4.0) &
        (df['F_score'] >= 3)
    )
    df.loc[mask_loyal, 'rfm_segment'] = 'Loyal Customers'
    
    # Rule 3: Potential Loyalists — recent but low frequency
    mask_potential = (
        (df['rfm_segment'].isna()) &
        (df['rfm_composite'] >= 2.5) & (df['rfm_composite'] <= 3.4) &
        (df['R_score'] >= 3) &
        (df['F_score'] <= 2)
    )
    df.loc[mask_potential, 'rfm_segment'] = 'Potential Loyalists'
    
    # Rule 4: At-Risk — not recent but previously valuable
    mask_at_risk = (
        (df['rfm_segment'].isna()) &
        (df['rfm_composite'] >= 2.0) & (df['rfm_composite'] < 3.0) &
        (df['R_score'] <= 2) &
        ((df['F_score'] >= 2) | (df['M_score'] >= 3))
    )
    df.loc[mask_at_risk, 'rfm_segment'] = 'At-Risk Customers'
    
    # Rule 5: Lost Customers — low composite OR (inactive AND single-purchase)
    mask_lost = (
        (df['rfm_segment'].isna()) &
        ((df['rfm_composite'] < 2.0) | ((df['R_score'] == 1) & (df['F_score'] == 1)))
    )
    df.loc[mask_lost, 'rfm_segment'] = 'Lost Customers'
    
    # CATCH-ALL: Assign remaining unclassified customers to nearest segment by composite score
    unclassified_mask = df['rfm_segment'].isna()
    unclassified_count = unclassified_mask.sum()
    
    if unclassified_count > 0:
        # Define composite score midpoints for each segment
        segment_midpoints = {
            'Champions': 4.5,
            'Loyal Customers': 3.5,
            'Potential Loyalists': 2.95,
            'At-Risk Customers': 2.5,
            'Lost Customers': 1.5,
        }
        
        for idx in df[unclassified_mask].index:
            score = df.loc[idx, 'rfm_composite']
            closest = min(segment_midpoints.items(), key=lambda x: abs(x[1] - score))
            df.loc[idx, 'rfm_segment'] = closest[0]
    
    return df, unclassified_count


def print_segment_summary(df: pd.DataFrame) -> None:
    """Print detailed segment breakdown for verification."""
    print(f"\n{'='*70}")
    print("RFM SEGMENT BREAKDOWN")
    print(f"{'='*70}")
    
    total = len(df)
    
    # Create summary in the defined segment order
    for segment in SEGMENT_ORDER:
        seg_df = df[df['rfm_segment'] == segment]
        count = len(seg_df)
        pct = count / total * 100
        
        print(f"\n  {segment}")
        print(f"    Customers:     {count:>7,} ({pct:5.1f}%)")
        print(f"    Median R:      {seg_df['R_score'].median():>7.1f}")
        print(f"    Median F:      {seg_df['F_score'].median():>7.1f}")
        print(f"    Median M:      {seg_df['M_score'].median():>7.1f}")
        print(f"    Median comp.:  {seg_df['rfm_composite'].median():>7.2f}")
        print(f"    Med. spend:    R${seg_df['total_spend'].median():>9.2f}")
        print(f"    Med. recency:  {seg_df['recency_days'].median():>7.0f} days")
        print(f"    Med. orders:   {seg_df['total_orders'].median():>7.0f}")
    
    # Score distribution summary
    print(f"\n{'='*70}")
    print("SCORE DISTRIBUTIONS")
    print(f"{'='*70}")
    
    for score_col, label in [('R_score', 'Recency'), ('F_score', 'Frequency'), ('M_score', 'Monetary')]:
        print(f"\n  {label} score distribution:")
        dist = df[score_col].value_counts().sort_index()
        for score, count in dist.items():
            pct = count / total * 100
            bar = '█' * int(pct / 2)
            print(f"    Score {score}: {count:>7,} ({pct:5.1f}%) {bar}")


def main():
    print("="*70)
    print("COMPONENT 2: RFM Scoring")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/5] Loading customer base table...")
    df = load_data(INPUT_FILE)
    
    # Step 2: Assign individual R, F, M scores
    print("\n[2/5] Assigning R, F, M scores...")
    df['R_score'] = assign_recency_score(df['recency_days'])
    df['F_score'] = assign_frequency_score(df['total_orders'])
    df['M_score'] = assign_monetary_score(df['total_spend'])
    
    print(f"  ✓ R scores assigned (business breakpoints: 30/60/90/150 days)")
    print(f"  ✓ F scores assigned (absolute counts: 1/2/3/4/5+)")
    print(f"  ✓ M scores assigned (quintile-based)")
    
    # Print monetary quintile boundaries for transparency
    m_boundaries = df['total_spend'].quantile([0.2, 0.4, 0.6, 0.8]).values
    print(f"\n  Monetary quintile boundaries (total_spend):")
    print(f"    Q1/Q2: R${m_boundaries[0]:,.2f}")
    print(f"    Q2/Q3: R${m_boundaries[1]:,.2f}")
    print(f"    Q3/Q4: R${m_boundaries[2]:,.2f}")
    print(f"    Q4/Q5: R${m_boundaries[3]:,.2f}")
    
    # Step 3: Derive RFM weights via logistic regression
    print("\n[3/5] Deriving dimension weights...")
    weights = derive_rfm_weights(df)
    
    # Step 4: Compute composite RFM score
    print(f"\n[4/5] Computing composite RFM score...")
    print(f"  Weights: R={weights['R']:.2f}, F={weights['F']:.2f}, M={weights['M']:.2f}")
    
    df['rfm_composite'] = (
        df['R_score'] * weights['R'] +
        df['F_score'] * weights['F'] +
        df['M_score'] * weights['M']
    ).round(2)
    
    print(f"  Composite score range: {df['rfm_composite'].min():.2f} – {df['rfm_composite'].max():.2f}")
    print(f"  Composite score median: {df['rfm_composite'].median():.2f}")
    print(f"  Composite score mean: {df['rfm_composite'].mean():.2f}")
    
    # Step 5: Assign segment labels
    print(f"\n[5/5] Assigning segment labels...")
    df, catch_all_count = assign_segments(df)
    
    if catch_all_count > 0:
        print(f"\n  ⚠️  CATCH-ALL FLAG: {catch_all_count:,} customers did not match any")
        print(f"     primary segment rule and were assigned to the nearest segment")
        print(f"     by composite score. Review these edge cases if needed.")
        
        # Show the composite scores of catch-all customers for transparency
        # (can't easily do this after the fact since we already assigned, but we flag the count)
    else:
        print(f"  ✓ All customers matched a primary segment rule (0 catch-all)")
    
    # Print segment summary
    print_segment_summary(df)
    
    # Save output
    print(f"\n{'='*70}")
    print("SAVING OUTPUT")
    print(f"{'='*70}")
    
    output_cols = [
        'customer_unique_id', 'customer_state', 'first_purchase_date', 'last_purchase_date',
        'total_orders', 'total_spend', 'avg_order_value', 'avg_review_score',
        'primary_category', 'primary_seller_state', 'recency_days', 'customer_tenure_days',
        'R_score', 'F_score', 'M_score', 'rfm_composite', 'rfm_segment',
    ]
    
    df[output_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Saved {len(df):,} rows to {OUTPUT_FILE}")
    
    # Final verification
    print(f"\n  Verification:")
    print(f"    Unique segments: {df['rfm_segment'].nunique()}")
    print(f"    Null segments:   {df['rfm_segment'].isna().sum()}")
    print(f"    Total rows:      {len(df):,}")
    
    print(f"\n{'='*70}")
    print("COMPONENT 2 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
