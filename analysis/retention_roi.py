"""
LumiSkin CLV & RFM Segmentation — Component 4: Retention ROI Model
====================================================================
This script models the retention investment ROI by comparing two approaches:
  1. CURRENT STATE: Uniform R$180,000 budget — same 15% discount coupon to
     every customer who hasn't purchased in 60+ days (R_score ≤ 3).
  2. PROPOSED STATE: Segment-differentiated interventions with tailored tactics,
     cost structures, and expected recovery rates per segment.

The headline finding — total expected revenue protection under current vs.
proposed approach at identical total spend — is the key CMO deliverable.

CRITICAL DATA NOTE:
  All analysis uses customer_unique_id (stable person-level identifier),
  NOT customer_id (which is unique per order in the Olist schema).

Inputs:  data/processed/clv_scores.csv
Outputs: data/processed/retention_roi.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
INPUT_FILE = os.path.join(PROCESSED_DIR, 'clv_scores.csv')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'retention_roi.csv')

# ---------------------------------------------------------------------------
# CURRENT STATE PARAMETERS
# ---------------------------------------------------------------------------
# Total annual retention budget (Brazilian Reais)
TOTAL_BUDGET = 180_000

# Current approach: uniform coupon to all customers with R_score ≤ 3
# (haven't purchased in 60+ days based on recency scoring breakpoints)
CURRENT_ELIGIBILITY_THRESHOLD = 3  # R_score ≤ this value

# Baseline recovery rate for uniform 15% discount coupon
# Assumption: 8% of recipients make a purchase after receiving the coupon.
# This is a reasonable baseline for untargeted re-engagement emails in D2C
# e-commerce. Industry benchmarks typically range from 5-12% depending on
# brand strength, category, and offer depth. We use 8% as a mid-range estimate.
BASELINE_RECOVERY_RATE = 0.08

# ---------------------------------------------------------------------------
# PROPOSED SEGMENTED INTERVENTION PARAMETERS
# ---------------------------------------------------------------------------
# Each segment gets a tailored intervention with different cost structures
# and expected impact metrics. All parameters are configurable here for
# easy sensitivity testing.
#
# Structure: {
#   'description': what the intervention is,
#   'cost_per_customer': R$ cost per targeted customer,
#   'metric_type': what metric the intervention improves,
#   'metric_value': the expected improvement value,
#   'mechanism': business rationale for why this works
# }

SEGMENT_INTERVENTIONS = {
    'Champions': {
        'description': 'Loyalty recognition program (no discount)',
        'cost_per_customer': 12.0,
        'budget_share': 0.02,  # 2% of budget → R$3,600
        # Expected churn reduction: from 8% baseline churn to 4% churn
        # Mechanism: Recognition (exclusive access, thank-you gifts) preserves
        # the relationship without conditioning Champions to expect discounts.
        # Discounting Champions is counterproductive — it erodes margin on
        # customers who would have purchased at full price.
        # With only 27 Champions, R$3,600 covers all of them comfortably.
        'metric_type': 'churn_reduction',
        'baseline_churn_rate': 0.08,
        'improved_churn_rate': 0.04,
        'mechanism': 'Preserves CLV without discount conditioning',
    },
    'Loyal Customers': {
        'description': 'Referral incentive program',
        'cost_per_customer': 18.0,
        'budget_share': 0.15,  # 15% of budget → R$27,000
        # Expected frequency uplift: 15% increase in annualized purchase frequency
        # Mechanism: Referral rewards (R$10 credit per successful referral) give
        # Loyal Customers a reason to engage beyond their own purchases.
        # R$27,000 covers all 1,578 Loyal Customers (R$17/customer).
        'metric_type': 'frequency_uplift',
        'uplift_pct': 0.15,
        'mechanism': '15% increase in annualized purchase frequency',
    },
    'Potential Loyalists': {
        'description': 'Second-purchase nudge at day 45 with 10% offer',
        'cost_per_customer': 9.0,
        'budget_share': 0.35,  # 35% of budget → R$63,000
        # Expected conversion to repeat purchaser: 22% of targeted customers
        # Mechanism: Category-based product recommendation + limited-time 10%
        # offer targets the critical first-to-second purchase transition.
        # R$63,000 at R$9/customer reaches ~7,000 of 11,480 (top 61% by CLV).
        'metric_type': 'conversion_to_repeat',
        'conversion_rate': 0.22,
        'mechanism': '22% conversion to repeat purchaser',
    },
    'At-Risk Customers': {
        'description': 'Three-email win-back sequence',
        'cost_per_customer': 24.0,
        'budget_share': 0.48,  # 48% of budget → R$86,400
        # Expected recovery rate: 28% of targeted customers
        # Mechanism: Escalating sequence — (1) value content, (2) 20% discount
        # on previous category, (3) free sample offer.
        # R$86,400 at R$24/customer reaches ~3,600 of 22,464 — prioritized
        # by CLV (target the most valuable At-Risk customers first).
        'metric_type': 'recovery_rate',
        'recovery_rate': 0.28,
        'mechanism': '28% recovery rate via escalating win-back',
    },
    'Lost Customers': {
        'description': 'No spend — reallocate budget elsewhere',
        'cost_per_customer': 0.0,
        'budget_share': 0.00,  # 0% of budget
        # No intervention for Lost Customers.
        # Rationale: These customers have both low recency AND low frequency.
        # The expected recovery rate is below the cost threshold, making
        # investment here NPV-negative.
        'metric_type': 'none',
        'mechanism': 'No spend — below ROI threshold',
    },
}

SEGMENT_ORDER = ['Champions', 'Loyal Customers', 'Potential Loyalists',
                 'At-Risk Customers', 'Lost Customers']


def load_data(filepath: str) -> pd.DataFrame:
    """Load CLV-scored customer data and validate."""
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found. Run Component 3 first.")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    required = ['customer_unique_id', 'rfm_segment', 'R_score',
                'clv_estimate', 'annualized_frequency', 'avg_order_value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} customers from clv_scores.csv")
    return df


def compute_current_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model the current uniform retention approach.
    
    Current approach: R$180,000 distributed uniformly across all customers
    with R_score ≤ 3 (haven't purchased in 60+ days). Every eligible customer
    gets the same 15% discount coupon regardless of segment, value, or
    churn probability.
    """
    print("\n  CURRENT STATE: Uniform budget allocation")
    
    # Identify eligible customers (R_score ≤ 3 = haven't purchased in 60+ days)
    eligible_mask = df['R_score'] <= CURRENT_ELIGIBILITY_THRESHOLD
    total_eligible = eligible_mask.sum()
    
    print(f"  Eligible customers (R_score ≤ {CURRENT_ELIGIBILITY_THRESHOLD}): {total_eligible:,}")
    print(f"  Total budget: R${TOTAL_BUDGET:,.2f}")
    
    cost_per_customer = TOTAL_BUDGET / total_eligible if total_eligible > 0 else 0
    print(f"  Cost per customer: R${cost_per_customer:.2f}")
    
    results = []
    for segment in SEGMENT_ORDER:
        seg_mask = (df['rfm_segment'] == segment) & eligible_mask
        seg_count = seg_mask.sum()
        seg_total = (df['rfm_segment'] == segment).sum()
        
        # Budget allocated to this segment under uniform approach
        seg_budget = seg_count * cost_per_customer
        
        # Expected recovered customers (uniform 8% recovery rate)
        recovered = seg_count * BASELINE_RECOVERY_RATE
        
        # Expected revenue impact = recovered customers × their median CLV
        # Using median CLV as the expected value per recovered customer
        seg_clv = df.loc[df['rfm_segment'] == segment, 'clv_estimate']
        median_clv = seg_clv.median()
        revenue_impact = recovered * median_clv
        
        # ROI = (revenue impact - cost) / cost
        roi = (revenue_impact - seg_budget) / seg_budget if seg_budget > 0 else 0
        
        results.append({
            'rfm_segment': segment,
            'total_customers': seg_total,
            'eligible_customers_current': seg_count,
            'current_budget_allocation': round(seg_budget, 2),
            'current_cost_per_customer': round(cost_per_customer, 2),
            'current_expected_recovered': round(recovered, 1),
            'current_revenue_impact': round(revenue_impact, 2),
            'roi_current': round(roi, 4),
            'median_clv': round(median_clv, 2),
        })
        
        print(f"\n  {segment}:")
        print(f"    Eligible:     {seg_count:>7,} of {seg_total:,}")
        print(f"    Budget:       R${seg_budget:>10,.2f}")
        print(f"    Recovered:    {recovered:>7,.1f}")
        print(f"    Revenue:      R${revenue_impact:>10,.2f}")
        print(f"    ROI:          {roi:>10.1%}")
    
    return pd.DataFrame(results)


def compute_proposed_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model the proposed segment-differentiated retention approach.
    
    BUDGET-FIRST ALLOCATION: Instead of computing cost × all_customers and
    rescaling (which produces unrealistic results), we allocate fixed budget
    shares to each segment and compute how many customers can be reached
    within the allocated budget. This is the approach a real consulting
    engagement would use — you don't have infinite budget.
    
    Reach = allocated_budget / cost_per_customer, capped at segment size.
    Impact metrics (recovery rate, conversion rate, etc.) apply only to
    the customers actually reached.
    
    PRIORITIZATION LOGIC: Within each segment, we assume the highest-CLV
    customers are targeted first (CLV-ranked targeting). This means the
    expected revenue impact per reached customer is above-median.
    """
    print(f"\n\n  PROPOSED STATE: Budget-share allocation")
    
    # Validate budget shares sum to 1.0
    total_share = sum(v['budget_share'] for v in SEGMENT_INTERVENTIONS.values())
    print(f"  Budget share total: {total_share:.2f} (should be 1.00)")
    assert abs(total_share - 1.0) < 0.01, f"Budget shares must sum to 1.0, got {total_share}"
    
    results = []
    total_proposed_cost = 0
    
    for segment in SEGMENT_ORDER:
        seg_mask = df['rfm_segment'] == segment
        seg_df = df[seg_mask].copy()
        seg_count = len(seg_df)
        intervention = SEGMENT_INTERVENTIONS[segment]
        
        # Budget allocation from share
        allocated_budget = TOTAL_BUDGET * intervention['budget_share']
        cost_per_cust = intervention['cost_per_customer']
        
        # Reach: how many customers can we target within budget?
        if cost_per_cust > 0:
            max_reach = int(allocated_budget / cost_per_cust)
            reach = min(max_reach, seg_count)  # Can't exceed segment size
        else:
            reach = 0
        
        actual_cost = reach * cost_per_cust
        total_proposed_cost += actual_cost
        coverage_pct = reach / seg_count * 100 if seg_count > 0 else 0
        
        # Compute revenue impact based on intervention type
        # For CLV-ranked targeting, use mean CLV of top-N customers
        seg_sorted = seg_df.sort_values('clv_estimate', ascending=False)
        if reach > 0:
            targeted = seg_sorted.head(reach)
            targeted_mean_clv = targeted['clv_estimate'].mean()
            targeted_median_clv = targeted['clv_estimate'].median()
        else:
            targeted_mean_clv = 0
            targeted_median_clv = 0
        
        if intervention['metric_type'] == 'churn_reduction':
            baseline_churn = intervention['baseline_churn_rate']
            improved_churn = intervention['improved_churn_rate']
            customers_saved = reach * (baseline_churn - improved_churn)
            revenue_impact = customers_saved * targeted_mean_clv
            expected_retained = customers_saved
            
        elif intervention['metric_type'] == 'frequency_uplift':
            uplift = intervention['uplift_pct']
            avg_aov = targeted['avg_order_value'].mean() if reach > 0 else 0
            avg_freq = targeted['annualized_frequency'].mean() if reach > 0 else 0
            additional_purchases = reach * avg_freq * uplift
            revenue_impact = additional_purchases * avg_aov * 0.65
            expected_retained = additional_purchases
            
        elif intervention['metric_type'] == 'conversion_to_repeat':
            conversion_rate = intervention['conversion_rate']
            converted = reach * conversion_rate
            revenue_impact = converted * targeted_median_clv
            expected_retained = converted
            
        elif intervention['metric_type'] == 'recovery_rate':
            recovery_rate = intervention['recovery_rate']
            recovered = reach * recovery_rate
            revenue_impact = recovered * targeted_median_clv
            expected_retained = recovered
            
        else:  # 'none'
            revenue_impact = 0
            expected_retained = 0
        
        roi = (revenue_impact - actual_cost) / actual_cost if actual_cost > 0 else 0
        
        results.append({
            'rfm_segment': segment,
            'intervention': intervention['description'],
            'proposed_budget_allocation': round(allocated_budget, 2),
            'proposed_cost_per_customer': cost_per_cust,
            'proposed_reach': reach,
            'proposed_coverage_pct': round(coverage_pct, 1),
            'proposed_expected_retained': round(expected_retained, 1),
            'proposed_revenue_impact': round(revenue_impact, 2),
            'roi_proposed': round(roi, 4),
            'targeted_mean_clv': round(targeted_mean_clv, 2),
        })
        
        print(f"\n  {segment}: {intervention['description']}")
        print(f"    Segment size: {seg_count:>7,}")
        print(f"    Budget:       R${allocated_budget:>10,.2f} ({intervention['budget_share']:.0%})")
        print(f"    Cost/cust:    R${cost_per_cust:>7.2f}")
        print(f"    Reach:        {reach:>7,} of {seg_count:,} ({coverage_pct:.0f}%)")
        print(f"    Retained:     {expected_retained:>7,.1f}")
        print(f"    Revenue:      R${revenue_impact:>10,.2f}")
        print(f"    ROI:          {roi:>10.1%}")
    
    print(f"\n  Total proposed spend: R${total_proposed_cost:,.2f}")
    
    return pd.DataFrame(results)


def print_headline_finding(merged: pd.DataFrame) -> None:
    """Print the headline finding comparing current vs proposed approaches."""
    print(f"\n{'='*70}")
    print("HEADLINE FINDING")
    print(f"{'='*70}")
    
    current_total_revenue = merged['current_revenue_impact'].sum()
    proposed_total_revenue = merged['proposed_revenue_impact'].sum()
    current_total_budget = merged['current_budget_allocation'].sum()
    proposed_total_budget = merged['proposed_budget_allocation'].sum()
    
    delta = proposed_total_revenue - current_total_revenue
    pct_improvement = delta / current_total_revenue * 100 if current_total_revenue > 0 else 0
    
    current_roi = (current_total_revenue - current_total_budget) / current_total_budget
    proposed_roi = (proposed_total_revenue - proposed_total_budget) / proposed_total_budget
    
    print(f"\n  Budget constraint:         R${TOTAL_BUDGET:>12,.2f} (identical)")
    print(f"\n  Current approach (uniform):")
    print(f"    Total revenue protected: R${current_total_revenue:>12,.2f}")
    print(f"    Portfolio ROI:           {current_roi:>12.1%}")
    print(f"\n  Proposed approach (segmented):")
    print(f"    Total revenue protected: R${proposed_total_revenue:>12,.2f}")
    print(f"    Portfolio ROI:           {proposed_roi:>12.1%}")
    print(f"\n  IMPROVEMENT:")
    print(f"    Additional revenue:      R${delta:>12,.2f}")
    print(f"    Percentage uplift:       {pct_improvement:>11.1f}%")
    print(f"\n  ─────────────────────────────────────────────")
    print(f"  By reallocating the same R${TOTAL_BUDGET:,.0f} budget from")
    print(f"  a uniform 15% discount to segment-differentiated interventions,")
    print(f"  LumiSkin can protect an additional R${delta:,.0f} in customer")
    print(f"  lifetime value — a {pct_improvement:.0f}% improvement in retention ROI.")
    
    # Per-segment comparison
    print(f"\n\n  SEGMENT-LEVEL COMPARISON:")
    print(f"  {'Segment':<22} {'Current Budget':>14} {'Proposed Budget':>15} {'Curr Revenue':>14} {'Prop Revenue':>14} {'ROI Δ':>8}")
    print(f"  {'─'*87}")
    for _, row in merged.iterrows():
        roi_delta = row['roi_proposed'] - row['roi_current']
        print(f"  {row['rfm_segment']:<22} "
              f"R${row['current_budget_allocation']:>11,.2f} "
              f"R${row['proposed_budget_allocation']:>12,.2f} "
              f"R${row['current_revenue_impact']:>11,.2f} "
              f"R${row['proposed_revenue_impact']:>11,.2f} "
              f"{roi_delta:>+7.1%}")


def main():
    print("="*70)
    print("COMPONENT 4: Retention ROI Model")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/4] Loading CLV-scored customer data...")
    df = load_data(INPUT_FILE)
    
    # Step 2: Compute current state
    print("\n[2/4] Modeling current uniform retention approach...")
    current_df = compute_current_state(df)
    
    # Step 3: Compute proposed state (budget-share allocation, no rescaling needed)
    print("\n[3/4] Modeling proposed segmented retention approach...")
    proposed_df = compute_proposed_state(df)
    
    # Step 4: Merge and save
    print("\n[4/4] Merging results and saving...")
    
    merged = current_df.merge(proposed_df, on='rfm_segment', how='outer')
    
    # Fill NaN for Lost Customers (no proposed spend)
    for col in merged.columns:
        if merged[col].dtype in ['float64', 'int64']:
            merged[col] = merged[col].fillna(0)
    
    # Print headline finding
    print_headline_finding(merged)
    
    # Save output
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✓ Saved {len(merged)} rows to {OUTPUT_FILE}")
    
    # Verification
    print(f"\n  Verification:")
    print(f"    Total current budget:  R${merged['current_budget_allocation'].sum():,.2f}")
    print(f"    Total proposed budget: R${merged['proposed_budget_allocation'].sum():,.2f}")
    print(f"    Segments covered:      {len(merged)}")
    print(f"    Null values:           {merged.isnull().sum().sum()}")
    
    print(f"\n{'='*70}")
    print("COMPONENT 4 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
