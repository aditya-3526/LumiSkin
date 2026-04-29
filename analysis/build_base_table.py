"""
LumiSkin CLV & RFM Segmentation — Component 1: Data Ingestion & Analytical Base Table
=======================================================================================
This script loads the raw Olist CSV files into an in-memory SQLite database,
executes the six-CTE SQL pipeline defined in sql/build_base_table.sql, and
exports the resulting customer-level analytical base table.

CRITICAL DATA WARNING:
  - customer_id is unique per ORDER, not per person. We use customer_unique_id.
  - order_payments has one row per installment — aggregated in the SQL pipeline.
  - Only delivered orders with non-null delivery dates are included.
  - Reference date = MAX(order_purchase_timestamp) in the dataset, not today().

Inputs:  data/raw/*.csv (7 Olist files + sellers)
Outputs: data/processed/customer_base.csv
"""

import os
import sys
import sqlite3
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
SQL_DIR = os.path.join(PROJECT_ROOT, 'sql')

# CSV files to load into SQLite, mapped to their table names
CSV_TABLE_MAP = {
    'olist_orders_dataset.csv': 'orders',
    'olist_order_items_dataset.csv': 'order_items',
    'olist_order_payments_dataset.csv': 'order_payments',
    'olist_customers_dataset.csv': 'customers',
    'olist_products_dataset.csv': 'products',
    'olist_order_reviews_dataset.csv': 'order_reviews',
    'olist_sellers_dataset.csv': 'sellers',
    'product_category_name_translation.csv': 'category_translation',
}


def validate_raw_files(raw_dir: str, csv_table_map: dict) -> None:
    """Check that all required CSV files exist in the raw data directory."""
    missing = []
    for filename in csv_table_map:
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            missing.append(filename)
    if missing:
        print("ERROR: The following required files are missing from data/raw/:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease place these files in data/raw/ before running this script.")
        sys.exit(1)


def load_csvs_to_sqlite(raw_dir: str, csv_table_map: dict, conn: sqlite3.Connection) -> dict:
    """Load each CSV into the SQLite database. Returns row counts per table."""
    row_counts = {}
    for filename, table_name in csv_table_map.items():
        filepath = os.path.join(raw_dir, filename)
        df = pd.read_csv(filepath)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        row_counts[table_name] = len(df)
        print(f"  Loaded {table_name:25s} → {len(df):>8,} rows  ({len(df.columns)} columns)")
    return row_counts


def execute_sql_pipeline(conn: sqlite3.Connection, sql_path: str) -> pd.DataFrame:
    """Read and execute the SQL pipeline, returning the result as a DataFrame."""
    with open(sql_path, 'r') as f:
        sql = f.read()
    return pd.read_sql_query(sql, conn)


def print_diagnostics(df: pd.DataFrame, table_name: str) -> None:
    """Print verification diagnostics for the output table."""
    print(f"\n{'='*70}")
    print(f"VERIFICATION: {table_name}")
    print(f"{'='*70}")
    
    # Row count
    print(f"\nRow count: {len(df):,}")
    
    # Column names
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    
    # Null counts per column
    print(f"\nNull counts per column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        pct = (count / len(df)) * 100
        flag = " ⚠️" if pct > 5 else ""
        print(f"  {col:30s} → {count:>6,} nulls ({pct:5.1f}%){flag}")
    
    # Basic stats for key numeric columns
    print(f"\nKey statistics:")
    numeric_cols = ['total_orders', 'total_spend', 'avg_order_value', 
                    'recency_days', 'customer_tenure_days', 'avg_review_score']
    for col in numeric_cols:
        if col in df.columns:
            print(f"  {col}:")
            print(f"    min={df[col].min():.2f}, median={df[col].median():.2f}, "
                  f"max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    # First 5 rows
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    
    # Data quality flags
    print(f"\n{'='*70}")
    print("DATA QUALITY FLAGS:")
    print(f"{'='*70}")
    
    # Check for customers with 0 spend
    zero_spend = (df['total_spend'] == 0).sum()
    if zero_spend > 0:
        print(f"  ⚠️  {zero_spend:,} customers with zero total spend")
    else:
        print(f"  ✓  No customers with zero spend")
    
    # Check for negative recency (should not happen)
    neg_recency = (df['recency_days'] < 0).sum()
    if neg_recency > 0:
        print(f"  ⚠️  {neg_recency:,} customers with negative recency days")
    else:
        print(f"  ✓  No negative recency values")
    
    # Repeat purchase rate
    repeat_pct = (df['total_orders'] > 1).sum() / len(df) * 100
    print(f"  ℹ  Repeat purchase rate: {repeat_pct:.1f}% "
          f"({(df['total_orders'] > 1).sum():,} of {len(df):,} customers)")
    
    # Frequency distribution
    print(f"\n  Order frequency distribution:")
    freq_dist = df['total_orders'].value_counts().sort_index()
    for orders, count in freq_dist.head(10).items():
        pct = count / len(df) * 100
        print(f"    {orders} order(s): {count:>6,} customers ({pct:5.1f}%)")
    if len(freq_dist) > 10:
        remaining = freq_dist.iloc[10:].sum()
        print(f"    {freq_dist.index[10]}+ orders: {remaining:>6,} customers")
    
    # Reference date
    ref_date = df['last_purchase_date'].max()
    print(f"\n  ℹ  Analysis reference date (max purchase): {ref_date}")
    
    # State distribution (top 5)
    print(f"\n  Top 5 states by customer count:")
    state_dist = df['customer_state'].value_counts().head(5)
    for state, count in state_dist.items():
        pct = count / len(df) * 100
        print(f"    {state}: {count:>6,} ({pct:.1f}%)")


def main():
    print("="*70)
    print("COMPONENT 1: Data Ingestion & Analytical Base Table")
    print("="*70)
    
    # Step 1: Validate raw files
    print("\n[1/4] Validating raw data files...")
    validate_raw_files(RAW_DIR, CSV_TABLE_MAP)
    print("  ✓ All required files found")
    
    # Step 2: Load CSVs into SQLite
    print("\n[2/4] Loading CSVs into in-memory SQLite database...")
    conn = sqlite3.connect(':memory:')
    row_counts = load_csvs_to_sqlite(RAW_DIR, CSV_TABLE_MAP, conn)
    print(f"\n  Total raw rows loaded: {sum(row_counts.values()):,}")
    
    # Step 3: Execute SQL pipeline
    print("\n[3/4] Executing SQL pipeline (6 CTEs)...")
    sql_path = os.path.join(SQL_DIR, 'build_base_table.sql')
    customer_base = execute_sql_pipeline(conn, sql_path)
    conn.close()
    print(f"  ✓ Pipeline complete: {len(customer_base):,} customers in base table")
    
    # Step 4: Export and verify
    print("\n[4/4] Exporting to data/processed/customer_base.csv...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, 'customer_base.csv')
    customer_base.to_csv(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")
    
    # Print diagnostics
    print_diagnostics(customer_base, "customer_base.csv")
    
    print(f"\n{'='*70}")
    print("COMPONENT 1 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
