-- =============================================================================
-- LumiSkin CLV & RFM Segmentation — Analytical Base Table Construction
-- =============================================================================
-- This SQL pipeline transforms the raw Olist e-commerce dataset into a single
-- customer-level analytical base table suitable for RFM scoring and CLV modeling.
--
-- CRITICAL DATA WARNING:
-- The Olist schema has TWO customer identifiers:
--   • customer_id       → unique per ORDER (same person gets a new ID per order)
--   • customer_unique_id → stable identifier for an individual across all orders
-- ALL customer-level aggregations in this pipeline use customer_unique_id.
-- Using customer_id would treat repeat buyers as separate customers, invalidating
-- the entire RFM and CLV analysis.
--
-- Executed via: analysis/build_base_table.py (Python sqlite3)
-- Output:       data/processed/customer_base.csv
-- =============================================================================

WITH

-- ---------------------------------------------------------------------------
-- CTE 1: filtered_orders
-- ---------------------------------------------------------------------------
filtered_orders AS (
    SELECT
        order_id,
        customer_id,
        order_purchase_timestamp,
        order_delivered_customer_date
    FROM orders
    WHERE order_status = 'delivered'
      AND order_delivered_customer_date IS NOT NULL
),

-- ---------------------------------------------------------------------------
-- CTE 2: payments_aggregated
-- Resolves the installment duplication problem. The order_payments table has
-- one row per installment — SUM to get true order total.
-- ---------------------------------------------------------------------------
payments_aggregated AS (
    SELECT
        order_id,
        SUM(payment_value) AS total_payment_value,
        COUNT(*) AS num_installments
    FROM order_payments
    GROUP BY order_id
),

-- ---------------------------------------------------------------------------
-- CTE 3: items_aggregated
-- Compute item_count per order and identify modal product category (English).
-- Also captures the dominant seller_state for logistics analysis.
-- Uses GROUP BY + subquery approach (SQLite doesn't support nested window fns).
-- ---------------------------------------------------------------------------
order_item_counts AS (
    SELECT order_id, COUNT(*) AS item_count
    FROM order_items
    GROUP BY order_id
),

-- Category frequency per order, ranked to find the mode
order_category_ranked AS (
    SELECT
        oi.order_id,
        COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
        COUNT(*) AS cat_count
    FROM order_items oi
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
    GROUP BY oi.order_id, COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')
),

order_modal_category AS (
    SELECT order_id, category AS primary_category
    FROM (
        SELECT
            order_id,
            category,
            ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY cat_count DESC, category ASC) AS rn
        FROM order_category_ranked
    )
    WHERE rn = 1
),

-- Dominant seller state per order (for logistics analysis)
order_seller_ranked AS (
    SELECT
        oi.order_id,
        s.seller_state,
        COUNT(*) AS seller_count
    FROM order_items oi
    LEFT JOIN sellers s ON oi.seller_id = s.seller_id
    WHERE s.seller_state IS NOT NULL
    GROUP BY oi.order_id, s.seller_state
),

order_modal_seller AS (
    SELECT order_id, seller_state
    FROM (
        SELECT
            order_id,
            seller_state,
            ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY seller_count DESC, seller_state ASC) AS rn
        FROM order_seller_ranked
    )
    WHERE rn = 1
),

-- ---------------------------------------------------------------------------
-- CTE 4: reviews_aggregated
-- Most recent review per order (handles duplicates).
-- ---------------------------------------------------------------------------
reviews_aggregated AS (
    SELECT order_id, review_score
    FROM (
        SELECT
            order_id,
            review_score,
            ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_answer_timestamp DESC) AS rn
        FROM order_reviews
    )
    WHERE rn = 1
),

-- ---------------------------------------------------------------------------
-- CTE 5: customer_orders
-- Join all CTEs. Maps customer_id → customer_unique_id via the customers table.
-- CRITICAL: customer_unique_id is the stable person-level identifier.
-- ---------------------------------------------------------------------------
customer_orders AS (
    SELECT
        c.customer_unique_id,
        c.customer_state,
        fo.order_id,
        fo.order_purchase_timestamp,
        COALESCE(pa.total_payment_value, 0) AS total_payment_value,
        COALESCE(pa.num_installments, 1) AS num_installments,
        COALESCE(oic.item_count, 0) AS item_count,
        COALESCE(omc.primary_category, 'unknown') AS primary_category,
        oms.seller_state,
        ra.review_score
    FROM filtered_orders fo
    INNER JOIN customers c ON fo.customer_id = c.customer_id
    LEFT JOIN payments_aggregated pa ON fo.order_id = pa.order_id
    LEFT JOIN order_item_counts oic ON fo.order_id = oic.order_id
    LEFT JOIN order_modal_category omc ON fo.order_id = omc.order_id
    LEFT JOIN order_modal_seller oms ON fo.order_id = oms.order_id
    LEFT JOIN reviews_aggregated ra ON fo.order_id = ra.order_id
),

-- ---------------------------------------------------------------------------
-- CTE 6: customer_aggregated
-- Collapse to one row per customer_unique_id. Reference date = MAX purchase
-- timestamp in the dataset (not today) for reproducibility.
-- ---------------------------------------------------------------------------
ref_date AS (
    SELECT MAX(order_purchase_timestamp) AS max_date FROM customer_orders
),

customer_modal_category AS (
    SELECT customer_unique_id, primary_category
    FROM (
        SELECT
            customer_unique_id,
            primary_category,
            ROW_NUMBER() OVER (
                PARTITION BY customer_unique_id
                ORDER BY COUNT(*) DESC, primary_category ASC
            ) AS rn
        FROM customer_orders
        GROUP BY customer_unique_id, primary_category
    )
    WHERE rn = 1
),

customer_modal_seller_state AS (
    SELECT customer_unique_id, seller_state AS primary_seller_state
    FROM (
        SELECT
            customer_unique_id,
            seller_state,
            ROW_NUMBER() OVER (
                PARTITION BY customer_unique_id
                ORDER BY COUNT(*) DESC, seller_state ASC
            ) AS rn
        FROM customer_orders
        WHERE seller_state IS NOT NULL
        GROUP BY customer_unique_id, seller_state
    )
    WHERE rn = 1
),

customer_latest_state AS (
    SELECT customer_unique_id, customer_state
    FROM (
        SELECT
            customer_unique_id,
            customer_state,
            ROW_NUMBER() OVER (
                PARTITION BY customer_unique_id
                ORDER BY order_purchase_timestamp DESC
            ) AS rn
        FROM customer_orders
    )
    WHERE rn = 1
),

customer_aggregated AS (
    SELECT
        co.customer_unique_id,
        cls.customer_state,
        MIN(co.order_purchase_timestamp) AS first_purchase_date,
        MAX(co.order_purchase_timestamp) AS last_purchase_date,
        COUNT(DISTINCT co.order_id) AS total_orders,
        SUM(co.total_payment_value) AS total_spend,
        AVG(co.total_payment_value) AS avg_order_value,
        AVG(co.review_score) AS avg_review_score,
        cmc.primary_category,
        cmss.primary_seller_state,
        CAST(
            julianday(rd.max_date) - julianday(MAX(co.order_purchase_timestamp))
        AS INTEGER) AS recency_days,
        CAST(
            julianday(MAX(co.order_purchase_timestamp))
            - julianday(MIN(co.order_purchase_timestamp))
        AS INTEGER) AS customer_tenure_days
    FROM customer_orders co
    CROSS JOIN ref_date rd
    LEFT JOIN customer_latest_state cls ON co.customer_unique_id = cls.customer_unique_id
    LEFT JOIN customer_modal_category cmc ON co.customer_unique_id = cmc.customer_unique_id
    LEFT JOIN customer_modal_seller_state cmss ON co.customer_unique_id = cmss.customer_unique_id
    GROUP BY co.customer_unique_id
)

SELECT
    customer_unique_id,
    customer_state,
    first_purchase_date,
    last_purchase_date,
    total_orders,
    total_spend,
    avg_order_value,
    avg_review_score,
    primary_category,
    primary_seller_state,
    recency_days,
    customer_tenure_days
FROM customer_aggregated
ORDER BY total_spend DESC;
