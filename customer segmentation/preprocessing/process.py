# preprocesser/process.py

import pandas as pd
from datetime import datetime

def preprocess_customer_data(df):
    
    df['membership_start_date'] = pd.to_datetime(df['membership_start_date'], dayfirst=True)
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], dayfirst=True)

    # Clean string columns
    df['product_preference'] = df['product_preference'].str.lower().str.strip()
    df['product_segment'] = df['product_segment'].str.lower().str.strip()

    # Reference date
    reference_date = df['last_purchase_date'].max()

    # Feature engineering
    df['Active_days'] = (reference_date - df['membership_start_date']).dt.days.round().astype(int)

    df['Avg_purchase_gap_days'] = df.apply(
        lambda x: x['Active_days'] / x['store_visit_frequency'] if x['store_visit_frequency'] > 0 else x['Active_days'],
        axis=1
    )

    df['Frequency'] = (df['purchase_frequency_per_month'] * 
                       ((df['last_purchase_date'] - df['membership_start_date']).dt.days / 30.0)).round().astype(int)

    df['Monetary'] = df['Frequency'] * df['average_purchase_value']
    df['Recency'] = (datetime.today() - df['last_purchase_date']).dt.days

    # Rename for consistency
    df.rename(columns={'average_purchase_value': 'Avg_monetary', 
                       'product_segment': 'Product_category'}, inplace=True)

    selected = df[['customer_id','Mobile','product_preference', 'store_visit_frequency', 
                   'Active_days', 'days_since_last_visit', 'Avg_purchase_gap_days',
                   'Recency', 'Monetary', 'Frequency', 'Avg_monetary']]
    

    return selected

def apply_reward_rules(row, freq_threshold=5, monetary_threshold=5000):
    if row['Frequency'] >= freq_threshold or row['Monetary'] >= monetary_threshold:
        return pd.Series([row.get('loyalty', 'N/A'), row.get('assigned_reward', 'N/A'), "✅ Eligible for reward"])
    else:
        purchase_gap = max(0, freq_threshold - row['Frequency'])
        money_gap = max(0, monetary_threshold - row['Monetary'])
        msg = f"🔔 You need {purchase_gap} more purchases or ₹{money_gap:.0f} more to earn a reward."
        return pd.Series(['No reward', 'No reward', msg])

