# preprocesser/process.py

import pandas as pd
from datetime import datetime

def preprocess_customer_d1(df):
    df['purchase_date'] = pd.to_datetime(df['purchase_date']).dt.date
    df['transaction_id'] = df['customer_id'].astype(str)

    d1 = df.groupby('customer_id').agg(
        Monetary=('total_amount', 'sum'),
        total_quantity=('quantity', 'sum'),
        Frequency=('transaction_id', 'count'),
        num_unique_products=('product_id', 'nunique'),
        last_purchase_date=('purchase_date', 'max'),
        avg_price_per_unit=('price_per_unit', 'mean'),
        store_visit_frequency=('purchase_date', 'nunique'),
        Mobile = ('Mobile','first')
    ).reset_index()

    membership_start = df.groupby('customer_id')['purchase_date'].min().reset_index()
    membership_start.rename(columns={'purchase_date': 'membership_start_date'}, inplace=True)
    d1 = d1.merge(membership_start, on='customer_id', how='left')

    reference_date = pd.to_datetime(d1['last_purchase_date'].max())
    today = pd.to_datetime(datetime.today().date())

    d1['membership_start_date'] = pd.to_datetime(d1['membership_start_date'])
    d1['last_purchase_date'] = pd.to_datetime(d1['last_purchase_date'])

    d1['Active_days'] = ((reference_date - d1['membership_start_date']).dt.days).round().astype(int)
    d1['Avg_purchase_gap_days'] = d1.apply(
        lambda x: x['Active_days'] / x['store_visit_frequency']
        if x['store_visit_frequency'] > 0 else x['Active_days'], axis=1)

    d1['Recency'] = (today - d1['last_purchase_date']).dt.days
    # d1['churn'] = d1.apply(lambda row: 1 if (
    #                                          (row['Recency'] > 1 * row['Avg_purchase_gap_days']) and  # More lenient recency threshold
    #                                          (row['Monetary'] < d1['Monetary'].median()) and       # Low monetary value
    #                                          (row['Frequency'] < d1['Frequency'].median())) else 0, axis=1)

    return d1



FREQ_THRESHOLD = 15
MONETARY_THRESHOLD = 30000

def apply_reward_rules(row, freq_threshold=FREQ_THRESHOLD, monetary_threshold=MONETARY_THRESHOLD):
    if row['Frequency'] >= freq_threshold or row['Monetary'] >= monetary_threshold:
        loyalty_tier = row.get('loyalty', 'N/A')
        assigned_reward = row.get('assigned_reward', 'N/A')
        msg = (f"🎊 Congrats Customer {row['customer_id']}! As a {loyalty_tier} member, "
               f"enjoy your reward: {assigned_reward}. We appreciate your loyalty! 💖")
    else:
        loyalty_tier = "No tier"
        assigned_reward = "No reward"
        purchase_gap = max(0, freq_threshold - row['Frequency'])
        money_gap = max(0, monetary_threshold - row['Monetary'])
        msg = (f"⚠️ Hi Customer {row['customer_id']}! You currently have no loyalty tier. "
               f"Make {purchase_gap} more purchases or spend ₹{money_gap:.0f} more to unlock exciting rewards! 🚀")
    
    return pd.Series([loyalty_tier, assigned_reward, msg])



def process_customer_d1frame(input_df, model, scaler):
    d1 = preprocess_customer_d1(input_df)

    features = ['Monetary', 'Frequency', 'Recency', 'Active_days',
                'total_quantity', 'avg_price_per_unit',
                'store_visit_frequency', 'Avg_purchase_gap_days']

    scaled_d1 = scaler.transform(d1[features])
    d1['cluster'] = model.predict(scaled_d1)

    agg = d1.groupby('cluster').agg({'Frequency': 'sum', 'Monetary': 'sum'}).reset_index()
    agg['Unit Price'] = agg['Monetary'] / agg['Frequency']
    agg = agg.sort_values('Unit Price', ascending=False).reset_index(drop=True)

    loyalty_labels = ['Platinum', 'Gold', 'Silver', 'Bronze']
    agg['loyalty'] = loyalty_labels[:len(agg)]

    reward_mapping = {
        'Platinum': '25% discount + VIP concierge access',
        'Gold': '20% discount + free shipping',
        'Silver': '15% discount or birthday bonus',
        'Bronze': 'Points-based rewards or 10% discount'
    }
    agg['assigned_reward'] = agg['loyalty'].map(reward_mapping)

    final = d1.merge(agg[['cluster', 'loyalty', 'assigned_reward']], on='cluster', how='left')

    final[['loyalty', 'assigned_reward', 'progress_message']] = final.apply(apply_reward_rules, axis=1)

    return final








