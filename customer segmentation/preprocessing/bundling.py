import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Step 1: Identify dead stock items
def get_dead_stock_items(stock_data, threshold=0.6):
    stock_data['sold_ratio'] = stock_data['total_sold'] / stock_data['initial_stock']
    dead_stock = stock_data[stock_data['sold_ratio'] < threshold]['product_name'].tolist()
    return dead_stock

# Step 2: Preprocess transaction data into basket format
def preprocess_basket_data(data):
    basket_data = data.groupby('transaction_id')['product_name'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(basket_data).transform(basket_data)
    basket = pd.DataFrame(te_ary, columns=te.columns_)
    return basket

# Step 3: Generate frequent itemsets using Apriori algorithm
def generate_frequent_itemsets(basket, min_support=0.1):
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    return frequent_itemsets

# Step 4: Generate association rules and filter by metrics
def generate_association_rules(frequent_itemsets, metric="lift", min_threshold=1, min_confidence=0.05, min_support=0.05):
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    rules = rules[(rules['confidence'] >= min_confidence) & (rules['support'] >= min_support)]
    bundling = rules.sort_values(by='confidence', ascending=False)
    return bundling

# Step 5: Keep rules where the consequents contain dead stock items
def filter_rules_for_dead_stock(bundling, dead_stock):
    bundling['consequents'] = bundling['consequents'].apply(lambda x: [p for p in x if p in dead_stock])
    bundling = bundling[bundling['consequents'].map(len) > 0]
    bundling = bundling[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    return bundling

# Step 6: Select best rule (highest confidence) per antecedent
def best_recommend(bundling):
    idx = bundling.groupby('antecedents')['confidence'].idxmax()
    bundling = bundling.loc[idx].reset_index(drop=True)
    return bundling


def recommend_dead_stock_products(input_products, input_df, stock_path):
    # Load data
    df = input_df
    stock_data = pd.read_excel(stock_path)

    # Dead stock detection
    dead_stock_items = get_dead_stock_items(stock_data)

    # Basket creation and association rule mining
    basket = preprocess_basket_data(df)
    frequent_itemsets = generate_frequent_itemsets(basket)

    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets found. Try with more data or different product combinations.")

    bundling = generate_association_rules(frequent_itemsets)

    if bundling.empty:
        raise ValueError("No association rules found. Consider checking the transaction pattern.")

    bundling_rules = filter_rules_for_dead_stock(bundling, dead_stock_items)

    if bundling_rules.empty:
        raise ValueError("No bundling rules involve dead stock items.")

    best_bundling = best_recommend(bundling_rules)

    # Find matched rules for the given input product(s)
    matched_rules = best_bundling[best_bundling['antecedents'] == frozenset(input_products)]

    if matched_rules.empty:
        raise ValueError(f"No matching bundle recommendations found for: {input_products}")

    # Extract recommended dead stock products
    recommended_dead_stock = set()
    for _, row in matched_rules.iterrows():
        recommended_dead_stock.update(row['consequents'])

    return list(recommended_dead_stock)
