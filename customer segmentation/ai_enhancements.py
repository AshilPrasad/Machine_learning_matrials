import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SportsRetailAI:
    def __init__(self, data_path='data/CS_Main.xlsx'):
        """Initialize the AI enhancement module"""
        self.data = pd.read_excel(data_path)
        self.today = pd.to_datetime('today').normalize()
        self.customer_features = self._engineer_customer_features()
        self.product_features = self._engineer_product_features()
        self.scaler = StandardScaler()
        
    def _engineer_customer_features(self):
        data = self.data.copy()
        data['purchase_date'] = pd.to_datetime(data['purchase_date'])
        group = data.groupby('customer_id').agg(
            Monetary=('total_amount', 'sum'),
            total_quantity=('quantity', 'sum'),
            Frequency=('transaction_id', 'count'),
            num_unique_products=('product_id', 'nunique'),
            last_purchase_date=('purchase_date', 'max'),
            avg_price_per_unit=('price_per_unit', 'mean'),
            store_visit_frequency=('purchase_date', 'nunique'),
        ).reset_index()
        membership_start = data.groupby('customer_id')['purchase_date'].min().reset_index()
        membership_start.rename(columns={'purchase_date':'membership_start_date'}, inplace=True)
        group = group.merge(membership_start, on='customer_id', how='left')
        group['days_since_last_purchase'] = (self.today - group['last_purchase_date']).dt.days
        group['membership_duration_months'] = ((self.today - group['membership_start_date']).dt.days / 30).clip(lower=1)
        group['purchase_frequency'] = group['Frequency'] / group['membership_duration_months']
        group['avg_purchase_value'] = group['Monetary'] / group['Frequency']
        return group

    def _engineer_product_features(self):
        data = self.data.copy()
        data['purchase_date'] = pd.to_datetime(data['purchase_date'])
        group = data.groupby('product_id').agg(
            total_quantity_sold=('quantity', 'sum'),
            avg_price_per_unit=('price_per_unit', 'mean'),
            num_customers=('customer_id', 'nunique'),
            first_sold=('purchase_date', 'min'),
            last_sold=('purchase_date', 'max'),
            num_transactions=('transaction_id', 'count'),
        ).reset_index()
        group['demand_level'] = group['total_quantity_sold'] / ((self.today - group['first_sold']).dt.days / 30).clip(lower=1)
        group['seasonality_month'] = group['last_sold'].dt.month
        return group

    def prepare_data(self):
        """Prepare and preprocess the data"""
        # Convert date columns to datetime
        date_columns = self.data.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                self.data[col] = pd.to_datetime(self.data[col])
            except:
                continue
                
        # Handle missing values
        self.data = self.data.fillna(0)
        
        return self.data
    
    def predict_customer_churn(self, customer_id):
        """Predict if a customer is likely to churn"""
        features = ['purchase_frequency', 'avg_purchase_value', 'days_since_last_purchase']
        df = self.customer_features.copy()
        df['churn'] = (df['days_since_last_purchase'] > 90).astype(int)
        X = df[features]
        y = df['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        customer_row = df[df['customer_id'] == int(customer_id)][features]
        if customer_row.empty:
            return "Customer not found"
        churn_prob = model.predict_proba(customer_row)[0][1]
        return {
            'churn_probability': churn_prob,
            'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.3 else 'Low'
        }
    
    def recommend_products(self, customer_id, n_recommendations=5):
        """Generate personalized product recommendations"""
        data = self.data.copy()
        user_item = pd.pivot_table(
            data,
            values='quantity',
            index='customer_id',
            columns='product_id',
            fill_value=0
        )
        if int(customer_id) not in user_item.index:
            return "Customer not found"
        # Find similar customers
        similarities = user_item.corrwith(user_item.loc[int(customer_id)], axis=1).dropna()
        similar_customers = similarities.sort_values(ascending=False)[1:6].index
        # Recommend products bought by similar customers but not by this customer
        customer_products = set(user_item.loc[int(customer_id)][user_item.loc[int(customer_id)] > 0].index)
        recommendations = {}
        for cust in similar_customers:
            for prod, qty in user_item.loc[cust].items():
                if prod not in customer_products and qty > 0:
                    recommendations[prod] = recommendations.get(prod, 0) + qty
        sorted_recs = dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True))
        return dict(list(sorted_recs.items())[:n_recommendations])
    
    def optimize_pricing(self, product_id):
        """Suggest optimal pricing based on demand and competition"""
        df = self.product_features.copy()
        if product_id not in df['product_id'].astype(str).values:
            return "Product not found"
        prod_row = df[df['product_id'] == product_id]
        # Use avg_price_per_unit and demand_level for simple price optimization
        avg_price = prod_row['avg_price_per_unit'].values[0]
        demand = prod_row['demand_level'].values[0]
        # Simulate price elasticity: lower price by 10% increases demand by 5%
        optimal_price = avg_price * 0.9
        expected_demand = demand * 1.05
        return {
            'current_price': avg_price,
            'optimal_price': optimal_price,
            'expected_demand_increase': expected_demand - demand
        }
    
    def forecast_inventory_demand(self, product_id, forecast_days=30):
        """Forecast inventory demand for the next 30 days"""
        data = self.data.copy()
        data['purchase_date'] = pd.to_datetime(data['purchase_date'])
        prod_data = data[data['product_id'] == product_id]
        if prod_data.empty:
            return "Product not found"
        daily_sales = prod_data.groupby('purchase_date')['quantity'].sum()
        if len(daily_sales) < 2:
            return "Not enough data to forecast"
        # Simple moving average forecast
        avg_daily = daily_sales.mean()
        forecast = [avg_daily] * forecast_days
        future_dates = pd.date_range(start=daily_sales.index.max() + pd.Timedelta(days=1), periods=forecast_days)
        return {
            'forecast_dates': [str(d) for d in future_dates],
            'predicted_demand': forecast,
            'total_forecasted_demand': sum(forecast)
        }
    
    def predict_customer_lifetime_value(self, customer_id):
        """Predict Customer Lifetime Value (CLV)"""
        df = self.customer_features.copy()
        row = df[df['customer_id'] == int(customer_id)]
        if row.empty:
            return "Customer not found"
        # Simple CLV: Monetary value * (purchase_frequency / (1 + churn probability))
        churn_prob = self.predict_customer_churn(customer_id)['churn_probability']
        clv = row['Monetary'].values[0] * (row['purchase_frequency'].values[0] / (1 + churn_prob))
        return {
            'current_clv': row['Monetary'].values[0],
            'predicted_future_clv': clv,
            'total_predicted_clv': row['Monetary'].values[0] + clv
        }

# Example usage
if __name__ == "__main__":
    ai = SportsRetailAI()
    print(ai.customer_features.head())
    print(ai.product_features.head())
    print(ai.predict_customer_churn(181))
    print(ai.recommend_products(181))
    print(ai.optimize_pricing('P002'))
    print(ai.forecast_inventory_demand('P002'))
    print(ai.predict_customer_lifetime_value(181)) 