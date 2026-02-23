from flask import Flask, render_template, request, jsonify
import pandas as pd
from ai_enhancements import SportsRetailAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Initialize AI module
ai = SportsRetailAI()

@app.route('/')
def index():
    """Render the AI dashboard"""
    return render_template('ai_dashboard.html')

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized product recommendations for a customer"""
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        n_recommendations = data.get('n_recommendations', 5)
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400
            
        recommendations = ai.recommend_products(customer_id, n_recommendations)
        return jsonify({
            'status': 'success',
            'customer_id': customer_id,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/churn-risk', methods=['POST'])
def get_churn_risk():
    """Get churn risk prediction for a customer"""
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400
            
        churn_prediction = ai.predict_customer_churn(customer_id)
        return jsonify({
            'status': 'success',
            'customer_id': customer_id,
            'churn_prediction': churn_prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize-price', methods=['POST'])
def get_optimal_price():
    """Get optimal pricing suggestions for a product"""
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        
        if not product_id:
            return jsonify({'error': 'Product ID is required'}), 400
            
        pricing = ai.optimize_pricing(product_id)
        return jsonify({
            'status': 'success',
            'product_id': product_id,
            'pricing': pricing
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory-forecast', methods=['POST'])
def get_inventory_forecast():
    """Get inventory demand forecast for a product"""
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        forecast_days = data.get('forecast_days', 30)
        
        if not product_id:
            return jsonify({'error': 'Product ID is required'}), 400
            
        forecast = ai.forecast_inventory_demand(product_id, forecast_days)
        return jsonify({
            'status': 'success',
            'product_id': product_id,
            'forecast': forecast
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customer-value', methods=['POST'])
def get_customer_value():
    """Get customer lifetime value prediction"""
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400
            
        clv = ai.predict_customer_lifetime_value(customer_id)
        return jsonify({
            'status': 'success',
            'customer_id': customer_id,
            'customer_value': clv
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-analysis', methods=['POST'])
def batch_analysis():
    """Perform batch analysis on multiple customers or products"""
    try:
        data = request.get_json()
        analysis_type = data.get('type')  # 'customers' or 'products'
        ids = data.get('ids', [])
        
        if not analysis_type or not ids:
            return jsonify({'error': 'Analysis type and IDs are required'}), 400
            
        results = {}
        if analysis_type == 'customers':
            for customer_id in ids:
                churn = ai.predict_customer_churn(customer_id)
                clv = ai.predict_customer_lifetime_value(customer_id)
                recs = ai.recommend_products(customer_id)
                results[customer_id] = {
                    'churn_risk': churn if isinstance(churn, dict) else {'error': churn},
                    'lifetime_value': clv if isinstance(clv, dict) else {'error': clv},
                    'recommendations': recs if isinstance(recs, dict) else {'error': recs}
                }
        elif analysis_type == 'products':
            for product_id in ids:
                pricing = ai.optimize_pricing(product_id)
                forecast = ai.forecast_inventory_demand(product_id)
                results[product_id] = {
                    'pricing': pricing if isinstance(pricing, dict) else {'error': pricing},
                    'inventory_forecast': forecast if isinstance(forecast, dict) else {'error': forecast}
                }
        else:
            return jsonify({'error': 'Invalid analysis type'}), 400
            
        return jsonify({
            'status': 'success',
            'analysis_type': analysis_type,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Running on a different port than the main app 