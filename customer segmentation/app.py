import os
import re
import pickle
import threading
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file, jsonify
from tensorflow.keras.models import load_model
from preprocessing.preprocessing import process_customer_d1frame, preprocess_customer_d1
from preprocessing.bundling import recommend_dead_stock_products
from twilio.rest import Client

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable not set. Please set it in .env file")

# Load ML models and scalers
model = pickle.load(open('Models/CS_model.pkl', 'rb'))
scaler = pickle.load(open('Models/CS_scalers.pkl', 'rb'))
churn_model = load_model('Models/churn_model.h5')
churn_scaler = pickle.load(open('Models/churn_scaler.pkl', 'rb'))

# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
    raise RuntimeError("Twilio credentials not set in environment variables")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Paths
OUTPUT_FILE = 'data/customer_segmention.csv'
STOCK_FILE = 'data/stock_data2.xlsx'
global_input_df = None


# Utility functions
def is_valid_number(number):
    return re.match(r'^\+?\d{10,15}$', number)


def send_sms(to_number, message, mock=True):
    if not is_valid_number(to_number):
        print(f"❌ Invalid number: {to_number}")
        return False

    if mock:
        print(f"[MOCK SMS] To: {to_number} | Message: {message}")
        return True

    try:
        msg = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        print(f"✅ Sent SMS: {msg.sid}")
        return True
    except Exception as e:
        print(f"❌ Error sending SMS to {to_number}: {e}")
        return False


def churn_prediction(input_df, model, scaler):
    data = preprocess_customer_d1(input_df)
    data = data[['customer_id', 'Monetary', 'Frequency', 'Avg_purchase_gap_days', 'Recency']]
    features = data[['Monetary', 'Frequency', 'Avg_purchase_gap_days', 'Recency']]
    scaled = scaler.transform(features)
    pred_probs = model.predict(scaled)
    data['churn_prediction'] = (pred_probs[:, 0] > 0.5).astype(int)
    data['prediction_probability'] = pred_probs[:, 0]
    data['risk_level'] = ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in pred_probs[:, 0]]
    return data


@app.route('/', methods=['GET', 'POST'])
def index():
    global global_input_df

    table_data = None
    show_download = False
    mock_mode = True
    bundling_results = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.lower().endswith(('.xlsx', '.xls', '.csv')):
                try:
                    if file.filename.lower().endswith('.csv'):
                        global_input_df = pd.read_csv(file)
                    else:
                        global_input_df = pd.read_excel(file)

                    final = process_customer_d1frame(global_input_df, model=model, scaler=scaler)

                    # Churn prediction
                    churn_results = churn_prediction(global_input_df.copy(), churn_model, churn_scaler)
                    final = pd.merge(final, churn_results[['customer_id', 'churn_prediction', 'prediction_probability', 'risk_level']],
                                     on='customer_id', how='left')

                    # Send SMS
                    no_reward_customers = final[final['assigned_reward'] == 'No reward']
                    for _, row in no_reward_customers.head(3).iterrows():
                        mobile = str(row['Mobile'])
                        message = row['progress_message']
                        threading.Thread(
                            target=send_sms,
                            args=(mobile, message, mock_mode)
                        ).start()

                    final.to_csv(OUTPUT_FILE, index=False)
                    table_data = final.to_html(classes='table table-striped', index=False)
                    show_download = True

                except Exception as e:
                    return f"❌ Error processing file: {str(e)}"

        elif 'product' in request.form:
            product = request.form['product']
            if global_input_df is None:
                return "❌ Please upload a customer file before requesting bundling."
            try:
                recommended_products = recommend_dead_stock_products(
                    [product],
                    global_input_df,
                    STOCK_FILE
                )
                bundling_results = {
                    'input_product': product,
                    'recommended_products': recommended_products
                }
            except Exception as e:
                return f"❌ Error processing product bundling: {str(e)}"

    product_list = global_input_df['product_name'].unique().tolist() if global_input_df is not None else []
    return render_template('index.html',
                           table=table_data,
                           show_download=show_download,
                           bundling_results=bundling_results,
                           product_list=product_list)


@app.route('/download_csv')
def download_csv():
    if os.path.exists(OUTPUT_FILE):
        return send_file(OUTPUT_FILE, as_attachment=True)
    return "⚠️ No processed file found. Please upload a file first."


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
