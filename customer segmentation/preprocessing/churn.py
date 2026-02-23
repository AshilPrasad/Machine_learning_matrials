
from preprocessing.preprocessing import preprocess_customer_d1
import pickle
from tensorflow.keras.models import load_model

scaler = pickle.load(open('Models\churn_scaler.pkl', 'rb'))
model=load_model('Models\churn_model.h5')


def churn_prediction(input):

    data=preprocess_customer_d1(input)
    data=data[['Monetary','Frequency','Avg_purchase_gap_days','Recency']] 

    # Step 2: Scale the features
    data_scaled = scaler.transform(data)

    # Step 3: Predict probabilities
    prediction = model.predict(data_scaled)

    # Step 4: Generate churn labels (0 or 1)
    churn_labels = (prediction[:, 0] > 0.5).astype(int)

    # Step 5: Assign risk level based on probabilities
    risk_levels = ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in prediction[:, 0]]

    # Step 6: Attach results to the input DataFrame
    input['churn_prediction'] = churn_labels
    input['prediction_probability'] = prediction[:, 0]
    input['risk_level'] = risk_levels
    

    return input
