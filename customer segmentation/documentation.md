Customer Analytics Platform Documentation

Overview
This documentation covers two main components of the Customer Analytics Platform:
1. Customer Segmentation System
2. Churn Prediction System

## 1. Customer Segmentation System

### Technical Implementation

#### Core Components
- **Main Application**: `app.py`
- **Preprocessing**: `preprocessing/preprocessing.py`
- **Product Bundling**: `preprocessing/bundling.py`
- **Models**: 
  - Customer Segmentation Model: `Models/CS_model.pkl`
  - Scaler: `Models/CS_scalers.pkl`

#### Data Processing Pipeline
1. **Data Preprocessing** (`preprocess_customer_d1`):
   - Converts purchase dates to datetime format
   - Calculates key metrics:
     - Monetary value (total spending)
     - Frequency (number of transactions)
     - Recency (days since last purchase)
     - Active days (customer tenure)
     - Average purchase gap
     - Store visit frequency
     - Number of unique products purchased

2. **Customer Segmentation** (`process_customer_d1frame`):
   - Applies ML model to segment customers
   - Assigns loyalty tiers:
     - Platinum
     - Gold
     - Silver
     - Bronze
   - Generates personalized rewards based on segments

3. **Product Bundling** (`recommend_dead_stock_products`):
   - Identifies dead stock items (sold ratio < 60%)
   - Uses Apriori algorithm for association rule mining
   - Generates product recommendations based on:
     - Support (frequency of item sets)
     - Confidence (conditional probability)
     - Lift (correlation strength)

### Business Implementation

#### Customer Segments and Rewards
1. **Platinum Customers**
   - Highest value customers
   - Rewards: 25% discount + VIP concierge access
   - Business Impact: Retention of high-value customers

2. **Gold Customers**
   - High-value regular customers
   - Rewards: 20% discount + free shipping
   - Business Impact: Encourages continued loyalty

3. **Silver Customers**
   - Moderate-value customers
   - Rewards: 15% discount or birthday bonus
   - Business Impact: Incentivizes increased engagement

4. **Bronze Customers**
   - New or low-value customers
   - Rewards: Points-based rewards or 10% discount
   - Business Impact: Encourages repeat purchases

#### Product Bundling Strategy
- **Dead Stock Management**:
  - Identifies slow-moving inventory
  - Creates bundles with popular products
  - Reduces inventory holding costs
  - Improves cash flow

- **Bundle Generation Parameters**:
  - Minimum support: 0.1 (10% of transactions)
  - Minimum confidence: 0.05 (5% conditional probability)
  - Minimum lift: 1.0 (positive correlation)

## 2. Churn Prediction System

### Technical Implementation

#### Core Components
- **Main Application**: `churn_app.py`
- **Preprocessing**: `preprocessing/churn.py`
- **Model**: Churn prediction model (pickle file)

#### Data Processing Pipeline
1. **Feature Engineering**:
   - Recency (days since last purchase)
   - Frequency (number of transactions)
   - Monetary value (total spending)
   - Average purchase gap
   - Store visit frequency
   - Active days

2. **Churn Definition**:
   - Based on multiple factors:
     - Recency > Average purchase gap
     - Monetary value below median
     - Frequency below median

3. **Risk Assessment**:
   - High Risk: Probability > 0.7
   - Medium Risk: Probability 0.3-0.7
   - Low Risk: Probability < 0.3

### Business Implementation

#### Churn Prevention Strategy
1. **High-Risk Customers**
   - Immediate intervention required
   - Personalized retention offers
   - Direct communication via SMS
   - Special discounts or incentives

2. **Medium-Risk Customers**
   - Regular engagement
   - Targeted marketing campaigns
   - Loyalty program benefits
   - Product recommendations

3. **Low-Risk Customers**
   - Standard engagement
   - Regular communication
   - Basic loyalty benefits
   - Cross-selling opportunities

#### Business Impact
- Reduced customer churn
- Improved customer retention
- Better resource allocation
- Targeted marketing effectiveness
- Increased customer lifetime value

## System Integration

### Communication System
- **SMS Integration**:
  - Uses Twilio API
  - Automated notifications
  - Personalized messages
  - Real-time delivery

### Data Flow
1. Customer data input
2. Preprocessing and feature engineering
3. Model prediction
4. Action generation
5. Communication delivery

### Security
- Environment variable management
- Secure API key storage
- Data encryption
- Access control

## Advantages

### Customer Segmentation
1. **Personalized Marketing**
   - Targeted campaigns
   - Relevant offers
   - Better customer experience

2. **Resource Optimization**
   - Efficient allocation of marketing budget
   - Focused customer service
   - Optimized inventory management

3. **Revenue Growth**
   - Increased customer lifetime value
   - Better cross-selling opportunities
   - Improved retention rates

### Churn Prediction
1. **Proactive Retention**
   - Early warning system
   - Timely intervention
   - Reduced customer loss

2. **Cost Efficiency**
   - Targeted retention efforts
   - Optimized marketing spend
   - Better resource allocation

3. **Business Intelligence**
   - Customer behavior insights
   - Market trend analysis
   - Strategic decision support

## Technical Requirements

### Dependencies
- Python 3.x
- Flask
- Pandas
- Scikit-learn
- Twilio
- MLxtend
- Pickle

### Environment Variables
- SECRET_KEY
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_PHONE_NUMBER

### File Structure
```
project/
├── app.py
├── churn_app.py
├── Models/
│   ├── CS_model.pkl
│   └── CS_scalers.pkl
├── preprocessing/
│   ├── preprocessing.py
│   ├── bundling.py
│   └── churn.py
└── data/
    ├── customer_segmention.csv
    └── stock_data2.xlsx
``` 
