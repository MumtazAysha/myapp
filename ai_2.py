import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Sample dataset (from  database)
data = {
    'shop_id': [1, 2, 3, 4, 5],
    'shop_name': ['Retail A', 'Wholesale B', 'Retail C', 'Wholesale D', 'Retail E'],
    'shop_type': ['retail', 'wholesale', 'retail', 'wholesale', 'retail'],
    'location': ['Nelukkulam', 'Vavuniya', 'Kandy', 'Polonnaruwa', 'Anuradhapura'],
    'milk_packet_price': [25, 20, 26, 19, 24],  # Price per packet
    'min_quantity': [1, 50, 1, 40, 1]  # Minimum quantity for wholesale
}
shops_df = pd.DataFrame(data)

# Linear regression model for price prediction
def train_price_model():
    # Sample data for training (quantity vs price per packet)
    retail_data = pd.DataFrame({
        'quantity': [1, 10, 20, 30, 40],
        'price_per_packet': [25, 24.5, 24, 23.5, 23]
    })
    wholesale_data = pd.DataFrame({
        'quantity': [40, 50, 100, 200, 300],
        'price_per_packet': [20, 19.5, 19, 18.5, 18]
    })

    # Train retail model
    retail_model = LinearRegression()
    retail_model.fit(retail_data[['quantity']], retail_data['price_per_packet'])

    # Train wholesale model
    wholesale_model = LinearRegression()
    wholesale_model.fit(wholesale_data[['quantity']], wholesale_data['price_per_packet'])

    return retail_model, wholesale_model

retail_model, wholesale_model = train_price_model()

# Function to calculate savings
def calculate_savings(quantity, retail_price, wholesale_price, wholesale_min_quantity):
    if quantity >= wholesale_min_quantity:
        wholesale_total = wholesale_price * quantity
        retail_total = retail_price * quantity
        savings = retail_total - wholesale_total
        return savings, wholesale_total
    return 0, 0

# Function to filter nearby wholesale shops
def filter_nearby_wholesalers(user_loc, max_distance_km=5):
    nearby_shops = []
    for _, shop in shops_df[shops_df['shop_type'] == 'wholesale'].iterrows():
        shop_location = (shop['location'])
        user_location = (user_loc)
        distance = geodesic(shop_location, user_location).kilometers
        if distance <= max_distance_km:
            nearby_shops.append({
                'shop_id': shop['shop_id'],
                'shop_name': shop['shop_name'],
                'distance_km': round(distance, 2),
                'milk_packet_price': shop['milk_packet_price'],
                'min_quantity': shop['min_quantity']
            })
    return nearby_shops

# Flask API endpoint
@app.route('/api/check_bulk_order', methods=['POST'])
def check_bulk_order():
    try:
        data = request.get_json()
        quantity = int(data.get('quantity'))
        user_loc = (data.get('location'))
        # Predict prices using linear regression
        retail_price = retail_model.predict(np.array([[quantity]]))[0]
        wholesale_price = wholesale_model.predict(np.array([[quantity]]))[0]

        # Find a suitable wholesaler
        wholesalers = shops_df[shops_df['shop_type'] == 'wholesale']
        if wholesalers.empty:
            return jsonify({'error': 'No wholesalers found'}), 404

        # Use the first wholesaler's min_quantity for calculation
        wholesaler = wholesalers.iloc[0]
        savings, wholesale_total = calculate_savings(
            quantity, retail_price, wholesale_price, wholesaler['min_quantity']
        )

        # Generate notification message
        message = ""
        if savings > 0:
            message = (
                f"This shop sells wholesale items. It is much cheaper to buy {quantity} milk packets "
                f"from a wholesaler than buying {quantity} milk packets from a retailer. "
                f"You can save ₹{savings:.2f}."
            )

        # Filter nearby wholesalers
        nearby_wholesalers = filter_nearby_wholesalers(user_loc)

        return jsonify({
            'message': message,
            'savings': round(savings, 2),
            'wholesale_price': round(wholesale_price, 2),
            'retail_price': round(retail_price, 2),
            'nearby_wholesalers': nearby_wholesalers
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)