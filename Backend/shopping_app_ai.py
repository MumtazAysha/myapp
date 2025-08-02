import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Sample dataset with location as place names
data = {
    'shop_id': [1, 2, 3, 4, 5],
    'shop_name': ['Retail A', 'Wholesale B', 'Retail C', 'Wholesale D', 'Retail E'],
    'shop_type': ['retail', 'wholesale', 'retail', 'wholesale', 'retail'],
    'location': ['Kandy', 'Horana', 'Vavuniya', 'Jaffna', 'Colombo'],
    # Price per packet in LKR
    'min_quantity': [1, 50, 1, 40, 1],  # Minimum quantity for wholesale
    'item':['Milk packet','Soap','Chocolate','Biscuit','Napkins']
}
shops_df = pd.DataFrame(data)

# Linear regression model for price prediction
def train_price_model(item):
    #if item != 'milk_packet':
       # raise ValueError(f"Unsupported item: {item}")
    
    retail_data = pd.DataFrame({
        'quantity': [1, 10, 20, 30, 40],
        'price_per_packet': [250, 245, 240, 235, 230]
    })
    wholesale_data = pd.DataFrame({
        'quantity': [40, 50, 100, 200, 300],
        'price_per_packet': [200, 195, 190, 185, 180]
    })

    retail_model = LinearRegression()
    retail_model.fit(retail_data[['quantity']], retail_data['price_per_packet'])

    wholesale_model = LinearRegression()
    wholesale_model.fit(wholesale_data[['quantity']], wholesale_data['price_per_packet'])

    return retail_model, wholesale_model

# Function to calculate savings
def calculate_savings(quantity, retail_price, wholesale_price, wholesale_min_quantity):
    if quantity >= wholesale_min_quantity:
        wholesale_total = wholesale_price * quantity
        retail_total = retail_price * quantity
        savings = retail_total - wholesale_total
        return savings, wholesale_total
    return 0, 0

# Function to filter nearby wholesale shops
def filter_nearby_wholesalers(user_location, item):
    if item != 'milk_packet':
        return []
    
    nearby_shops = []
    for _, shop in shops_df[(shops_df['shop_type'] == 'wholesale') & (shops_df['location'] == user_location)].iterrows():
        nearby_shops.append({
            'shop_id': shop['shop_id'],
            'shop_name': shop['shop_name'],
            'milk_packet_price': shop['milk_packet_price'],
            'min_quantity': shop['min_quantity'],
            'location': shop['location']
        })
    return nearby_shops

# Flask API endpoint
@app.route('/api/check_bulk_order', methods=['POST'])
def check_bulk_order():
    try:
        data = request.get_json()
        quantity = int(data.get('quantity'))
        item = data.get('item')
        location = data.get('location')

        if not all([quantity, item, location]):
            return jsonify({'error': 'Quantity, item, and location are required'}), 400

        # Train price model for the item
        retail_model, wholesale_model = train_price_model(item)

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
                f"This shop sells wholesale {item}s. It is much cheaper to buy {quantity} {item}s "
                f"from a wholesaler than buying {quantity} {item}s from a retailer. "
                f"You can save LKR {savings:.2f}."
            )

        # Filter nearby wholesalers
        nearby_wholesalers = filter_nearby_wholesalers(location, item)

        return jsonify({
            'message': message,
            'savings': round(savings, 2),
            'wholesale_price': round(wholesale_price, 2),
            'retail_price': round(retail_price, 2),
            'nearby_wholesalers': nearby_wholesalers
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)