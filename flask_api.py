from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

def pre_processing(test_point):
    filtered_columns = ['battery_power','blue','dual_sim','fc','four_g','int_memory','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time','three_g','touch_screen','wifi']
    indices_to_keep = [0, 1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    new_arr = test_point[:,indices_to_keep]
    print(new_arr.shape)

    # Load the scaler model from the pickle file
    with open('scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)

    # Apply standardization to your DataFrame
    test_standardized = scaler.fit_transform(new_arr)
    # Convert the standardized array back to a DataFrame
    test_standardized = pd.DataFrame(test_standardized,columns= filtered_columns)

    return test_standardized


# Step 1: Load the model
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

# Step 2: Create Flask app
app = Flask(__name__)

# Step 3: Define API endpoint
@app.route('/predict', methods=['GET'])
def predict():
    # Get input data from query parameters
    feature_str = request.args.get('features')
    
    # Convert input data to numpy array
    features = np.array([float(x) for x in feature_str.split(',')])
    test_processed = pre_processing(features.reshape(1, -1))
    # Perform prediction using the loaded model
    prediction = model.predict(test_processed)
    
    # Return prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

# Step 4: Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)