from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('svc.lb')

@app.route('/')
def landing_page():
    return render_template('landing.html')

# Form page route
@app.route('/form', methods=['GET'])
def form_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Predict using the loaded model
    prediction = model.predict(final_features)
    output = prediction[0]
    
    # Return the result
    return render_template('index.html', 
                           prediction_text='Diabetes Prediction: {}'.format('Positive' if output == 1 else 'Negative'))

if __name__ == "__main__":
    app.run(debug=True)
