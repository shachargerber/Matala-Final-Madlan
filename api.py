from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import pickle
import joblib
from madlan_data_prep import prepare_data
app = Flask(__name__)
# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form

    City = str(data['City'])
    Area = int(data['Area'])
    hasBalcony = int(data.get('hasBalcony', 0))
    hasMamad = int(data.get('hasMamad', 0))
    hasElevator = int(data.get('hasElevator', 0))
    

    # Create a feature array
    data = {'City': City,'Area': Area,
            'hasBalcony': hasBalcony, 
              'hasMamad ': hasMamad,'hasElevator': hasElevator
            }
  
    df = pd.DataFrame(data, index=[0])
    # Make a prediction
    y_pred =round( model.predict(df)[0])

    # Return the predicted price
    return render_template('index.html', price=y_pred)

if __name__ == '__main__':
    app.run()


