from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# from sklearn import accuracy_score

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
encoding = pickle.load( open("label_encoding.pkl",'rb'))
# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    # Get the input values from the form
        age = int(request.form['age'])
        workclass = int(request.form['workclass'])
        education_num = int(request.form['education_num'])
        marital_status = int(request.form['marital_status'])
        occupation = int(request.form['occupation'])
        capital_gain = int(request.form['capital_gain'])
        capital_loss = int(request.form['capital_loss'])
        hours_per_week = int(request.form['hours_per_week'])
        sex = int(request.form['sex'])

    # Create a DataFrame with the input values
        input_data = pd.DataFrame([[age, workclass, education_num, marital_status, occupation,sex,
                                capital_gain, capital_loss, hours_per_week]],
                              columns=['age', 'workclass', 'education-num', 'marital-status', 'occupation','sex',
                                        'capital-gain', 'capital-loss', 'hours-per-week'])

    # Perform the same data preprocessing steps
        def hrs_edit(val):
            if val < 40:
                return 2
            elif val == 40:
                return 1
            else:
                return 0
        
        input_data['hours-per-week'] = input_data['hours-per-week'].apply(hrs_edit)


        # Make the prediction
        prediction = model.predict(input_data)
        # accuracy_score = accuracy_score(y_test,prediction)

        # Convert the prediction to the corresponding label
        if prediction[0] == 0:
            result = 'annual income is  then less 50,000 !'
        else:
            result = ' greater then 50,000 !'

        # Render the prediction result
        return render_template('index.html', result= "Predicted income  : {}".format(result))
    else :
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True,port=8000)
