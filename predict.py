
import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_C=10.bin'

with open(model_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

app = Flask('diabetes')

@app.route('/predict', methods=['POST'])
def predict():
    trial_participant = request.get_json()

    X = dv.transform([trial_participant])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'diabetes_probability': float(y_pred),
        'diabetes': bool(churn)
    }

    return jsonify(result)








# trial_participant = {'pregnancies': 10,
#     'glucose': 111,
#     'blood_pressure': 70,
#     'skin_thickness': 27,
#     'insulin': 0,
#     'bmi': 27.5,
#     'diabetes_pedigree_function': 0.141,
#     'age': 40
#     }

# def predict(trial_participant):
#     X_small = dv.transform([trial_participant])

#     y_pred = model.predict_proba(pd.DataFrame(X_small)).round(3)[0,1]
#     return y_pred

# print('input ', trial_participant)
# print('diabetes probability ', y_pred)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)