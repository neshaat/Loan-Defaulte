from flask import Flask, render_template, request, jsonify
from test import load_model, predict_loan_default

app = Flask(__name__)

spark, model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        amt_income_total = float(request.form['amt_income_total'])
        amt_credit = float(request.form['amt_credit'])
        amt_annuity = float(request.form['amt_annuity'])
        days_employed = float(request.form['days_employed'])
        days_registration = float(request.form['days_registration'])
        flag_own_car = request.form['flag_own_car']
        flag_own_realty = request.form['flag_own_realty']
        test_data = (amt_income_total, amt_credit, amt_annuity, days_employed, days_registration, flag_own_car, flag_own_realty)
        result = predict_loan_default(spark, model, test_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
