from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the models and label encoder
regression_model, regression_scaler = joblib.load('random_forest_regressor.pkl')
classification_model = joblib.load('productivity_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def get_quarter_from_date(date):
    week_of_month = (date.day - 1) // 7 + 1
    if week_of_month == 1:
        return 'Quarter1'
    elif week_of_month == 2:
        return 'Quarter2'
    elif week_of_month == 3:
        return 'Quarter3'
    else:
        return 'Quarter4'

def preprocess_input_for_regression(df):
    df.drop(columns=['date'], inplace=True)
    department_repl_dict = {'sewing': 0, 'finishing': 1}
    df['department'] = df['department'].replace(department_repl_dict)
    return df

def preprocess_input_for_classification(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month_name()
    df['day_num'] = df['date'].dt.day
    df['day'] = df['date'].dt.day_name()
    df['quarter'] = df['date'].apply(get_quarter_from_date)
    df['quarter'] = df['quarter'].str.replace('Quarter', '').astype(int)
    df.drop(columns='date', inplace=True)
    day_repl_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['day'] = df['day'].replace(day_repl_dict)
    department_repl_dict = {'sewing': 0, 'finishing': 1}
    df['department'] = df['department'].replace(department_repl_dict)
    month_repl_dict = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['month'] = df['month'].replace(month_repl_dict)
    return df

def predict_actual_productivity(model, scaler, input_data):
    preprocessed_data = preprocess_input_for_regression(input_data.copy())
    scaled_data = scaler.transform(preprocessed_data)
    predicted_productivity = model.predict(scaled_data)
    return predicted_productivity

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('home.html')

@app.route('/team')
def team():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Process form data and make predictions
        date = request.form['date']
        department = request.form['department']
        team = int(request.form['team'])
        targeted_productivity = float(request.form['targeted_productivity'])
        std_minute_value = float(request.form['std_minute_value'])
        work_in_progress = int(request.form['work_in_progress'])
        over_time = int(request.form['over_time'])
        incentive = float(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = int(request.form['idle_men'])
        no_of_style_change = int(request.form['no_of_style_change'])
        no_of_workers = int(request.form['no_of_workers'])

        input_data = pd.DataFrame({
            'date': [date],
            'department': [department],
            'team': [team],
            'targeted_productivity': [targeted_productivity],
            'std_minute_value': [std_minute_value],
            'work_in_progress': [work_in_progress],
            'over_time': [over_time],
            'incentive': [incentive],
            'idle_time': [idle_time],
            'idle_men': [idle_men],
            'no_of_style_change': [no_of_style_change],
            'no_of_workers': [no_of_workers]
        })

        actual_productivity = predict_actual_productivity(regression_model, regression_scaler, input_data)
        input_data['actual_productivity'] = actual_productivity[0]
        input_data_classification = preprocess_input_for_classification(input_data.copy())
        predicted_productivity_class = classification_model.predict(input_data_classification)
        predicted_productivity_label = label_encoder.inverse_transform(predicted_productivity_class)

        return render_template('index.html', current_date=datetime.now().strftime('%Y-%m-%d'),
                               predicted_actual_productivity=round(actual_productivity[0], 3),
                               predicted_productivity_label=predicted_productivity_label[0],
                               form_data=request.form)

    return render_template('index.html', current_date=datetime.now().strftime('%Y-%m-%d'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)