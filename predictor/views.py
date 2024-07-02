from django.shortcuts import render
from .forms import ProductivityForm
import pandas as pd
import joblib
from datetime import datetime, date

# Load the models and label encoder
regression_model, regression_scaler = joblib.load('models/random_forest_regressor.pkl')
classification_model = joblib.load('models/productivity_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def home(request):
    return render(request, 'predictor/home.html')

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
    return round(predicted_productivity[0], 3)

def index(request):
    if request.method == 'POST':
        form = ProductivityForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            input_data = pd.DataFrame([data])

            actual_productivity = predict_actual_productivity(regression_model, regression_scaler, input_data)
            input_data['actual_productivity'] = actual_productivity
            input_data_classification = preprocess_input_for_classification(input_data.copy())
            predicted_productivity_class = classification_model.predict(input_data_classification)
            predicted_productivity_label = label_encoder.inverse_transform(predicted_productivity_class)

            return render(request, 'predictor/index.html', {
                'form': form,
                'current_date': datetime.now().strftime('%Y-%m-%d'),
                'predicted_actual_productivity': actual_productivity,
                'predicted_productivity_label': predicted_productivity_label[0]
            })
    else:
        form = ProductivityForm(initial={'date': date.today()})

    return render(request, 'predictor/index.html', {'form': form, 'current_date': datetime.now().strftime('%Y-%m-%d')})
