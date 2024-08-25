# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Define functions for data cleaning and preprocessing
def calculate_productivity(row, threshold=0.1):
    target = row['targeted_productivity']
    actual = row['actual_productivity']

    if actual >= target * (1 + threshold):
        return 'High'
    elif actual <= target * (1 - threshold):
        return 'Low'
    else:
        return 'Neutral'
    
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

def clean_data(df):
    df['productivity'] = df.apply(calculate_productivity, axis=1)
    df = df.rename(columns={'wip': 'work_in_progress', 'smv':'std_minute_value'})
    df['work_in_progress'].fillna(df['work_in_progress'].median(), inplace=True)
    df['no_of_workers'] = df['no_of_workers'].apply(lambda x: int(x))
    df['department'] = df['department'].str.replace('sweing','sewing')
    df['department'] = df['department'].str.replace('finishing ','finishing')
    df['date'] = pd.to_datetime(df['date'])
    df.drop(columns=['day'], inplace=True)
    df['month'] = df['date'].dt.month_name()
    df['day_num'] = df['date'].dt.day
    df['day'] = df['date'].dt.day_name()
    df['quarter'] = df.quarter.str.replace('Quarter5', 'Quarter4')
    df['Quarter'] = df['quarter'].str.replace('Quarter','').astype(int)
    df.drop(columns=['quarter'], inplace=True)
    df = df.rename(columns={'Quarter': 'quarter'})
    return df

def preprocess(df):
    if 'month' not in df.columns or 'day_num' not in df.columns or 'day' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month_name()
        df['day_num'] = df['date'].dt.day
        df['day'] = df['date'].dt.day_name()

    df.drop(columns='date', inplace=True)
    day_repl_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Saturday': 4, 'Sunday': 5}
    df['day'] = df['day'].replace(day_repl_dict)
    department_repl_dict = {'sewing': 0, 'finishing': 1}
    df['department'] = df['department'].replace(department_repl_dict)
    month_repl_dict = {'January': 1, 'February': 2, 'March': 3}
    df['month'] = df['month'].replace(month_repl_dict)

    return df

# Load and clean the data
df = pd.read_csv('productivity_predictor.csv')
df = clean_data(df)

# Encode target variable
label_encoder = LabelEncoder()
df['productivity'] = label_encoder.fit_transform(df['productivity'])

# Split the data
X = df.drop('productivity', axis=1)
y = df['productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train = preprocess(X_train)
X_test = preprocess(X_test)

# Train the model
model = XGBClassifier(random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model)
])
pipeline.fit(X_train, y_train)

# Save the model and label encoder
joblib.dump(pipeline, 'productivity_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
