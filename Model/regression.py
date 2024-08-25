import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

def clean_data(df):
    df = df.rename(columns={'wip': 'work_in_progress', 'smv':'std_minute_value'})

    # Filling in missing values with median for the 'work_in_progress' column
    if 'work_in_progress' in df.columns:
        df['work_in_progress'].fillna(df['work_in_progress'].median(), inplace=True)

    # no of workers should be whole number
    df['no_of_workers'] = df['no_of_workers'].apply(lambda x: int(x))

    # replacing quarter5 (given to jan days above 28) with quarter4
    df['quarter'] = df.quarter.str.replace('Quarter5', 'Quarter4')

    # Removing the word Quarter from the quarter column and leave the numbers
    df['quarter'] = df['quarter'].str.replace('Quarter','')

    # Changing the datatype to numeric
    df['quarter'] = df['quarter'].astype(int)

    # Correcting the spelling of sewing in the department column
    df['department'] = df['department'].str.replace('sweing','sewing')

    # Removing the spacing from the word finishing in the department column
    df['department'] = df['department'].str.replace('finishing ','finishing')

    # Dropping unnecessary columns
    df.drop(columns=['date', 'quarter', 'day'], inplace=True)

    return df

def preprocess(X):
    # Creating a copy of the dataframe
    df_enc = X.copy()

    # Encoding department column
    department_repl_dict = {'sewing': 0, 'finishing': 1}
    df_enc['department'] = df_enc['department'].replace(department_repl_dict)

    return df_enc

def train_random_forest(X_train, X_test, y_train, y_test):
    # Preprocess the data
    X_train_preprocessed = preprocess(X_train)
    X_test_preprocessed = preprocess(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_preprocessed)
    X_test_scaled = scaler.transform(X_test_preprocessed)

    # Define the Random Forest hyperparameters
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 9]
    }

    # Initialize the Random Forest model
    rf = RandomForestRegressor(random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pct_diff = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Print the results
    print(f"Best Params: {grid_search.best_params_}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"Percentage Difference: {pct_diff}")

    return best_model, scaler

# Loading the dataset
df = pd.read_csv('productivity_predictor.csv')

# Clean the dataset
df = clean_data(df)

# Define X and y
y = df['actual_productivity']
X = df.drop(columns=['actual_productivity'])

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the Random Forest model
best_model, scaler = train_random_forest(X_train, X_test, y_train, y_test)

# Save the trained model and scaler
with open('random_forest_regressor.pkl', 'wb') as f:
    pickle.dump((best_model, scaler), f)

print("Model and scaler saved to random_forest_regressor.pkl")
