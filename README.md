# Productivity Predictor

![GitHub repo size](https://img.shields.io/github/repo-size/VarshaLenin12/Productivity_Predictor)
![GitHub stars](https://img.shields.io/github/stars/VarshaLenin12/Productivity_Predictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/VarshaLenin12/Productivity_Predictor?style=social)

Welcome to the Productivity Predictor project! This project utilizes machine learning to predict productivity in the garments industry. The goal is to provide accurate predictions based on various input features, helping industry professionals make informed decisions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview
The Productivity Predictor is a machine learning application designed to predict the productivity levels in the garments industry. The project includes a web interface built with Django and Flask, allowing users to input relevant features and receive productivity predictions.

### Live Demo
Check out the live demo of the project [here](https://productivity-predictor.onrender.com/).

## Features
- 📈 Predict productivity based on input features.
- 💻 User-friendly web interface for data input and prediction results.

## Installation
To get started with the Productivity Predictor, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/VarshaLenin12/Productivity_Predictor.git
    cd Productivity_Predictor
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```sh
    python app.py
    ```

## Usage
Once the application is running, you can access the web interface at `http://127.0.0.1:8000`. Input the required features and click on the "Predict" button to get the productivity prediction.

## Model
The machine learning models used in this project include a regression model and a classification model:

- **Regression Model**: Random Forest Regressor
- **Classification Model**: XGBoost

### Key Metrics:
- **Accuracy**: High accuracy in predictions.
- **R-squared Value**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Features**: Various factors influencing productivity.

## Technologies Used
- **Python**: Core programming language.
- **Flask**: Web framework for frontend and backendw.
- **Scikit-Learn**: Machine learning library.
- **XGBoost**: Extreme Gradient Boosting for classification.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Matplotlib**: Data visualization.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request.

## Contact
If you have any questions or want to connect, feel free to reach out:

- **Email**: [varshalenin999@gmail.com](mailto:varshalenin999@gmail.com)
- **LinkedIn**: [Varsha L](https://www.linkedin.com/in/varsha-l-ml)

Thank you for visiting the Productivity Predictor project! Feel free to explore, use, and contribute to this project.
