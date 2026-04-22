
# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score



# LOAD DATA

def load_data():
    df = pd.read_csv("weather.csv")
    df = df.dropna().drop_duplicates()
    return df



# PREPROCESS DATA

def preprocess_data(df):
    le = LabelEncoder()
    df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
    df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])
    return df


# RAIN MODEL + GRAPH
def rain_model(df):
    features = ['MinTemp', 'MaxTemp', 'WindGustDir',
                'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']

    X = df[features]
    y = df['RainTomorrow']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Rain Prediction Accuracy: {acc:.2f}")

    #  Feature Importance Graph
    importance = model.feature_importances_
    plt.figure()
    plt.bar(features, importance)
    plt.title("Feature Importance (Rain Prediction)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return model


# REGRESSION MODEL + GRAPH

def regression_model(df, column):
    X, y = [], []

    for i in range(len(df) - 1):
        X.append(df[column].iloc[i])
        y.append(df[column].iloc[i + 1])

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"{column} Prediction R2 Score: {r2:.2f}")

    #  Actual vs Predicted Graph
    plt.figure()
    plt.plot(y_test[:50], label="Actual")
    plt.plot(y_pred[:50], label="Predicted")
    plt.title(f"{column} Prediction (Actual vs Predicted)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


# MAIN

def main():
    df = load_data()
    df = preprocess_data(df)

    print("\n===== TRAINING MODELS =====\n")

    rain_model(df)
    regression_model(df, 'Temp')
    regression_model(df, 'Humidity')


# RUN
if __name__ == "__main__":
    main()
