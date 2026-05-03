import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load and preprocess dataset
def load_dataset(path):
    df = pd.read_csv(path)
    if 'FloodProbability' not in df.columns:
        raise ValueError("Missing 'FloodProbability' column")
    return df

# Split features and target
def split_features_target(df):
    X = df.drop(columns=['FloodProbability'])
    y = df['FloodProbability']
    return X, y

# Normalize and standardize features
def normalize_standardize(X):
    norm_df = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    std_df = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    return norm_df, std_df

# Evaluate multiple regression models
def evaluate_models(X, y):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor()
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nModel Comparison Summary:\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name:25s} | R2: {r2:.4f} | MSE: {mse:.4f}")

# Train model and show accuracy metrics
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nLinear Regression Accuracy Metrics:")
    print(f"R2 Score               : {r2:.4f}")
    print(f"Mean Squared Error     : {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error    : {mae:.4f}")
    return model

# User input prediction
def predict_user_input(model, X):
    print("\nEnter feature values to predict Flood Probability:")
    user_input = []
    for feature in X.columns:
        while True:
            try:
                val = float(input(f"Enter value for {feature}: "))
                user_input.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    input_df = pd.DataFrame([user_input], columns=X.columns)
    pred = model.predict(input_df)[0]
    print(f"\nPredicted Flood Probability: {pred:.4f}")

# Main Execution
if __name__ == "__main__":
    csv_path = r"data/flood.csv"
    df = load_dataset(csv_path)
    X, y = split_features_target(df)

    norm_df, std_df = normalize_standardize(X)
    print("\n🔹 Normalized Data Sample:")
    print(norm_df.head())
    print("\n🔹 Standardized Data Sample:")
    print(std_df.head())

    evaluate_models(X, y)
    model = train_and_evaluate(X, y)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/flood_model.pkl")
    print("\n✅ Model saved as models/flood_model.pkl")

    # Optional CLI test
    # predict_user_input(model, X)

