import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('../models', exist_ok=True)

def evaluate_model(model_path, X_test, y_test):
    # Load the trained model
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    # Print evaluation results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('/Users/roamsaleh/Desktop/tactiaii/data/tennis_matches_with_tactics.csv')

    # Separate features and target variable
    X = data.drop(columns=['tactical_decision'])  # Features
    y = data['tactical_decision']  # Target

    # Convert problematic columns to numeric or drop them
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)  # Replace NaN values with 0

    # Ensure consistent preprocessing
    X = pd.get_dummies(X, drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the feature names used during training
    feature_names = joblib.load('../models/feature_names.pkl')

    # Align test dataset columns with training dataset
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    # Print the size of the data to verify
    print(f"Training data size: {X_train.shape}")
    print(f"Testing data size: {X_test.shape}")

    # Evaluate the model
    model_path = '../models/model.pkl'
    evaluate_model(model_path, X_test, y_test)