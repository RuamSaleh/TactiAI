import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import os

# Load the dataset
data = pd.read_csv('/Users/roamsaleh/Desktop/tactiaii/data/tennis_matches_with_tactics.csv')

# Preprocess the data
X = data.drop(columns=['tactical_decision'])
y = data['tactical_decision']

# Separate numeric and non-numeric columns
numeric_columns = X.select_dtypes(include=['number']).columns
non_numeric_columns = X.select_dtypes(exclude=['number']).columns

# Handle missing values for numeric columns
imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(imputer.fit_transform(X[numeric_columns]), columns=numeric_columns)

# For non-numeric columns, fill missing values with a placeholder or drop them
X_non_numeric = X[non_numeric_columns].fillna('missing')

# Combine numeric and non-numeric columns back together
X = pd.concat([X_numeric, X_non_numeric], axis=1)

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Ensure the models directory exists
os.makedirs('../models', exist_ok=True)

# Save the trained model
joblib.dump(model, '../models/model.pkl')

# Save the feature names
joblib.dump(X_train.columns, '../models/feature_names.pkl')

print("Model and feature names saved successfully.")