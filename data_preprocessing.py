import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    data = pd.read_csv("/Users/roamsaleh/Desktop/tactiaii/data/tennis_matches_with_tactics.csv")
    return data

def clean_data(data):
    # Drop rows with missing values
    data = data.dropna()
    return data

def encode_categorical_features(data):
    label_encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    return data, label_encoders

def split_features_target(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def preprocess_data(filepath, target_column):
    data = load_data(filepath)
    data = clean_data(data)
    data, label_encoders = encode_categorical_features(data)
    X, y = split_features_target(data, target_column)
    return X, y, label_encoders