import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def load_and_preprocess_data(csv_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    print("Cleaning data...")
    # Convert TotalCharges to numeric, replace spaces with NaN, fill with 0.0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Map target Churn to binary (0/1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'Telco-Customer-Churn.csv')
    df = load_and_preprocess_data(csv_path)
    
    # Define features to use
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
        'tenure', 'PhoneService', 'InternetService', 
        'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    X = df[features]
    y = df['Churn']
    
    # Define feature types
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract', 'PaymentMethod']
    numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    
    print("Setting up preprocessing pipeline...")
    # Create ColumnTransformer with OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Define model with robust hyperparameters
    clf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=8, 
        min_samples_leaf=4, 
        class_weight='balanced', 
        random_state=42
    )
    
    # Create the complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Split the dataset
    print("Splitting dataset into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the pipeline
    print("Training Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    print("\n=== Evaluation Metrics (Test Set) ===")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = pipeline.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the pipeline
    model_path = os.path.join(base_dir, 'trained_model.pkl')
    print(f"\nSaving model pipeline to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Model pipeline saved successfully!")

if __name__ == '__main__':
    train_model()
