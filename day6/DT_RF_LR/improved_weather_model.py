import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv('Weather Data.csv')

# Drop rows with missing values
df = df.dropna()

# Feature Engineering
# Convert Date/Time to datetime and extract useful features
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df['Hour'] = df['Date/Time'].dt.hour
df['Month'] = df['Date/Time'].dt.month
df['Day'] = df['Date/Time'].dt.day
df['DayOfWeek'] = df['Date/Time'].dt.dayofweek

# Set target and features
target = 'Weather'
X = df.drop(columns=[target, 'Date/Time'])  # Drop Date/Time as it's not useful for prediction
y = df[target]

# Encode categorical target
y_encoded = y.astype('category').cat.codes
label_mapping = dict(enumerate(y.astype('category').cat.categories))

# Encode features if needed
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define parameter grids for each model
param_grids = {
    'Random Forest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'Decision Tree': {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'Logistic Regression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['liblinear', 'saga']
    }
}

# Models with preprocessing pipeline
models = {
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k='all')),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k='all')),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k='all')),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
}

results = {}

for name, model in models.items():
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} - Cross-validation scores: {cv_scores}")
    print(f"{name} - Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': best_model,
        'accuracy': acc,
        'best_params': grid_search.best_params_
    }
    
    print(f"{name} - Best parameters: {grid_search.best_params_}")
    print(f"{name} - Test Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name}")
print(f"Best model accuracy: {results[best_model_name]['accuracy']:.3f}")

# Save the best model, features, and label mapping
joblib.dump(best_model, 'best_weather_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')
joblib.dump(label_mapping, 'label_mapping.pkl') 