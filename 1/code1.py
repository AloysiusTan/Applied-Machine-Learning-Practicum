#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import joblib
from baseDefinitions import *


# In[2]:


def print_rmse(y_true, y_pred, dataset_type):
    """Utility function to compute and print RMSE"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{dataset_type} RMSE: {rmse}")


# In[3]:


data = pd.read_pickle('appml-assignment1-testing-v4.pkl')
X = data['X']
y = data['y']


# In[4]:


datetime_features = ['date']
numerical_features = [column for column in X.columns if X[column].dtype in ['int64', 'float64'] and column not in datetime_features]
categorical_features = [column for column in X.columns if X[column].dtype == 'object']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(), categorical_features),
        ('date', Pipeline(steps=[
            ('extract', DateFeaturesExtractor())
        ]), datetime_features)
    ],
)


# In[5]:


# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(max_depth = 10, min_samples_leaf = 4, min_samples_split = 2, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Predictions for train and test set
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

# Print RMSE for both train and test sets
print_rmse(y_train, y_train_pred, "Train")
print_rmse(y_test, y_test_pred, "Test")


# In[6]:


# Save the pipeline and model
joblib.dump(preprocessor, 'pipeline1.pkl')


# In[7]:


joblib.dump(model_pipeline.named_steps['regressor'], 'model1.pkl')

