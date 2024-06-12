from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DateFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ext = X.copy()
        X_ext['day_of_week'] = X_ext['date'].dt.dayofweek
        X_ext['hour_of_day'] = X_ext['date'].dt.hour
        
        # Drop the original 'date' column
        X_ext.drop('date', axis=1, inplace=True)
        
        return X_ext

    def get_feature_names_out(self, input_features=None):
        return ['day_of_week', 'hour_of_day']

    
