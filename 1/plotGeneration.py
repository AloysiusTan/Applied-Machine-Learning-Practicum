#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from baseDefinitions import *


# In[2]:


# Define a custom function for RMS
def rms(x):
    return np.sqrt(np.mean(x**2))


# In[3]:


# Load the data and the pipeline
data = pd.read_pickle('appml-assignment1-testing-v4.pkl')
pipeline = joblib.load('pipeline1.pkl')
X = data['X']
y = data['y']
X_processed = pipeline.transform(X)


# In[4]:


# Add the date features extracted from the pipeline to X
df_processed = pd.DataFrame(X_processed, columns=pipeline.get_feature_names_out())

# Adding the next hour's high from y to the DataFrame
df_processed['Next_Hour_High'] = y

# Calculate the difference using the 'USD.CAD-MIDPOINT-close'
df_processed['difference'] = df_processed['num__USD.TRY-BID-high'].shift(-1) - df_processed['num__USD.TRY-BID-close']

# Calculate the total hours of the week for each row
df_processed['total_hour_of_week'] = df_processed['date__day_of_week'] * 24 + df_processed['date__hour_of_day']
df_processed


# In[5]:


stats = df_processed.groupby(['total_hour_of_week']).agg({
    'difference': [
        ('mean', 'mean'),
        ('p5', lambda x: np.percentile(x, 5)),
        ('p10', lambda x: np.percentile(x, 10)),
        ('p25', lambda x: np.percentile(x, 25)),
        ('median', 'median'),
        ('p75', lambda x: np.percentile(x, 75)),
        ('p90', lambda x: np.percentile(x, 90)),
        ('p95', lambda x: np.percentile(x, 95)),
        ('rms', rms) 
    ]
}).reset_index()
stats


# In[6]:


# Create the figure and axis
plt.figure(figsize=(15, 8))

# Loop through each column in the DataFrame except for the 'total_hour_of_week' to plot
for column in stats.columns[1:]:
    plt.plot(stats['total_hour_of_week'], stats[column], label=column)

plt.title('Statistical Metrics for Each Hour of the Week')
plt.xlabel('Total Hour of the Week')
plt.ylabel('Metric Values')

plt.legend(title='Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('cad-change-stats.png')
# Show the plot
plt.show()

