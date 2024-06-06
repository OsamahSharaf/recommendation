#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


df = pd.read_csv('new_df_recomen.csv')


# In[3]:


df


# In[ ]:


# Normalization functions for numerical data
def standard_scale(value):
    scaler = StandardScaler()
    scaled_value = scaler.fit_transform(np.array(value).reshape(-1, 1)).flatten()
    return scaled_value

def min_max_scale(value):
    scaler = MinMaxScaler()
    scaled_value = scaler.fit_transform(np.array(value).reshape(-1, 1)).flatten()
    return scaled_value

def log_transform(value):
    # Replace invalid values with a small positive constant
    value = np.where(value <= 0, 1e-10, value)
    transformed_value = np.log1p(value)
    return transformed_value

def get_normalization_method(values, column_name):
    values = np.array(values)
    print(f"Analyzing column: {column_name}")
    print(f"Mean: {np.mean(values)}, Median: {np.median(values)}, Std: {np.std(values)}, Min: {np.min(values)}, Max: {np.max(values)}")
    if np.any(values <= 0):
        print(f"Using log_transform for column: {column_name}")
        return log_transform, 'log_transform'
    elif np.abs(np.mean(values) - np.median(values)) / np.std(values) < 0.1:
        print(f"Using standard_scale for column: {column_name}")
        return standard_scale, 'standard_scale'
    else:
        print(f"Using min_max_scale for column: {column_name}")
        return min_max_scale, 'min_max_scale'


# In[ ]:


# Normalization function for text data using TF-IDF
def normalize_text_data(text_data):
    text_data = text_data.fillna('')  # Replace NaN values with empty string
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    return tfidf_df, 'TF-IDF'

# Function to normalize both numerical and text data
def normalize_data(df):
    new_df = pd.DataFrame()
    normalization_methods = {}
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:  # Numerical columns
            values = df[col].values
            normalization_method, method_name = get_normalization_method(values, col)
            if normalization_method:
                new_df[col] = normalization_method(values)
                normalization_methods[col] = method_name
        elif df[col].dtype == 'object':  # Text columns
            tfidf_df, method_name = normalize_text_data(df[col])
            tfidf_df = tfidf_df.add_prefix(col + '_')
            new_df = pd.concat([new_df, tfidf_df], axis=1)
            normalization_methods[col] = method_name
        else:
            new_df[col] = df[col]
            normalization_methods[col] = 'none'
    
    return new_df, normalization_methods


# In[39]:


# Normalize the data
normalized_df, normalization_methods = normalize_data(df)

# Save the normalized data back to a CSV file
# normalized_df.to_csv('normalized_new_df_recom.csv', index=False)

print("Data normalization complete.")
print("Normalization methods used for each column:")
for col, method in normalization_methods.items():
    print(f"{col}: {method}")


# In[11]:


# normalized_data


# In[40]:


normalized_df


# In[ ]:





# In[ ]:




