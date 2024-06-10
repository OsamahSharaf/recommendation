#!/usr/bin/env python
# coding: utf-8

# In[29]:


# numpy==1.24.3
# pandas==1.5.3
# scikit-learn==1.3.0

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def standard_scale(values):
    """
    Normalization function for numerical data using standard scaling.
    It creates a StandardScaler object, fits and transforms the values, then flattens the result.
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
    return scaled_values

def min_max_scale(values):
    """
    Normalization function for numerical data using min-max scaling.
    It creates a MinMaxScaler object, fits and transforms the values, then flattens the result.
    """
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
    return scaled_values

def log_transform(values):
    """
    Normalization function for numerical data using log transformation.
    It replaces non-positive values with a small positive constant and applies log1p transformation.
    """
    values = np.where(values <= 0, 1e-10, values)
    transformed_value = np.log1p(values)
    return transformed_value

def tfidf_transform(values):
    """
    Normalization function for text data using TF-IDF.
    It creates a TfidfVectorizer object, fits and transforms the text values to a TF-IDF matrix,
    gets the feature names, converts the matrix to a DataFrame, and then returns the values.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(values)
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names).values

def get_numerical_normalization_method(values):
    """
    Function to determine the appropriate normalization method for numerical data.
    It checks if any values are non-positive, if the distribution of values is approximately normal,
    and returns the corresponding normalization method.
    """
    if np.any(values <= 0):
        return log_transform
    elif np.abs(np.mean(values) - np.median(values)) / np.std(values) < 0.1:
        return standard_scale
    else:
        return min_max_scale

def normalize_data(input_data):
    """
    Main function to normalize both numerical and text data.
    It separates numerical and text data, applies the appropriate normalization methods,
    and updates the input data with normalized values.
    """
    # Separate numerical and text data
    numerical_values = [feature["value"] for feature in input_data if feature["type"] == "numerical"]
    text_values = [feature["value"] for feature in input_data if feature["type"] == "text"]

    # Normalize numerical data
    if numerical_values:
        numerical_values = np.array(numerical_values, dtype=float)
        normalization_method = get_numerical_normalization_method(numerical_values)
        normalized_numerical_values = normalization_method(numerical_values)
    
    # Normalize text data
    if text_values:
        normalized_text_values = tfidf_transform(text_values)

    # Update input data with normalized values
    normalized_data = []
    num_index = 0
    text_index = 0
    for feature in input_data:
        if feature["type"] == "numerical":
            feature["value"] = normalized_numerical_values[num_index]
            num_index += 1
        elif feature["type"] == "text":
            feature["value"] = normalized_text_values[text_index]
            text_index += 1
        normalized_data.append(feature)

    return normalized_data

# Example usage
input_data = [
    {"value": 123456, "type": "numerical"},
    {"value": 500, "type": "numerical"},
    {"value": "Some Text", "type": "text"},
    {"value": "Example", "type": "text"}
]

# Normalize the input data
normalized_data = normalize_data(input_data)
normalized_data





# In[ ]:




